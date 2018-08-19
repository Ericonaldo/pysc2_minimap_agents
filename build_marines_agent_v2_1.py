import random
import math
import os.path

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_SCV_SUPPLY = 6
_PLAYER_BACKGROUND = 0


# added by taylor
_COLLECTED_MINERALS = 8 # 当前收集到的水晶数量
_FOOD_USED = 3 # 当前人口
_FOOD_CAP = 4  # 当前人口上限
_ARMY_COUNT = 8 # 当前军队总数

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45 
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_NEUTRAL_MINERAL_FIELD = 341 # for detecting the location of mineral patches

_NOT_QUEUED = [0]  # 现在执行
_QUEUED = [1]      # 延迟执行
_SELECT_ALL = [2]

DATA_FILE = 'sparse_agent_data_v2_1'

ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_BUILD_SCV = 'buildscv'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MARINE,
    ACTION_BUILD_SCV,
]

MAX_SCV_NUM = 17 # 最大的SCV数量
POPULATION_THRESHOLD_ENABLE_BUILD_SUPPLY_DEPOT = 15 # 当可用人口数低于阈值时，建立补给站的动作有效


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
            
        return action

    def learn(self, s, a, r, s_):
        if s == s_:
            return

        self.check_state_exist(s_)
        self.check_state_exist(s)
        
        q_predict = self.q_table.ix[s, a]
        
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        else:
            q_target = r  # next state is terminal
            
        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

class SparseAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SparseAgent, self).__init__()
        
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        
        self.previous_action = None
        self.previous_state = None
        
        # keep track of the command centre location
        self.cc_y = None 
        self.cc_x = None
        
        # track the sequence position within a multi-step action
        self.move_number = 0
        self.last_army = 0
        
        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')
        
    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        
        return [x + x_distance, y + y_distance]
    
    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [64 - x, 64 - y]
        
        return [x, y]
    
    def splitAction(self, action_id):
        smart_action = smart_actions[action_id]
            
        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return (smart_action, x, y)
        
    def step(self, obs):
        super(SparseAgent, self).step(obs)

        if obs.last():       
            current_army = obs.observation['player'][_ARMY_COUNT]
            reward = current_army - self.last_army
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')
            
            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
            
            self.previous_action = None
            self.previous_state = None
            
            self.move_number = 0
            self.last_army = 0
            
            return actions.FunctionCall(_NO_OP, [])
        
        unit_type = obs.observation['screen'][_UNIT_TYPE]

        if obs.first():
            player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        
            self.cc_y, self.cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

        cc_y, cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
        cc_count = 1 if cc_y.any() else 0
        
        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        supply_depot_count = int(round(len(depot_y) / 69))

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        barracks_count = int(round(len(barracks_y) / 137))
            
        if self.move_number == 0:
            self.move_number += 1
            
            # current_state = np.zeros(8)
            current_state = np.zeros(7)
            current_state[0] = cc_count
            current_state[1] = supply_depot_count
            current_state[2] = barracks_count
            current_state[3] = obs.observation['player'][_ARMY_SUPPLY]

            # TODO(by taylor) - 状态添加：当前人口数/人口容量，矿物的数量
            current_state[4] = obs.observation['score_cumulative'][_COLLECTED_MINERALS] # 当前收集到的水晶数量
            current_state[5] = obs.observation['player'][_FOOD_USED] # 当前人口
            current_state[6] = obs.observation['player'][_FOOD_CAP]  # 当前人口上限
    
            if self.previous_action is not None:
                current_army = obs.observation['player'][_ARMY_COUNT]
                reward = current_army - self.last_army
                self.last_army = current_army

                self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))
        
            rl_action = self.qlearn.choose_action(str(current_state))

            self.previous_state = current_state
            self.previous_action = rl_action
        
            smart_action, x, y = self.splitAction(self.previous_action)
            
            if smart_action == ACTION_BUILD_BARRACKS:
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
                    
                if unit_y.any():
                    i = random.randint(0, len(unit_y) - 1)
                    target = [unit_x[i], unit_y[i]]
                    
                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

            elif smart_action == ACTION_BUILD_SUPPLY_DEPOT and obs.observation['player'][_FOOD_CAP] - obs.observation['player'][_FOOD_USED] < POPULATION_THRESHOLD_ENABLE_BUILD_SUPPLY_DEPOT:
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
                    
                if unit_y.any():
                    i = random.randint(0, len(unit_y) - 1)
                    target = [unit_x[i], unit_y[i]]
                    
                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
                
            elif smart_action == ACTION_BUILD_MARINE:
                if barracks_y.any():
                    i = random.randint(0, len(barracks_y) - 1)
                    target = [barracks_x[i], barracks_y[i]]
            
                    return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])

            elif smart_action == ACTION_BUILD_SCV and obs.observation['player'][_SCV_SUPPLY] < MAX_SCV_NUM: # 最大SCV数不超过17个
                target = [round(self.cc_x.mean()), round(self.cc_y.mean())] # 指挥中心的位置

                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

        
        elif self.move_number == 1:
            self.move_number += 1
            
            smart_action, x, y = self.splitAction(self.previous_action)
                
            if smart_action == ACTION_BUILD_SUPPLY_DEPOT and obs.observation['player'][_FOOD_CAP] - obs.observation['player'][_FOOD_USED] < POPULATION_THRESHOLD_ENABLE_BUILD_SUPPLY_DEPOT:

                if _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                    blank_y, blank_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_BACKGROUND).nonzero()                        
                    if blank_y.any():
                        i = random.randint(0, len(blank_y) - 1)                           
                        m_x = blank_x[i]
                        m_y = blank_y[i]                            
                        target = [int(m_x), int(m_y)]
                        return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])
            
            elif smart_action == ACTION_BUILD_BARRACKS:

                if _BUILD_BARRACKS in obs.observation['available_actions']:
                    blank_y, blank_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_BACKGROUND).nonzero()                        
                    if blank_y.any():
                        i = random.randint(0, len(blank_y) - 1)                            
                        m_x = blank_x[i]
                        m_y = blank_y[i]                            
                        target = [int(m_x), int(m_y)]    
                        return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])
    
            elif smart_action == ACTION_BUILD_MARINE:
                if _TRAIN_MARINE in obs.observation['available_actions']:
                    return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

            elif smart_action == ACTION_BUILD_SCV and obs.observation['player'][_SCV_SUPPLY] < MAX_SCV_NUM:
                if _TRAIN_SCV in obs.observation['available_actions']:
                    return actions.FunctionCall(_TRAIN_SCV, [_QUEUED])
                
        elif self.move_number == 2:
            self.move_number = 0
            
            smart_action, x, y = self.splitAction(self.previous_action)
                
            if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT or (smart_action == ACTION_BUILD_SCV and obs.observation['player'][_SCV_SUPPLY] < MAX_SCV_NUM):
                if _HARVEST_GATHER in obs.observation['available_actions']:
                    unit_y, unit_x = (unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()
                    
                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)
                        
                        m_x = unit_x[i]
                        m_y = unit_y[i]
                        
                        target = [int(m_x), int(m_y)]
                        
                        return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])
        
        return actions.FunctionCall(_NO_OP, [])
