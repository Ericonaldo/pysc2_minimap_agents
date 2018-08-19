# Chinese is noted by talor, eric
import random
import math
import os.path

import numpy as np
import pandas as pd
import tensorflow as tf
import collections

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


_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45 
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_NEUTRAL_MINERAL_FIELD = 341 # for detecting the location of mineral patches

_NOT_QUEUED = [0]  # 现在执行
_QUEUED = [1]      # 延迟执行
_SELECT_ALL = [2]

DATA_FILE = 'sparse_agent_data_dqn_v3'
LOG_DIR = 'logs/'
WEIGHT_DIR = 'weights/'

ACTION_DO_NOTHING = 0 # 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 1 # 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 2 # 'buildbarracks'
ACTION_BUILD_MARINE = 3 # 'buildmarine'
ACTION_BUILD_SCV = 4 # 'buildscv'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MARINE,
    ACTION_BUILD_SCV,
]

MAX_SCV_NUM = 17 # 最大的SCV数量
POPULATION_THRESHOLD_ENABLE_BUILD_SUPPLY_DEPOT = 15 # 当可用人口数低于阈值时，建立补给站的动作有效

# added by eric
STATE = np.zeros(7) # 状态是7维
MEMORY_SIZE = 10000 # 记忆库
BATCH_SIZE = 32
UPDATE_PERIOD = 200 # 更新频率
DECAY_EPS = 200 # epsilon decay频率

class DeepQNetwork():
    def __init__(self, sess=None, gamma = 0.8, epsilon=0.9):
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = len(smart_actions)
        self.state_dim = len(STATE)
        self.network()
        self.step = tf.Variable(0, trainable=False)
        self.sess = sess
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        if os.path.isdir(DATA_FILE+'/'+WEIGHT_DIR):
            self.saver.restore(self.sess, tf.train.latest_checkpoint(DATA_FILE+'/'+WEIGHT_DIR))

        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(DATA_FILE+'/'+LOG_DIR, graph=sess.graph)
        self.summary_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), sess.run(self.step))

    def net_frame(self, scope, collections_name, num_actions, inputs):
        "basic net frame"
        weights_init = tf.truncated_normal_initializer(0, 0.3)
        bias_init = tf.constant_initializer(0.1)

        with tf.variable_scope(scope):
            with tf.variable_scope("layer1"):
                weights1 = tf.get_variable(name = "weights", dtype=tf.float32, shape=[self.state_dim, 64], initializer=weights_init, collections=collections_name)
                bias1 = tf.get_variable(name = "bias", dtype=tf.float32, shape=[64], initializer=bias_init, collections=collections_name)
                wx_b = tf.matmul(inputs, weights1) + bias1
                h1 = tf.nn.relu(wx_b)            
            with tf.variable_scope("layer2"):
                weights2 = tf.get_variable(name = "weights", dtype=tf.float32, shape=[64, 64], initializer=weights_init, collections=collections_name)
                bias2 = tf.get_variable(name = "bias", dtype=tf.float32, shape=[64], initializer=bias_init, collections=collections_name)
                wx_b = tf.matmul(h1, weights2) + bias2
                h2 = tf.nn.relu(wx_b)
            
            with tf.variable_scope("layer3"):
                weights3 = tf.get_variable(name = "weights", dtype=tf.float32, shape=[64, num_actions], initializer=weights_init, collections=collections_name)
                bias3 = tf.get_variable(name = "bias", dtype=tf.float32, shape=[num_actions], initializer=bias_init, collections=collections_name)
                q_out = tf.matmul(h2, weights3) + bias3
            
            return q_out            
    
    def network(self):
        "networks"
        # q_network
        self.inputs_q = tf.placeholder(dtype = tf.float32, shape = [None, self.state_dim], name = "inputs_q")
        scope_var = "q_network"
        clt_name_var = ["q_net_prmt", tf.GraphKeys.GLOBAL_VARIABLES] # 定义了collections
        self.q_value = self.net_frame(scope_var, clt_name_var, self.action_dim, self.inputs_q)
    
        # target_network
        self.inputs_target = tf.placeholder(dtype = tf.float32, shape = [None, self.state_dim], name = "inputs_target")
        scope_var = "target_network"
        clt_name_var = ["target_net_prmt", tf.GraphKeys.GLOBAL_VARIABLES] # 定义了collections
        self.q_target = self.net_frame(scope_var, clt_name_var, self.action_dim, self.inputs_target)

        with tf.variable_scope("loss"):
            self.target = tf.placeholder(dtype = tf.float32, shape = [None, self.action_dim], name="target")
            self.loss = tf.reduce_mean(tf.square(self.q_value - self.target))

        with tf.variable_scope("train"):
            self.train_op = tf.train.RMSPropOptimizer(0.01).minimize(self.loss)
    
    def learn(self, state, action, reward, state_next, done, step):
        "train process"
        if step % 500 == 0:
            self.saver.save(self.sess, DATA_FILE+'/'+WEIGHT_DIR, global_step=self.step)
        
        q, q_target = self.sess.run([self.q_value, self.q_target], feed_dict={self.inputs_q: state, self.inputs_target: state_next})
        target = reward + self.gamma * np.max(q_target, axis=1)

        self.reform_target = q.copy()
        batch_index = np.arange(BATCH_SIZE, dtype = np.int32)
        self.reform_target[batch_index, action] = target

        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.inputs_q:state, self.target:self.reform_target})

    def update_prmt(self):
        "update target network parameters"
        q_prmts = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "q_network")
        target_prmts = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "target_network")
        self.sess.run([tf.assign(t, q) for t,q in zip(target_prmts, q_prmts)]) #将Q网络参数赋值给target
        print("updating target-network parameters...")
    
    def choose_action(self, current_state):
        current_state = current_state[np.newaxis, :]
        # array dim : (xx, ) --> (1, xx)
        q = self.sess.run(self.q_value, feed_dict={self.inputs_q: current_state})

        # e-greedy
        if np.random.random() < self.epsilon:
            action_chosen = np.random.randint(0, self.action_dim)
        else:
            action_chosen = np.argmax(q)
        return action_chosen
    
    def decay_epsilon(self):
        pass
        # if self.epsilon > 0.03:
        #     self.epsilon -= 0.02

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class SparseAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SparseAgent, self).__init__()
        
        self.memory = []#memory for memory replay
        self.global_step = 0
        self.sess = tf.Session()
        self.DQN = DeepQNetwork(self.sess)
        
        self.previous_action = None
        self.previous_state = None
        
        # keep track of the command centre location
        self.cc_y = None 
        self.cc_x = None
        
        # track the sequence position within a multi-step action
        self.move_number = 0
        

    def __del__(self):
        self.sess.close()

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
 
        if obs.last(): # done
            reward = obs.reward
            
            '''
            #--------train
            if len(self.memory) > MEMORY_SIZE:
                self.memory.pop(0)
            self.memory.append(Transition(self.previous_state, self.previous_action, reward, [-1], float(obs.last())))
            if len(self.memory) > BATCH_SIZE * 4:
                batch_trasition = random.sample(self.memory, BATCH_SIZE)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = map(np.array, zip(*batch_trasition))
                self.DQN.learn(batch_state, batch_action, batch_reward, batch_next_state, batch_done, self.global_step)
                self.global_step += 1
                print("trained",self.global_step)

            if self.global_step % UPDATE_PERIOD == 0:
                self.DQN.update_prmt()

            if self.global_step % DECAY_EPS == 0:
                self.DQN.decay_epsilon()
            #--------train
            '''
           
            self.previous_action = None
            self.previous_state = None
            
            self.move_number = 0
            
            return actions.FunctionCall(_NO_OP, [])
        
        unit_type = obs.observation['screen'][_UNIT_TYPE]

        if obs.first(): # first state = reset
            player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0 # 有己方单位且在地图上方
        
            self.cc_y, self.cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero() # 人族基地

        cc_y, cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
        cc_count = 1 if cc_y.any() else 0
        
        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero() # 人族供给站
        supply_depot_count = int(round(len(depot_y) / 69)) # 69应该是建筑宽度

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero() # 人族兵营
        barracks_count = int(round(len(barracks_y) / 137)) # 137也是建筑宽度
        
        # move_number 表示连续实现的动作序列的index    
        if self.move_number == 0: # 该做第一个动作了
            self.move_number += 1
            
            # current_state = np.zeros(8)
            current_state = STATE
            current_state[0] = cc_count
            current_state[1] = supply_depot_count
            current_state[2] = barracks_count
            current_state[3] = obs.observation['player'][_ARMY_SUPPLY]

            # TODO(by taylor) - 状态添加：当前人口数/人口容量，矿物的数量
            current_state[4] = obs.observation['score_cumulative'][_COLLECTED_MINERALS] # 当前收集到的水晶数量
            # print("minerals:",current_state[4])
            current_state[5] = obs.observation['player'][_FOOD_USED] # 当前人口
            current_state[6] = obs.observation['player'][_FOOD_CAP]  # 当前人口上限
    
            if self.previous_action is not None:
                #--------train
                if len(self.memory) > MEMORY_SIZE:
                    self.memory.pop(0)
                self.memory.append(Transition(self.previous_state, self.previous_action, 0, current_state, float(obs.last())))
                if len(self.memory) > BATCH_SIZE * 4:
                    batch_trasition = random.sample(self.memory, BATCH_SIZE)
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done = map(np.array, zip(*batch_trasition))
                    self.DQN.learn(batch_state, batch_action, batch_reward, batch_next_state, batch_done, self.global_step)
                    self.global_step += 1
        
                if self.global_step and self.global_step % UPDATE_PERIOD == 0:
                    self.DQN.update_prmt()

                if self.global_step and self.global_step % DECAY_EPS == 0:
                    self.DQN.decay_epsilon()
                #--------train
        
            rl_action = self.DQN.choose_action(current_state) # 选择动作

            self.previous_state = current_state
            self.previous_action = rl_action
        
            # smart_action, x, y = self.splitAction(self.previous_action)
            smart_action = rl_action
            # print("action:", smart_action)
            
            if smart_action == ACTION_BUILD_BARRACKS: # 造兵营， 第一步先选中SCV
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
                if unit_y.any():
                    i = random.randint(0, len(unit_y) - 1)
                    target = [unit_x[i], unit_y[i]]
                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target]) # 选中SCV

            elif smart_action == ACTION_BUILD_SUPPLY_DEPOT and obs.observation['player'][_FOOD_CAP] - obs.observation['player'][_FOOD_USED] < POPULATION_THRESHOLD_ENABLE_BUILD_SUPPLY_DEPOT:
                # 造补给站，第一步先选中SCV
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
                    
                if unit_y.any(): # 有SCV
                    i = random.randint(0, len(unit_y) - 1) # 随机选择SCV
                    target = [unit_x[i], unit_y[i]]
                    
                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
                
            elif smart_action == ACTION_BUILD_MARINE: # 造marine，第一步先选中兵营
                if barracks_y.any():
                    i = random.randint(0, len(barracks_y) - 1) # 兵营位置，取中心
                    target = [barracks_x[i], barracks_y[i]]
            
                    return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])

            elif smart_action == ACTION_BUILD_SCV and obs.observation['player'][_SCV_SUPPLY] < MAX_SCV_NUM: # 造SCV，最大SCV数不超过17个
                target = [round(self.cc_x.mean()), round(self.cc_y.mean())] # 指挥中心的位置

                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

        
        elif self.move_number == 1: # 该做第二个动作，就不用重新选择动作了
            self.move_number += 1
            
            # smart_action, x, y = self.splitAction(self.previous_action)
            smart_action = self.previous_action
                
            if smart_action == ACTION_BUILD_SUPPLY_DEPOT and obs.observation['player'][_FOOD_CAP] - obs.observation['player'][_FOOD_USED] < POPULATION_THRESHOLD_ENABLE_BUILD_SUPPLY_DEPOT:
                # 造补给站，第二步开始造兵营
                
                if _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                    
                    blank_y, blank_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_BACKGROUND).nonzero() # 随机选择集合点                 
                    if blank_y.any():
                        i = random.randint(0, len(blank_y) - 1)                           
                        m_x = blank_x[i]
                        m_y = blank_y[i]                            
                        target = [int(m_x), int(m_y)]
                        return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])
            
            elif smart_action == ACTION_BUILD_BARRACKS: # 造兵营，第二步开始造兵营
                if _BUILD_BARRACKS in obs.observation['available_actions']:
                    blank_y, blank_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_BACKGROUND).nonzero()
                    if blank_y.any():
                        i = random.randint(0, len(blank_y) - 1)                            
                        m_x = blank_x[i]
                        m_y = blank_y[i]                            
                        target = [int(m_x), int(m_y)]    
                        return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])
    
            elif smart_action == ACTION_BUILD_MARINE: # 造marine，第二步开始造marine
                if _TRAIN_MARINE in obs.observation['available_actions']:
                    return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

            elif smart_action == ACTION_BUILD_SCV and obs.observation['player'][_SCV_SUPPLY] < MAX_SCV_NUM: # 造SCV，第二步开始造SCV
                if _TRAIN_SCV in obs.observation['available_actions']:
                    return actions.FunctionCall(_TRAIN_SCV, [_QUEUED])
                
        elif self.move_number == 2: # 该做第三个动作，不用重复选动作
            self.move_number = 0 # 先清零
            
            # smart_action, x, y = self.splitAction(self.previous_action)
            smart_action = self.previous_action
                
            if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT or (smart_action == ACTION_BUILD_SCV and obs.observation['player'][_SCV_SUPPLY] < MAX_SCV_NUM): # 造兵营和造补给站，第三步命令SCV采矿 
                if _HARVEST_GATHER in obs.observation['available_actions']:
                    unit_y, unit_x = (unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()
                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)
                        
                        m_x = unit_x[i]
                        m_y = unit_y[i]
                        
                        target = [int(m_x), int(m_y)]
                        
                        return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])
        
        return actions.FunctionCall(_NO_OP, []) # 其他都不做
