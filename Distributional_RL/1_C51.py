###########################################################################################
# Implementation of C51
# Author for codes: Chu Kun(chukun1997@163.com)
# Paper: https://arxiv.org/abs/1707.06887v1
# Reference: https://github.com/sungyubkim/Deep_RL_with_pytorch
###########################################################################################
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from replay_memory import ReplayBuffer, PrioritizedReplayBuffer

import random
import os
import pickle
import time
from collections import deque
import matplotlib.pyplot as plt
from wrappers import wrap, wrap_cover, SubprocVecEnv

# Parameters
import argparse
parser = argparse.ArgumentParser(description='Some settings of the experiment.')
parser.add_argument('games', type=str, nargs=1, help='name of the games. for example: Breakout')
args = parser.parse_args()
args.games = "".join(args.games)

'''DQN settings'''
# sequential images to define state
STATE_LEN = 4
# target policy sync interval
TARGET_REPLACE_ITER = 1
# simulator steps for start learning
LEARN_START = int(1e+3)
# (prioritized) experience replay memory size
MEMORY_CAPACITY = int(1e+5)
# simulator steps for learning interval
LEARN_FREQ = 1
# atom number. default is C51 algorithm
N_ATOM = 51


'''Environment Settings'''
# number of environments for C51
N_ENVS = 16
# openai gym env name
ENV_NAME = args.games+'NoFrameskip-v4'
env = SubprocVecEnv([wrap_cover(ENV_NAME) for i in range(N_ENVS)])
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape
# prior knowledge of return distribution, 
V_MIN = -5.
V_MAX = 10.
V_RANGE = np.linspace(V_MIN, V_MAX, N_ATOM)
V_STEP = ((V_MAX-V_MIN)/(N_ATOM-1))
# Total simulation step
STEP_NUM = int(1e+8)
# gamma for MDP
GAMMA = 0.99
# visualize for agent playing
RENDERING = False

'''Training settings'''
# check GPU usage
USE_GPU = torch.cuda.is_available()
print('USE GPU: '+str(USE_GPU))
# mini-batch size
BATCH_SIZE = 32
# learning rage
LR = 1e-4
# epsilon-greedy
EPSILON = 1.0

'''Save&Load Settings'''
# check save/load
SAVE = True
LOAD = False
# save frequency
SAVE_FREQ = int(1e+3)
# paths for predction net, target net, result log
PRED_PATH = './data/model/C51_pred_net_'+args.games+'.pkl'
TARGET_PATH = './data/model/C51_target_net_'+args.games+'.pkl'
RESULT_PATH = './data/plots/C51_result_'+args.games+'.pkl'

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # nn.Sequential을 사용하면 다음과 같입 코드를 간결하게 바꿀 수 있습니다.
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(STATE_LEN, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(7 * 7 * 64, 512)
        
        # action value distribution
        self.fc_q = nn.Linear(512, N_ACTIONS * N_ATOM) 
            
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.orthogonal_(m.weight, gain = np.sqrt(2))
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            

    def forward(self, x):
        # x.size(0) : minibatch size
        mb_size = x.size(0)
        # x: (m, 84, 84, 4) tensor
        x = self.feature_extraction(x / 255.0)
        # x.size(0) : mini-batch size
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        
        # note that output of C-51 is prob mass of value distribution
        action_value = F.softmax(self.fc_q(x).view(mb_size, N_ACTIONS, N_ATOM), dim=2)

        return action_value

    def save(self, PATH):
        torch.save(self.state_dict(),PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))

class DQN(object):
    def __init__(self):
        self.pred_net, self.target_net = ConvNet(), ConvNet()
        # sync eval target
        self.update_target(self.target_net, self.pred_net, 1.0)
        # use gpu
        if USE_GPU:
            self.pred_net.cuda()
            self.target_net.cuda()
            
        # simulator step conter
        self.memory_counter = 0
        # target network step counter
        self.learn_step_counter = 0
        
        # ceate the replay buffer
        self.replay_buffer = ReplayBuffer(MEMORY_CAPACITY)
        
        # define optimizer
        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=LR)
        
        # discrete values
        self.value_range = torch.FloatTensor(V_RANGE) # (N_ATOM)
        if USE_GPU:
            self.value_range = self.value_range.cuda()
        
    def update_target(self, target, pred, update_rate):
        # update target network parameters using predcition network
        for target_param, pred_param in zip(target.parameters(), pred.parameters()):
            target_param.data.copy_((1.0 - update_rate) \
                                    * target_param.data + update_rate*pred_param.data)
            
    def save_model(self):
        # save prediction network and target network
        self.pred_net.save(PRED_PATH)
        self.target_net.save(TARGET_PATH)

    def load_model(self):
        # load prediction network and target network
        self.pred_net.load(PRED_PATH)
        self.target_net.load(TARGET_PATH)

    def choose_action(self, x, EPSILON):
        x = torch.FloatTensor(x)
        if USE_GPU:
            x = x.cuda()

        if np.random.uniform() >= EPSILON:
            # greedy case
            action_value_dist = self.pred_net(x) # (N_ENVS, N_ACTIONS, N_ATOM)
            action_value = torch.sum(action_value_dist * self.value_range.view(1, 1, -1), dim=2) # (N_ENVS, N_ACTIONS)
            action = torch.argmax(action_value, dim=1).data.cpu().numpy()
        else:
            # random exploration case
            action = np.random.randint(0, N_ACTIONS, (x.size(0)))
        return action

    def store_transition(self, s, a, r, s_, done):
        self.memory_counter += 1
        self.replay_buffer.add(s, a, r, s_, float(done))

    def learn(self):
        self.learn_step_counter += 1
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.update_target(self.target_net, self.pred_net, 1e-2)
    
        b_s, b_a, b_r,b_s_, b_d = self.replay_buffer.sample(BATCH_SIZE)
        b_w, b_idxes = np.ones_like(b_r), None
            
        b_s = torch.FloatTensor(b_s)
        b_a = torch.LongTensor(b_a)
        b_s_ = torch.FloatTensor(b_s_)

        if USE_GPU:
            b_s, b_a, b_s_ = b_s.cuda(), b_a.cuda(), b_s_.cuda()

        # action value distribution prediction
        q_eval = self.pred_net(b_s) # (m, N_ACTIONS, N_ATOM)
        mb_size = q_eval.size(0)
        q_eval = torch.stack([q_eval[i].index_select(0, b_a[i]) for i in range(mb_size)]).squeeze(1) 
        # (m, N_ATOM)
        
        # target distribution
        q_target = np.zeros((mb_size, N_ATOM)) # (m, N_ATOM)
        
        # get next state value
        q_next = self.target_net(b_s_).detach() # (m, N_ACTIONS, N_ATOM)
        # next value mean
        q_next_mean = torch.sum(q_next * self.value_range.view(1, 1, -1), dim=2) # (m, N_ACTIONS)
        best_actions = q_next_mean.argmax(dim=1) # (m)
        q_next = torch.stack([q_next[i].index_select(0, best_actions[i]) for i in range(mb_size)]).squeeze(1) 
        q_next = q_next.data.cpu().numpy() # (m, N_ATOM)

        # categorical projection
        '''
        next_v_range : (z_j) i.e. values of possible return, shape : (m, N_ATOM)
        next_v_pos : relative position when offset of value is V_MIN, shape : (m, N_ATOM)
        '''
        # we vectorized the computation of support and position
        next_v_range = np.expand_dims(b_r, 1) + GAMMA * np.expand_dims((1. - b_d),1) \
        * np.expand_dims(self.value_range.data.cpu().numpy(),0)
        next_v_pos = np.zeros_like(next_v_range)
            # clip for categorical distribution
        next_v_range = np.clip(next_v_range, V_MIN, V_MAX)
        # calc relative position of possible value
        next_v_pos = (next_v_range - V_MIN)/ V_STEP
        # get lower/upper bound of relative position
        lb = np.floor(next_v_pos).astype(int)
        ub = np.ceil(next_v_pos).astype(int)
        # we didn't vectorize the computation of target assignment.
        for i in range(mb_size):
            for j in range(N_ATOM):
                # calc prob mass of relative position weighted with distance
                q_target[i, lb[i,j]] += (q_next * (ub - next_v_pos))[i,j]
                q_target[i, ub[i,j]] += (q_next * (next_v_pos - lb))[i,j]
                
        q_target = torch.FloatTensor(q_target)
        if USE_GPU:
            q_target = q_target.cuda()
        
        # calc huber loss, dont reduce for importance weight
        loss = q_target * ( - torch.log(q_eval + 1e-8)) # (m , N_ATOM)
        loss = torch.mean(loss)
        
        # calc importance weighted loss
        b_w = torch.Tensor(b_w)
        if USE_GPU:
            b_w = b_w.cuda()
        loss = torch.mean(b_w*loss)
        
        # backprop loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()

# model load with check
if LOAD and os.path.isfile(PRED_PATH) and os.path.isfile(TARGET_PATH):
    dqn.load_model()
    pkl_file = open(RESULT_PATH,'rb')
    result = pickle.load(pkl_file)
    pkl_file.close()
    print('Load complete!')
else:
    result = []
    print('Initialize results!')

print('Collecting experience...')

# episode step for accumulate reward 
epinfobuf = deque(maxlen=100)
# check learning time
start_time = time.time()

# env reset
s = np.array(env.reset())

# for step in tqdm(range(1, STEP_NUM//N_ENVS+1)):
for step in range(1, STEP_NUM//N_ENVS+1):
    a = dqn.choose_action(s, EPSILON)

    # take action and get next state
    s_, r, done, infos = env.step(a)
    # log arrange
    for info in infos:
        maybeepinfo = info.get('episode')
        if maybeepinfo: epinfobuf.append(maybeepinfo)
    s_ = np.array(s_)

    # clip rewards for numerical stability
    clip_r = np.sign(r)

    # store the transition
    for i in range(N_ENVS):
        dqn.store_transition(s[i], a[i], clip_r[i], s_[i], done[i])

    # annealing the epsilon(exploration strategy)
    if step <= int(1e+3):
        # linear annealing to 0.9 until million step
        EPSILON -= 0.9/1e+3
    elif step <= int(1e+4):
        # linear annealing to 0.99 until the end
        EPSILON -= 0.09/(1e+4 - 1e+3)

    # if memory fill 50K and mod 4 = 0(for speed issue), learn pred net
    if (LEARN_START <= dqn.memory_counter) and (dqn.memory_counter % LEARN_FREQ == 0):
        dqn.learn()

    # print log and save
    if step % SAVE_FREQ == 0:
        # check time interval
        time_interval = round(time.time() - start_time, 2)
        # calc mean return
        mean_100_ep_return = round(np.mean([epinfo['r'] for epinfo in epinfobuf]),2)
        result.append(mean_100_ep_return)
        # print log
        print('Used Step:',dqn.memory_counter,
              'EPS: ', round(EPSILON, 3),
              '| Mean ep 100 return: ', mean_100_ep_return,
              '| Used Time:',time_interval)
        # save model
        dqn.save_model()
        pkl_file = open(RESULT_PATH, 'wb')
        pickle.dump(np.array(result), pkl_file)
        pkl_file.close()

    s = s_

    if RENDERING:
        env.render()
print("The training is done!")