###########################################################################################
# Implementation of The Quantile Option Architecture for Reinforcement Learning (QUOTA)
# Author for codes: Kun Chu(kun_chu@outlook.com)
# Paper: https://arxiv.org/abs/1811.02073v2
# Reference: https://github.com/ShangtongZhang/DeepRL/tree/QUOTA-discrete
###########################################################################################
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import pickle
import time
from collections import deque
from copy import deepcopy
import argparse
from wrappers import wrap, wrap_cover, SubprocVecEnv


# Parameters
parser = argparse.ArgumentParser(description='Some settings of the experiment.')
parser.add_argument('games', type=str, nargs=1, help='name of the games. for example: Breakout')
args = parser.parse_args()
args.games = "".join(args.games)

'''QUOTA settings'''
# sequential images to define state
STATE_LEN = 4
# target policy sync interval
TARGET_REPLACE_ITER = 1
# simulator steps for start learning
LEARN_START = int(1e+3)
# experience replay memory size
MEMORY_CAPACITY = int(1e+5)
# simulator steps for learning interval
LEARN_FREQ = 4
# quantile and option numbers for QUOTA
N_QUANT = 200
N_OPTIONS = 10

'''Environment Settings'''
# number of environments for C51
N_ENVS = 16
# Total simulation step
STEP_NUM = int(1e+8)
# gamma for MDP
GAMMA = 0.99
# visualize for agent playing
RENDERING = False
# openai gym env name
ENV_NAME = args.games+'NoFrameskip-v4'
env = SubprocVecEnv([wrap_cover(ENV_NAME) for i in range(N_ENVS)])
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape

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
EPSILON_O = 1.0
# option paramater
Target_beta = 0.01
Behavior_beta = 0.01

'''Save&Load Settings'''
# check save/load
SAVE = True
LOAD = False
# save frequency
SAVE_FREQ = int(1e+3)
# paths for predction net, target net, result log
PRED_PATH = './data/model/quota_pred_net_'+args.games+'.pkl'
TARGET_PATH = './data/model/quota_target_net_'+args.games+'.pkl'
RESULT_PATH = './data/plots/quota_result_'+args.games+'.pkl'

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, option=None):
        data = (obs_t, action, reward, obs_tp1, done, option)
        self.option = option

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, options = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, option = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            options.append(option)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), np.array(options)

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.feature_extraction = nn.Sequential(
            # Conv2d(输入channels, 输出channels, kernel_size, stride)
            nn.Conv2d(STATE_LEN, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(7 * 7 * 64, 512)
        # Quantile output
        self.fc_quantiles = nn.Linear(512, N_ACTIONS * N_QUANT)
        # Option output
        self.fc_options = nn.Linear(512, N_OPTIONS)
        
        # Initialization    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.orthogonal_(m.weight, gain = np.sqrt(2))
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            
    def forward(self, x):
        # x.size(0) : minibatch size
        mb_size = x.size(0)
        # x: (m, 84, 84, 4), tensor
        y = self.feature_extraction(x / 255.0)      # (m, 64 * 7 * 7)
        y = y.view(y.size(0), -1)                   # (m, 3136)
        y = F.relu(self.fc(y))                      # (m, 512)
        quantiles = self.fc_quantiles(y)            # (m, N_action * N_quantile)
        quantiles = quantiles.view(-1, N_ACTIONS, N_QUANT)  # (m, N_action, N_quantile)
        options = self.fc_options(y)                # (m, N_options)
        
        return quantiles, options


    def save(self, PATH):
        torch.save(self.state_dict(),PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))

class QUOTA(object):
    def __init__(self):
        self.pred_net, self.target_net = ConvNet(), ConvNet()
        # sync evac target
        self.update_target(self.target_net, self.pred_net, 1.0)
        # use gpu
        if USE_GPU:
            self.pred_net.cuda()
            self.target_net.cuda()
            
        # simulator step counter
        self.memory_counter = 0
        # target network step counter
        self.learn_step_counter = 0
        
        # ceate the replay buffer
        self.replay_buffer = ReplayBuffer(MEMORY_CAPACITY)
        
        # define optimizer
        # self.optimizer = torch.optim.RMSprop(self.pred_net.parameters(), lr=LR, alpha=0.99, eps=1e-5)
        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=LR)

        # rewards
        # self.episode_rewards = np.zeros(N_ENVS)
        # self.last_episode_rewards = np.zeros(N_ENVS)

        # set cumulative density
        self.cumulative_density = torch.cuda.FloatTensor((2 * np.arange(N_QUANT) + 1) / (2.0 * N_QUANT))

        # set options
        self.options = torch.cuda.FloatTensor(np.random.randint(N_OPTIONS, size=N_ENVS)).long()

    # update_target(soft update)
    def update_target(self, target, pred, update_rate):
        # update target network parameters using predcition network
        for target_param, pred_param in zip(target.parameters(), pred.parameters()):
            target_param.data.copy_((1.0 - update_rate) \
                                    * target_param.data + update_rate * pred_param.data)
    
    def save_model(self):
        # save prediction network and target network
        self.pred_net.save(PRED_PATH)
        self.target_net.save(TARGET_PATH)

    def load_model(self):
        # load prediction network and target network
        self.pred_net.load(PRED_PATH)
        self.target_net.load(TARGET_PATH)

    def choose_action(self, x, EPSILON, EPSILON_O):
        # x:state
        x = torch.FloatTensor(x)
        if USE_GPU:
            x = x.cuda()

        # Network output  (m, N_action, N_quantile), (m, N_option)
        quantile_values, option_values = self.pred_net(x)

        ## Choose Option
        # Get new option (beta)
        mb_size = quantile_values.size(0)
        greedy_options = torch.argmax(option_values, dim=-1)
        random_options = torch.cuda.FloatTensor(np.random.randint(N_OPTIONS, size=mb_size)).long()
        dice = torch.cuda.FloatTensor(np.random.rand(mb_size))
        # Judge if random (epsilon) or greedy (1-epsilon)
        new_options = torch.where(dice < EPSILON_O, random_options, greedy_options)
        # Remain Old Option (1-beta)
        dice = np.random.rand(mb_size)
        start_new_options = dice < Behavior_beta
        start_new_options = torch.cuda.FloatTensor(start_new_options.astype(np.uint8)).byte()
        # Set option
        self.options = torch.where(start_new_options, new_options, self.options)
        q_values = self.option_to_q_values(self.options, quantile_values)

        ## Choose action
        if np.random.uniform() >= EPSILON:
            # greedy case
            action = torch.argmax(q_values, dim=1).data.cpu().numpy()
        else:
            # random exploration case
            action = np.random.randint(0, N_ACTIONS, (x.size(0)))
        return action

    def store_transition(self, s, a, r, s_, done, option):
        self.memory_counter += 1
        self.replay_buffer.add(s, a, r, s_, float(done), option)

    def option_to_q_values(self, options, quantiles):
        if N_QUANT % N_OPTIONS:
            raise Exception('Quantile options is not supported')
        # (m, N_action, N_quantile)
        quantiles = quantiles.view(quantiles.size(0), quantiles.size(1), N_OPTIONS, -1)
        # (m, N_action, N_option, K)  K:windows_size
        quantiles = quantiles.mean(-1)
        q_values = quantiles[range(quantiles.size(0)), :, options]
        return q_values

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1 - cond)

    def learn(self):
        self.learn_step_counter += 1
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.update_target(self.target_net, self.pred_net, 1e-2)
            # self.target_net.load_state_dict(self.pred_net.state_dict())
    
        b_s, b_a, b_r, b_s_, b_d, options= self.replay_buffer.sample(BATCH_SIZE)
    
        b_s = torch.FloatTensor(b_s)
        b_a = torch.LongTensor(b_a)
        b_r = torch.FloatTensor(b_r)
        b_s_ = torch.FloatTensor(b_s_)
        b_d = torch.FloatTensor(b_d)

        if USE_GPU:
            b_s, b_a, b_r, b_s_, b_d = b_s.cuda(), b_a.cuda(), b_r.cuda(), b_s_.cuda(), b_d.cuda()

        # Pre_net output (BATCH_size, N_action, N_quantile), (BATCH_size, N_option)
        quantile_values, option_values = self.pred_net(b_s)
        
        quantile_values = quantile_values[range(BATCH_SIZE), b_a, :]  # (BATCH_size, N_quantile)
        quantile_values = quantile_values.unsqueeze(2)

        # Target_net output (BATCH_size, N_action, N_quantile), (BATCH_size, N_option)
        quantile_values_target, option_values_target = self.target_net(b_s_)
        a_next = torch.argmax(quantile_values_target.sum(-1), dim=1) # (BATCH_size)
        quantile_values_target = quantile_values_target[range(BATCH_SIZE), a_next, :].detach()
        
        # Calc target y
        # y = beta * max_w' Q_o(St+1, w') + (1 - beta) * Q_o(St+1, Wt)
        option_values_target = option_values_target.detach()
        option_values_target = Target_beta * torch.max(option_values_target, dim=1)[0] + \
                         (1 - Target_beta) * option_values_target[range(BATCH_SIZE), options]
        # option_valves_target (BATCH_size)
        option_values_target = option_values_target.unsqueeze(1)
        # r + gamma * y
        b_d = b_d.unsqueeze(1)
        option_values_target = GAMMA * (1. - b_d) * option_values_target
        option_values_target = option_values_target + b_r

        # Calc Q_o(St+1, Wt)
        option_values = option_values[range(BATCH_SIZE), options]

        # Yt,i = r + gamma * qi(St+1, a_next)
        b_d = b_d.expand(BATCH_SIZE, N_QUANT)
        quantile_values_target = GAMMA * (1. - b_d) * quantile_values_target
        quantile_values_target = quantile_values_target + b_r.unsqueeze(1)
        quantile_values_target = quantile_values_target.unsqueeze(1).detach() #(m, 1, N_quant)

        # TD loss (m, N_quant, N_quant)
        diff = quantile_values_target - quantile_values
        loss = self.huber(diff) * (self.cumulative_density.view(1, -1).unsqueeze(0) -\
                                 (diff.detach() < 0).float()).abs()
        loss = torch.mean(torch.mean(loss, dim=1).mean(1))
        # Option loss
        option_loss = (option_values - option_values_target).pow(2).mul(0.5).mean()
        # Total loss
        loss = loss + option_loss

        self.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(self.pred_net.parameters(), Gradient_clip)
        self.optimizer.step()

quota = QUOTA()

# model load with check
if LOAD and os.path.isfile(PRED_PATH) and os.path.isfile(TARGET_PATH):
    quota.load_model()
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

# Trainning
for step in range(1, STEP_NUM//N_ENVS + 1):
    a = quota.choose_action(s, EPSILON, EPSILON_O)

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
        quota.store_transition(s[i], a[i], clip_r[i], s_[i], done[i], quota.options[i].item())

    # annealing the epsilon(exploration strategy)
    if step <= int(1e+4):
        # linear annealing to 0.9 until million step
        EPSILON -= 0.9/1e+4
    elif step <= int(2e+4):
        # linear annealing to 0.99 until the end
        EPSILON -= 0.09/1e+4

    if step <= int(2e+4):
    # linear annealing to 0.9 until million step
        EPSILON_O -= 0.95/2e+4

    # if memory fill 50K and mod 4 = 0(for speed issue), learn pred net
    if (LEARN_START <= quota.memory_counter) and (quota.memory_counter % LEARN_FREQ == 0):
        loss = quota.learn()

    # print log and save
    if step % SAVE_FREQ == 0:
        # check time interval
        time_interval = round(time.time() - start_time, 2)
        # calc mean return
        mean_100_ep_return = round(np.mean([epinfo['r'] for epinfo in epinfobuf]),2)
        result.append(mean_100_ep_return)
        # print log
        print('Used Step:',quota.memory_counter,
              '| EPS_A: ', round(EPSILON, 3),
              '| EPS_O: ', round(EPSILON_O, 3),
              '| Mean ep 100 return: ', mean_100_ep_return,
              '| Used Time:',time_interval)
        # save model
        quota.save_model()
        pkl_file = open(RESULT_PATH, 'wb')
        pickle.dump(np.array(result), pkl_file)
        pkl_file.close()

    s = s_

    if RENDERING:
        env.render()
print("The training is done!")