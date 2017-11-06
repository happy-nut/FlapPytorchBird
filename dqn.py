# -*- coding: utf-8 -*-
import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from collections import deque
from itertools import count
from copy import deepcopy
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random

ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
REPLAY_MEMORY = 1000000 # number of previous transitions to remember
MINIMUM_REPLAY_SIZE = 5000
BATCH_SIZE = 32 # size of minibatch
FRAME_PER_ACTION = 1
EPSILON_FIRST = 1.0 # Exploration rate
EPSILON_FINAL = 0.001
EPSILON_DECAY = 200
OBSERVED = 50000
M = 100000 # episode count
C = 10000 # Q reset interval
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
resize = T.Compose([T.ToPILImage(),
                    T.Scale(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def downsize(self):
        if len(self.memory) > REPLAY_MEMORY:
            self.memory = self.memory[OBSERVED:]

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self):

        super(DQN, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=4)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(1600, 512)
        self.fc2 = nn.Linear(512, ACTIONS)

    def forward(self, x):
        #print("network input: "+str(x.size()))
        x = self.pool(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(-1, 1600)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


Q = DQN()
target_Q = DQN()

if use_cuda:
    Q.cuda()
    target_Q.cuda()

optimizer = optim.RMSprop(Q.parameters())
memory = ReplayMemory(REPLAY_MEMORY)


steps_done = 0

def select_action(state):
    global steps_done
    steps_done += 1
    action = np.zeros([ACTIONS])
    eps_threshold = EPSILON_FINAL + (EPSILON_FIRST - EPSILON_FINAL) * math.exp(-1. * steps_done / EPSILON_DECAY)

    if random.random() <= eps_threshold:
        #print("----------Random Action----------", steps_done)
        action_index = random.randrange(ACTIONS)
        action[action_index] = 1
    else:
        readout = Q(Variable(state, volatile=True).type(FloatTensor)).data.max(0)[0]
        readout = readout.unsqueeze(0).type(Tensor)
        action_index = np.argmax(readout)
        action[action_index] = 1
    action = torch.from_numpy(action)
    return action


def optimize_model():
    global steps_done

    if steps_done % C == 0:
        target_Q.load_state_dict(Q.state_dict())

    if len(memory) < MINIMUM_REPLAY_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE) # Batch_size만큼 샘플 추출

    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)

    state_batch = Variable(torch.cat(batch.state))  # 32개의 샘플 중 state만 모아놓음
    reward_batch = Variable(torch.cat(batch.reward))
    Q_Values = Q(state_batch).max(1)[0]

    target_Q_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    target_Q_values[non_final_mask] = target_Q(non_final_next_states).max(1)[0]
    target_Q_values.volatile = False

    y_values = reward_batch + (target_Q_values * GAMMA)
    loss = F.mse_loss(y_values, Q_Values)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in Q.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()




if __name__ == '__main__':

    # for i in range(0, M) :
    while True:
        game_state = game.GameState()
        do_nothing = torch.zeros(ACTIONS)
        do_nothing[0] = 1
        x_0, r_0, terminal = game_state.frame_step(do_nothing)
        x_0 = cv2.cvtColor(cv2.resize(x_0, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_0 = cv2.threshold(x_0, 1, 255, cv2.THRESH_BINARY)

        x_0 = torch.from_numpy(x_0).type(FloatTensor).unsqueeze(0)
        s_0 = torch.cat((x_0, x_0, x_0, x_0), dim=0)  # 맨 처음 초기 상태 s를 구함
        state = s_0.unsqueeze(0)
        a_0 = select_action(state)
        action = a_0

        while True:
            if terminal:
                # Optimize the model
                optimize_model()
                memory.downsize()
                break
            action = select_action(state)
            x_t_colored, reward, terminal = game_state.frame_step(action)  # 초기 상태에서 action을 선택한 대로 진행
            #print(reward)
            x_t = cv2.cvtColor(cv2.resize(x_t_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
            x_t = torch.from_numpy(x_t).type(FloatTensor).unsqueeze(0)
            temp_state = state.squeeze(0)
            next_state = torch.cat((x_t, temp_state[:3, ...]), dim=0).unsqueeze(0)
            reward = FloatTensor([reward])
            memory.push(state, action, next_state, reward)
            state = next_state

print('Complete')


