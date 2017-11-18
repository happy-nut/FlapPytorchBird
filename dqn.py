# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
from visdom import Visdom

ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # discount factor
LEARNING_RATE = 0.00025
REPLAY_MEMORY = 1000000 # number of previous transitions to remember
MINIMUM_REPLAY_SIZE = 50000
BATCH_SIZE = 32 # size of minibatch
INITIAL_EPSILON = 1.0 # Exploration rate
FINAL_EPSILON = 0.1
EPSILON_CONVERGE = 1000000
OBSERVED = 50000
MOMENTUM = 0
C = 10000 # Q reset interval
NO_OP_MAX = 30

VIS_UPDATE_RATE = 1000
EPISODE_UPDATE_RATE = 1000

SAVED_FILE = "saved_network.pt"

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# vis setting

vis = Visdom()

def initVisWindows(steps=0, episodes=0):
    global eps_window, score_window, max_q_window, loss_window, reward_window
    eps_window = vis.line(Y=np.array([getEpsThreshold(steps)]),
                          X=np.array([steps]),
                          opts=dict(xlabel='steps',
                                    ylabel='rate',
                                    title='Exploration Rate',
                                    ))


    score_window = vis.line(Y=np.array([np.nan]),
                            X=np.array([np.nan]),
                            opts=dict(xlabel='episodes',
                                      ylabel='score',
                                      title='Score',
                                      ))
    max_q_window = vis.line(Y=np.array([np.nan]),
                            X=np.array([np.nan]),
                            opts=dict(xlabel='steps',
                                      ylabel='Q.max',
                                      title='Max Q value',
                                      ))
    loss_window = vis.line(Y=np.array([np.nan]),
                           X=np.array([np.nan]),
                           opts=dict(xlabel='steps',
                                     ylabel='loss',
                                     title='Loss',
                                     ))
    reward_window = vis.line(Y=np.array([np.nan]),
                             X=np.array([np.nan]),
                             opts=dict(xlabel='steps',
                                       ylabel='reward',
                                       title='Reward',
                                       ))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

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

criterion = nn.MSELoss()
optimizer = optim.RMSprop(Q.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(REPLAY_MEMORY)


steps_done = 0
episodes_done = 0


def getEpsThreshold(steps):
    if steps_done < EPSILON_CONVERGE:
        eps_threshold = INITIAL_EPSILON - ((INITIAL_EPSILON - FINAL_EPSILON) / EPSILON_CONVERGE * steps)
    else :
        eps_threshold = FINAL_EPSILON
    return eps_threshold

def select_action(state):

    global steps_done
    steps_done += 1
    action = torch.zeros(ACTIONS)
    epsilon = getEpsThreshold(steps_done)

    if steps_done % VIS_UPDATE_RATE == 0:
        vis.line(Y=np.array([epsilon]),
                 X=np.array([steps_done]),
                 update='append',
                 win=eps_window)

    if random.random() <= epsilon:
        if random.random() < 0.1:
            action_index = 1
        else:
            action_index = 0
        #print("----------Random Action----------", steps_done, action_index)
    else:
        readout = Q(Variable(state, volatile=True).float().cuda())
        action_index = readout.max(1)[1].data[0]
        #print(readout.data, action_index, steps_done)

    action[action_index] = 1
    return action


max_q_sum = loss_sum = train_step_log_count = 0

def optimize_model():
    global loss_sum, train_step_log_count, max_q_sum

    if steps_done % C == 0:
        target_Q.load_state_dict(Q.state_dict())

    if len(memory) < MINIMUM_REPLAY_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True).float().cuda()

    state_batch = Variable(torch.cat(batch.state)).float().cuda()
    reward_batch = Variable(torch.cat(batch.reward)).cuda()
    Q_Values = Q(state_batch).max(1)[0]

    target_Q_values = Variable(torch.zeros(BATCH_SIZE)).cuda()
    target_Q_values[non_final_mask] = target_Q(non_final_next_states).max(1)[0]
    target_Q_values.volatile = False
    # if not non_final_mask.all():
    #     import code
    #     code.interact(local=locals())

    y_values = reward_batch + (GAMMA * target_Q_values)
    loss = criterion(Q_Values, y_values)

    max_q_sum += Q_Values.max().data[0]
    loss_sum += loss.data[0]
    train_step_log_count += 1

    if steps_done % VIS_UPDATE_RATE == 0:
        vis.line(Y=np.array([max_q_sum / train_step_log_count]),
                 X=np.array([steps_done]),
                 update='append',
                 win=max_q_window)

        vis.line(Y=np.array([loss_sum / train_step_log_count]),
                 X=np.array([steps_done]),
                 update='append',
                 win=loss_window)

        max_q_sum = loss_sum = train_step_log_count = 0


    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    #for param in Q.parameters():
    #    param.grad.data.clamp_(-1, 1)
    optimizer.step()


def normalizeInputImages(image_x):
    return image_x/255

def terminateInputImages(image_x):
    return 0*image_x



def save(filename, iteration=0, episodes=0):
    checkpoint = {
        'iteration': iteration,
        'episodes': episodes,
        'Q': Q.state_dict(),
        'target_Q': target_Q.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(checkpoint, filename)
    print('Saved DQN checkpoint (%d steps) into' % iteration, filename)


def load(filename):
    global steps_done, episodes_done
    checkpoint = torch.load(filename)
    steps_done = checkpoint['iteration']
    episodes_done = checkpoint['episodes']
    Q.load_state_dict(checkpoint['Q'])
    target_Q.load_state_dict(checkpoint['target_Q'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Loaded DQN checkpoint (%d steps) from' % steps_done, filename)


if __name__ == '__main__':

    try:
        load(SAVED_FILE)
    except FileNotFoundError as e:
        print(e)

    initVisWindows()
    target_Q.load_state_dict(Q.state_dict())
    reward_sum = 0
    score_sum = 0
    reward_count = 0
    score_count = 0


    while True:
        game_state = game.GameState()
        do_nothing = torch.zeros(ACTIONS)
        do_nothing[0] = 1
        x_0, r_0, terminal = game_state.frame_step(do_nothing)
        x_0 = cv2.cvtColor(cv2.resize(x_0, (80, 80)), cv2.COLOR_BGR2GRAY)
        # _, x_0 = cv2.threshold(x_0, 1, 255, cv2.THRESH_BINARY)

        reward_sum += r_0
        reward_count += 1

        x_0 = torch.from_numpy(x_0).unsqueeze(0)
        x_0 = normalizeInputImages(x_0)
        s_0 = torch.cat((x_0, x_0, x_0, x_0), dim=0)
        state = s_0.unsqueeze(0)
        a_0 = select_action(state)
        action = a_0

        while True:
            if terminal:
                episodes_done += 1
                score_count += 1
                score_sum += game_state.prev_score

                if episodes_done % 200 == 199:
                    vis.line(Y=np.array([score_sum / score_count]),
                             X=np.array([episodes_done]),
                             update='append',
                             win=score_window)
                    score_count = score_sum = 0
                break
            action = select_action(state)
            x_t_colored, reward, terminal = game_state.frame_step(action)
            reward_sum += reward
            reward_count += 1

            x_t = cv2.cvtColor(cv2.resize(x_t_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
            # _, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
            x_t = torch.from_numpy(x_t).unsqueeze(0)
            x_t = normalizeInputImages(x_t)
            temp_state = state.squeeze(0)
            next_state = torch.cat((x_t, temp_state[:3, ...]), dim=0).unsqueeze(0)
            reward = torch.FloatTensor([reward])

            if terminal:
                next_state = None
            memory.push(state, action, next_state, reward)
            state = next_state
            # Optimize the model
            optimize_model()

            if reward_count % VIS_UPDATE_RATE == 0:
                vis.line(Y=np.array([reward_sum / reward_count]),
                         X=np.array([steps_done]),
                         update='append',
                         win=reward_window)
                reward_sum = 0
                reward_count = 0

            if steps_done % 100000 == 0:
                save(SAVED_FILE, steps_done, episodes_done)

print('Complete')



