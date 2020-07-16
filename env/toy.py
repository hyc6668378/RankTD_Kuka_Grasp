from gym import spaces
import gym
import numpy as np


class Toy_Task(  gym.Env ):
    T = np.array([
        [0, 0, 0, 1, 2],
        [0, 0, 1, 2, 3],
        [0, 1, 2, 3, 4],
        [-1, -1, - 1, 4, 5, ],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, -1],
        [4, 5, 6, 7, 8],
        [5, 6, 7, 8, 9],
        [6, 7, -1, -1, -1],
        [7, 8, 9, 10, 11],
        [8, 9, -1, -1, -1],
        [9, 10, 11, 12, 13],
        [10, 11, 12, 13, 14],
        [11, 12, -1, -1, -1],
        [12, 13, 14, 15, 16],
        [13, 14, 15, 16, 17],
        [-1, -1, -1, 17, 18],
        [15, 16, 17, 18, 19],
        [16, -1, -1, -1, -1],
        [17, 18, 19, 20, 20]
    ], dtype=np.int16)
    action_space = spaces.discrete.Discrete(5)
    observation_space = spaces.discrete.Discrete(21)

    total_state = list(range(21))
    total_action = list(range(5))
    def __init__(self, model):
        if model == 'sparse_setting':
            self.model = 0
        elif model == 'stateTD':
            self.model = 1
        else:
            self.model = 2

        self.s = 0
        self.inverse_count = 0

    def step(self, action):

        s_ = self.T[self.s][action]

        info = {'is_success': True if s_==20 else False}

        done = True if info['is_success'] or s_==-1 else False

        td = s_ - self.s
        if s_==-1:
            r = -1
        else:
            if self.model == 0:
                r = 0
            elif self.model == 1:
                r = td
            else:
                if td < 0:
                    self.inverse_count += 1

                if self.inverse_count > 2:
                    r = -1
                    done = True
                else:
                    r = td

        self.s = s_
        return self.s, r, done, info

    def reset(self):
        self.s = 0
        self.inverse_count=0
        return 0


import pandas as pd


class QLearningTable:
    def __init__(self, actions, T, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame( np.zeros_like( T, dtype=np.float64 ) )

    def choose_action(self, observation):

        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_, done):

        q_predict = self.q_table.loc[s, a]
        if not done:
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

