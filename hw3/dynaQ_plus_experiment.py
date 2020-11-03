from collections import defaultdict
import numpy as np
import random


class DynaQPlusExperiment:
    def __init__(self, alpha=0.1, gamma=0.95, eps=0.2, planning_n=50, k=0.001):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.planning_n = planning_n
        self.action_space = ['up', 'down', 'right', 'left']
        self.model = defaultdict(int)
        self.Q = defaultdict(int)
        self.k = k
        self.tao_dict = defaultdict(int)

    def greedy(self, state):
        Q_list = [self.Q[state, a] + self.bonus(state, a) for a in self.action_space]
        _max = max(Q_list)
        if _max == 0:
            return np.random.choice(self.action_space)
        else:
            tie_actions = []
            for i, q in enumerate(Q_list):
                if q == _max:
                    tie_actions.append(self.action_space[i])
            return np.random.choice(tie_actions)

    def eps_greedy(self, state):
        if np.random.random() > self.eps:
            action = self.greedy(state)
        else:
            action = np.random.choice(self.action_space)
        for k in self.tao_dict.keys():
            if k != (state, action):
                self.tao_dict[k] += 1
        self.tao_dict[state, action] = 0
        return action

    def update(self, state, action, next_state, reward):
        greedy_action = self.greedy(next_state)
        self.Q[state, action] += self.alpha * (
                    reward + self.gamma * self.Q[next_state, greedy_action] - self.Q[state, action])

    def model_update(self, state, action, next_state, reward):
        self.model[state, action] = reward, next_state

    def bonus(self, state, action):
        return self.k * np.sqrt(self.tao_dict[state, action])

    def planning(self):
        for _ in range(self.planning_n):
            state, action = random.sample(self.model.keys(), 1)[0]
            reward, next_state = self.model[state, action]
            self.update(state, action, next_state, reward)

    def reset(self):
        self.Q = defaultdict(int)
        self.model = defaultdict(int)
        self.tao_dict = defaultdict(int)
