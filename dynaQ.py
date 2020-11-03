from collections import defaultdict
import numpy as np
import random


class DynaQ:
    def __init__(self, alpha=0.1, gamma=0.95, eps=0.2, planning_n=50):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.planning_n = planning_n
        self.action_space = ['up', 'down', 'right', 'left']
        self.model = defaultdict(int)
        self.Q = defaultdict(int)

    def greedy(self, state):
        Q_list = [self.Q[state,a] for a in self.action_space]
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
            return self.greedy(state)
        else:
            return np.random.choice(self.action_space)

    def update(self, state, action, next_state, reward):
        greedy_action = self.greedy(next_state)
        self.Q[state, action] += self.alpha*(reward + self.gamma*self.Q[next_state, greedy_action] - self.Q[state,action])

    def model_update(self, state, action, next_state, reward):
        self.model[state,action] = reward, next_state

    def planning(self):
        for _ in range(self.planning_n):
            state, action = random.sample(self.model.keys(), 1)[0]
            reward, next_state = self.model[state, action]
            self.update(state, action, next_state, reward)

    def reset(self):
        self.Q = defaultdict(int)
        self.model = defaultdict(int)

