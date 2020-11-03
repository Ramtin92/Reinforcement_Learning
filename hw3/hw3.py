import numpy as np
from dynaQ_plus_experiment import DynaQPlusExperiment
from dynaQ_plus import DynaQPlus
from dynaQ import DynaQ
from do_plot import plot_figures


class Maze(object):

    def __init__(self, size, block_position, goal):
        self.width, self.height = size
        self.block_position = block_position
        self.goal = goal

    def move(self, curr_state, action):
        x, y = curr_state
        dx, dy = 0, 0
        if action == 'up':
            dy = 1
        elif action == 'down':
            dy = -1
        elif action == 'left':
            dx = -1
        elif action == 'right':
            dx = 1
        else:
            raise ValueError('action {} is undefined'.format(action))

        x_change = min(self.width, max(1, x+dx))
        y_change = min(self.height, max(1, y+dy))

        if (x_change, y_change) in self.block_position:
            x_change, y_change = x,y

        if (x_change, y_change) == self.goal:
            reward = 1
        else:
            reward = 0

        return reward, (x_change, y_change)


def play(agent):
    EXPERIMENT = 5
    HIST = []
    env = Maze(size=(9, 6), block_position=[(3, 3), (3, 4), (3, 5), (6, 2), (8, 4), (8, 5), (8, 6)], goal=(9, 6))
    EPISODE = 150
    for experiment in range(EXPERIMENT):
        agent.reset()
        hist = []
        for episode in range(EPISODE):
            if episode == 50:
                env.block_position.append((3, 6))
                env.block_position.append((3, 2))
                env.block_position.append((8, 4))
                env.block_position.append((8, 5))
            if episode == 100:
                env.block_position = []
            state = (1, 4)
            for step in range(100000):
                action = agent.eps_greedy(state)
                reward, next_state = env.move(state, action)
                agent.update(state, action, next_state, reward)
                agent.model_update(state, action, next_state, reward)
                if (step+1) % 1000 == 0:
                    print(step)
                if episode >= 1:
                    agent.planning()
                state = next_state
                if reward == 1:
                    hist.append(step)
                    break
        HIST.append(hist)
    HIST = np.mean(HIST, axis=0)
    return HIST


def main():
    DynaQ_5 = play(DynaQ(planning_n=5))
    DynaQ_50 = play(DynaQ(planning_n=50))
    DynaQPlus_5 = play(DynaQPlus(planning_n=5))
    DynaQPlus_50 = play(DynaQPlus(planning_n=50))
    DynaQPlus_5_EXP = play(DynaQPlusExperiment(planning_n=5))
    DynaQPlus_50_EXP = play(DynaQPlusExperiment(planning_n=50))
    plot_figures(DynaQ_5, DynaQ_50, DynaQPlus_5, DynaQPlus_50, DynaQPlus_5_EXP, DynaQPlus_50_EXP)


if __name__ == "__main__":
    main()




