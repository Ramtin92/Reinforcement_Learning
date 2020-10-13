import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as m_patches
import constants


def update_mean(value, mean, count):
    return (value - mean) / (count + 1)


class MonteCarlo:
    NUMBER_ACTIONS = 9
    NUMBER_SPEEDS = 5
    ACTION_TO_ACCELERATION = np.array([[1, 1], [0, 1], [1, 0], [0, 0], [-1, 0], [0, -1], [1, -1], [-1, 1], [-1, -1]])

    def __init__(self, env, epsilon, init=-100):
        self.env = env
        self.epsilon = epsilon
        self.init = init

        self.action_values = None
        self.action_counts = None
        self.policy = None
        self.reset()

    def play_episode(self, explore=True, learn=True):
        the_sequence = []

        while not self.env.done:

            state = self.env.get_state

            if explore:
                action = self.explore(self.policy[state])
            else:
                action = self.policy[state]

            reward = self.env.act(*self.action_to_acceleration(action))

            the_sequence.append((state, action, reward))

        returns = np.zeros(len(the_sequence))

        for i in reversed(range(len(the_sequence))):
            for j in range(i + 1):
                returns[j] += the_sequence[i][2]

        if learn:
            for i in range(len(the_sequence)):
                state = the_sequence[i][0]
                action = the_sequence[i][1]
                state_action = state + (action,)
                ret = returns[i]

                self.action_values[state_action] += update_mean(ret, self.action_values[state_action],
                                                                self.action_counts[state_action])
                self.action_counts[state_action] += 1

        return returns[0], the_sequence

    def reset(self):
        self.action_values = \
            np.zeros((self.env.racetrack.shape[0], self.env.racetrack.shape[1], self.NUMBER_SPEEDS, self.NUMBER_SPEEDS,
                      self.NUMBER_ACTIONS), dtype=np.float32) - self.init
        self.action_counts = \
            np.zeros((self.env.racetrack.shape[0], self.env.racetrack.shape[1], self.NUMBER_SPEEDS, self.NUMBER_SPEEDS,
                      self.NUMBER_ACTIONS), dtype=np.int32)
        self.policy = np.zeros(
            (self.env.racetrack.shape[0], self.env.racetrack.shape[1], self.NUMBER_SPEEDS, self.NUMBER_SPEEDS),
            dtype=np.int32)

    def update_policy(self):
        self.policy = np.argmax(self.action_values, axis=-1)

    def action_to_acceleration(self, action):
        return self.ACTION_TO_ACCELERATION[action]

    def explore(self, action):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.NUMBER_ACTIONS)
        else:
            return action

    def plot_sequence(self, sequence, save_path=None, show_legend=True):
        track = self.env.racetrack.copy()

        for index, item in enumerate(sequence):
            state = item[0]
            track[state[0], state[1]] = 4

        im = plt.imshow(track)
        plt.axis("off")

        if show_legend:
            values = np.unique(track.ravel())
            labels = {
                constants.START_VALUE: "start",
                constants.END_VALUE: "end",
                constants.TRACK_VALUE: "track",
                constants.OBSTACLE_VALUE: "obstacle",
                constants.AGENT_VALUE: "agent"
            }
            colors = [im.cmap(im.norm(value)) for value in values]
            patches = [m_patches.Patch(color=colors[i], label=labels[values[i]]) for i in range(len(values))]
            plt.legend(handles=patches, loc=4, fontsize="x-small")

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
