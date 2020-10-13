import random
import numpy as np
import constants


class RaceTrack:
    STEP_REWARD = -1
    HIT_OBSTACLE_REWARD = -50

    def __init__(self, racetrack, random_displacement_probability=0.5):
        self.racetrack = racetrack
        self.random_displacement_probability = random_displacement_probability
        self.start_coordinates = None
        self.get_start_positions()

        self.position = None
        self.velocity = None
        self.done = None
        self.reset()

    def get_start_positions(self):
        self.start_coordinates = []
        x_coordinates, y_coordinates = np.where(self.racetrack == constants.START_VALUE)

        for x, y in zip(x_coordinates, y_coordinates):
            self.start_coordinates.append((x, y))

    def act(self, x_change, y_change):
        assert not self.done
        assert -1 <= x_change <= 1
        assert -1 <= y_change <= 1

        self.velocity = (self.velocity[0] + x_change, self.velocity[1] + y_change)
        self.correct_velocity()

        last_position = self.position
        self.update_position(self.velocity)
        self.random_displacement()

        if self.check_finish():
            self.done = True
            return self.STEP_REWARD

        invalid_position = False

        if self.check_position_out_of_bounds() or self.check_position_grass():
            invalid_position = True
            self.correct_invalid_position(last_position)
            self.correct_same_position()

        if self.check_finish():
            self.done = True

            if invalid_position:
                return self.HIT_OBSTACLE_REWARD
            else:
                return self.STEP_REWARD

        if invalid_position:
            return self.HIT_OBSTACLE_REWARD
        else:
            return self.STEP_REWARD

    def update_position(self, velocity):
        self.position = (self.position[0] - velocity[0], self.position[1] + velocity[1])

    def check_finish(self):
        tmp_position = self.position

        if self.position[0] < 0 or self.position[0] >= self.racetrack.shape[0]:
            return False

        if self.position[1] < 0:
            tmp_position = (self.position[0], 0)
        elif self.position[1] >= self.racetrack.shape[1]:
            tmp_position = (self.position[0], self.racetrack.shape[1] - 1)

        if self.racetrack[tmp_position] == constants.END_VALUE:
            return True
        else:
            return False

    def correct_velocity(self):
        if self.velocity[0] < 0:
            self.velocity = (0, self.velocity[1])
        elif self.velocity[0] > 4:
            self.velocity = (4, self.velocity[1])

        if self.velocity[1] < 0:
            self.velocity = (self.velocity[0], 0)
        elif self.velocity[1] > 4:
            self.velocity = (self.velocity[0], 4)

        if self.velocity == (0, 0):
            if random.choice([True, False]):
                self.velocity = (1, 0)
            else:
                self.velocity = (0, 1)

    def check_position_out_of_bounds(self):
        return self.position[0] < 0 or self.position[0] >= self.racetrack.shape[0] or self.position[1] < 0 or \
               self.position[1] >= self.racetrack.shape[1]

    def check_position_grass(self):
        return self.racetrack[self.position] == constants.OBSTACLE_VALUE

    def correct_invalid_position(self, last_position):
        self.position = last_position
        self.velocity = (0, 0)

    def correct_same_position(self):
        self.update_position((1, 0))
        if self.check_position_out_of_bounds() or self.check_position_grass():
            self.update_position((-1, 1))

    def random_displacement(self):
        if np.random.uniform(0, 1) < self.random_displacement_probability:
            if np.random.choice([True, False]):
                self.update_position((1, 0))
            else:
                self.update_position((0, 1))

    def reset(self):
        self.position = random.choice(self.start_coordinates)
        self.velocity = (0, 0)
        self.done = False

    @property
    def get_state(self):
        return self.position[0], self.position[1], self.velocity[0], self.velocity[1]
