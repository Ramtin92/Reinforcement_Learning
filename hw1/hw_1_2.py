import numpy as np
from plot_functions import print_plots_hw2
import time


# TestBed class containing the states and actions
class Testbed:

    def __init__(self, number_arms, mean, std, q_zero):
        # Number of arms
        self.number_arms = number_arms
        # Gaussian random actions
        self.mean = mean
        self.std = std
        self.action_values = np.zeros(number_arms)
        self.action_values = np.full(number_arms, q_zero)
        self.optimal_value = 0.0

    # Reset testbed for next iteration
    def reset(self):
        # Set random from a Gaussian distribution
        self.action_values += np.random.normal(self.mean, self.std, self.number_arms)
        # Determine the maximum value in action array
        self.optimal_value = np.argmax(self.action_values)


# Agent Class - Controls the agents movement and behavior in the environment interacting with the testbed
class Agent:

    def __init__(self, number_arms, epsilon_prob=0):
        self.number_arms = number_arms
        self.epsilon_prob = epsilon_prob
        self.time_step = 0
        self.last_action = None
        self.count_action = np.zeros(number_arms)  # Count actions taken up to time t
        self.reward_sum = np.zeros(number_arms)  # Sum number of rewards
        self.value_estimates = np.zeros(number_arms)  # Action value estimates

    # Return string for graph legend
    @property
    def __str__(self):
        if self.epsilon_prob == 0.0:
            return "Greedy"
        else:
            return "Epsilon = " + str(self.epsilon_prob)

    # Selects action based on a epsilon-greedy behavior
    def action(self):
        random_prob = np.random.random()
        if random_prob < self.epsilon_prob:
            # Epsilon method
            a = np.random.choice(len(self.value_estimates))
        else:
            # Greedy method
            max_action = np.argmax(self.value_estimates)
            action = np.where(self.value_estimates == np.argmax(self.value_estimates))[0]
            # break the tie in case multiple actions have the same value
            if len(action) == 0:
                a = max_action
            else:
                a = np.random.choice(action)
        # save last action in variable, and return result
        self.last_action = a
        return a

    # Update the value estimates based on the last action
    def update_value_estimate(self, reward, alpha=0.1, flag="sample-average"):

        at = self.last_action
        if flag == "sample-average":
            self.reward_sum[at] += reward  # Add reward to sum array
            self.count_action[at] += 1  # Add 1 to action selection
            # Calculate new action-value
            self.value_estimates[at] = self.reward_sum[at] / self.count_action[at]
        else:
            self.value_estimates[at] = self.value_estimates[at] + alpha * (
                    reward - self.value_estimates[at])

            self.reward_sum[at] += reward  # Add reward to sum array
            self.count_action[at] += 1  # Add 1 to action selection

        # Increase time step
        self.time_step += 1

    # Reset all variables for next iteration
    def reset(self):
        self.time_step = 0
        self.last_action = None

        self.count_action[:] = 0  # Count of actions taken at time t
        self.reward_sum[:] = 0
        self.value_estimates[:] = 0  # Action value estimates Qt ~= Q*(a)


# Environment class to control all objects (agent/Testbed)
class Environment:

    def __init__(self, testbed, agents, steps, iterations, flag):
        self.testbed = testbed
        self.agents = agents

        self.steps = steps
        self.iterations = iterations

        self.flag = flag

    @property
    def play(self):

        # Array to store the scores
        score_array = np.zeros((self.steps, len(self.agents)))
        # Array to maintain optimal count, Graph 2
        optimal_array = np.zeros((self.steps, len(self.agents)))

        # Loop for number of iterations
        for iteration in range(self.iterations):
            # print statement after every 100 iterations
            if (iteration % 100) == 0:
                print("Completed Iterations: ", iteration)

            # Reset testbed and all agents
            self.testbed.reset()
            for agent in self.agents:
                agent.reset()

            # Loop for number of steps
            for step in range(self.steps):
                agent_count = 0

                for agent in self.agents:
                    action_T = agent.action()

                    # Reward - normal dist (Q*(at), variance = 1)
                    reward_T = np.random.normal(self.testbed.action_values[action_T], scale=1)

                    # Agent checks state
                    agent.update_value_estimate(reward=reward_T, alpha=0.1, flag=self.flag)
                    # Add score in array, graph 1
                    score_array[step, agent_count] += reward_T

                    # check the optimal action, add optimal to array, graph 2
                    if action_T == self.testbed.optimal_value:
                        optimal_array[step, agent_count] += 1

                    agent_count += 1

        score_avg = score_array / self.iterations
        optimal_avg = optimal_array / self.iterations

        return score_avg, optimal_avg


if __name__ == "__main__":

    number_arms = 10
    iterations = 2000
    steps = 10000

    scores_one_list = []
    q_zero_list = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1.0, 2.0, 4.0]
    for q_zero in q_zero_list:
        np.random.seed(100)
        start_time = time.time()
        testbed = Testbed(number_arms=number_arms, mean=0.0, std=0.01, q_zero=q_zero)
        agents = [Agent(number_arms=number_arms, epsilon_prob=0.1)]
        environment = Environment(testbed=testbed, agents=agents, steps=steps, iterations=iterations, flag="recency")

        # Run Environment
        print("Running...")
        scores_one, _ = environment.play
        scores_one_list.append(np.mean(scores_one[int(steps/2):, ]))
        print("Execution time: %s seconds" % (time.time() - start_time))

    print_plots_hw2(scores_one_list, q_zero_list)
