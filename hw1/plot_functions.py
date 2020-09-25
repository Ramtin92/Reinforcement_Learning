import numpy as np
import matplotlib.pyplot as plt


def print_plots(modes, scores_one_list, scores_two_list):

    fig, axs = plt.subplots(2, 1)

    # Graph 1 - Rewards over all steps
    axs[0].set_title("10-Armed TestBed - Average vs. Recency Weighted Average Rewards")
    axs[0].plot(scores_one_list[0], label="epsilon = 0.1, " + modes[1])
    axs[0].plot(scores_one_list[1], label="epsilon = 0.1, " + modes[0])
    axs[0].set_ylabel('Average Reward')
    axs[0].set_xlabel('Steps')
    axs[0].legend(fontsize="x-small")

    # Graph 2 - Optimal selections over all steps
    axs[1].set_title("10-Armed TestBed - % Optimal Action")
    axs[1].plot(scores_two_list[0] * 100, label="epsilon = 0.1, " + modes[1])
    axs[1].plot(scores_two_list[1] * 100, label="epsilon = 0.1, " + modes[0])
    axs[1].set_ylim(0, 100)
    axs[1].set_ylabel('% Optimal Action')
    axs[1].set_xlabel('Steps')
    axs[1].legend(fontsize="x-small")

    plt.tight_layout()
    plt.savefig("figure/figs_hw1_1.png")
    plt.close()


def print_plots_hw2(scores_one_list, q_zero_list):
    plt.title("10-Armed TestBed - Recency Weighted Average Rewards")
    plt.plot(q_zero_list, scores_one_list, label="epsilon = 0.1")
    plt.ylabel('Average Reward')
    plt.xlabel('Q_zero')
    plt.legend(fontsize="small")
    plt.savefig("figure/figs_hw1_2.png")
    plt.close()
