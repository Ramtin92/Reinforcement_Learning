import matplotlib.pyplot as plt

def print_plots(avg_rewards, episode_numbers, epsilon):
    plt.title("Average rewards vs. episode number")
    plt.plot(episode_numbers, avg_rewards, label="epsilon = {}".format(epsilon))
    plt.ylabel('Average rewards')
    plt.xlabel('Episode number')
    plt.legend(fontsize="small")
    plt.savefig("figures/track_1.png")
    plt.close()
