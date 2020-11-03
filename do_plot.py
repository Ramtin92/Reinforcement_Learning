import matplotlib.pyplot as plt


def plot_figures(DynaQ_5, DynaQ_50, DynaQPlus_5, DynaQPlus_50, DynaQPlus_5_EXP,
                 DynaQPlus_50_EXP):
    plt.figure(figsize=(10, 10))
    plt.ylim(top=400)
    plt.title('DynaQ vs. DynaQ+ vs. DynaQPlus Extra Experiment for Maze', fontsize='xx-large')
    plt.xlabel('Episodes', fontsize='xx-large')
    plt.ylabel('Steps', fontsize='xx-large')
    plt.plot(DynaQ_5, '-', c='blue', label='DynaQ 5 steps')
    plt.plot(DynaQ_50, '-', c='green', label='DynaQ 50 steps')
    plt.plot(DynaQPlus_5, '-', c='red', label='DynaQPlus 5 steps')
    plt.plot(DynaQPlus_50, '-', c='orange', label='DynaQPlus 50 steps')
    plt.plot(DynaQPlus_5_EXP, '-', c='gray', label='DynaQPlus Extra Experiment 5 steps')
    plt.plot(DynaQPlus_50_EXP, '-', c='black', label='DynaQPlus Extra Experiment 50 steps')
    plt.legend(loc='best', prop={'size': 12})
    plt.savefig("figures/1.png")

    plt.figure(figsize=(10, 10))
    plt.ylim(top=70)
    plt.title('DynaQ vs. DynaQ+ for Maze', fontsize='xx-large')
    plt.xlabel('Episodes', fontsize='xx-large')
    plt.ylabel('Steps', fontsize='xx-large')
    plt.plot(DynaQ_5, '-', c='green', label='DynaQ 5 steps')
    plt.plot(DynaQPlus_5, '-', c='blue', label='DynaQPlus 5 steps')
    plt.legend(loc='best', prop={'size': 12})
    plt.savefig("figures/2.png")

    plt.figure(figsize=(10, 10))
    plt.ylim(top=70)
    plt.title('DynaQ+ vs. DynaQ+ Extra Experiment for Maze', fontsize='xx-large')
    plt.xlabel('Episodes', fontsize='xx-large')
    plt.ylabel('Steps', fontsize='xx-large')
    plt.plot(DynaQPlus_50, '-', c='green', label='DynaQPlus 50 steps')
    plt.plot(DynaQPlus_50_EXP, '-', c='blue', label='DynaQPlus Extra Experiment 50 steps')
    plt.legend(loc='best', prop={'size': 12})
    plt.savefig("figures/3.png")
