import argparse
import numpy as np
import agent, constants, environment, racetrack_options
from plot_functions import print_plots

NUM_TRAINING_EPISODES = 50000
NUM_EVALUATION_EPISODES = 100
EVALUATION_RATE = 1


def main(args):
    track = racetrack_options.TRACKS_OPTIONS[args.racetrack]
    env = environment.RaceTrack(track)
    mc = agent.MonteCarlo(env, 0.1)

    mean_return_each_episode = []
    for episode_idx in range(NUM_TRAINING_EPISODES):
        mc.play_episode()
        mc.update_policy()

        env.reset()

        if episode_idx > 0 and episode_idx % EVALUATION_RATE == 0:

            returns = []
            for _ in range(NUM_EVALUATION_EPISODES):
                ret, _ = mc.play_episode(explore=False, learn=False)
                returns.append(ret)
                env.reset()

            mean_return = np.mean(returns)
            mean_return_each_episode.append(mean_return)

    print_plots(mean_return_each_episode, np.arange(len(mean_return_each_episode)) + 1, args.epsilon)

    for i, start_coordinates in enumerate(env.start_coordinates):
        env.reset()
        ret, seq = mc.play_episode(explore=False, learn=False)

        save_path = "{:s}_{:d}.{:s}".format(args.save_path, i + 1, args.format)
        mc.plot_sequence(seq, save_path=save_path, show_legend=not args.disable_legend)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("racetrack.")

    parser.add_argument("--racetrack", default='track_1',
                        help="{} or {}".format(constants.RACETRACK_1, constants.RACETRACK_2))
    parser.add_argument("--epsilon", type=float, default=0.001, help='epsilon')
    parser.add_argument("--save-path", default='figures/', help="where to save the figure")
    parser.add_argument("--format", default="png", help="image format")
    parser.add_argument("--disable-legend", default=False, action="store_true",
                        help="disable legend in the racetrack image")

    my_parser = parser.parse_args()
    main(my_parser)
