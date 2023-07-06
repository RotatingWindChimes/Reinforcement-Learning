import sys
from argparse import ArgumentParser
sys.path.append("..")
from utils.env import CliffWalkingEnv
from utils.Sarsa import Sarsa
from utils.QLearning import QLearning
from utils import config


CLASSNAMES = {"sarsa": Sarsa, "qlearning": QLearning}


class Trainer:
    def __init__(self, time_steps, show_plot, algorithm_name):
        self.time_steps = time_steps
        self.show_plot = show_plot
        self.algorithm_name = algorithm_name

    def train(self):
        env = CliffWalkingEnv(ncol=12, nrow=4)
        model = CLASSNAMES[self.algorithm_name](env=env, **config.__dict__["{}_PARAMS".
                                                format(self.algorithm_name.upper())])

        model.run(timesteps=self.time_steps)

        if self.show_plot:
            model.plot_return()


if __name__ == "__main__":
    parser = ArgumentParser(description="Parameters for training.")

    parser.add_argument("--name", "-n", default="qlearning", type=str)
    parser.add_argument("--time_steps", "-ts", default=500, type=int)
    parser.add_argument("--show_plot", "-show", default=True, type=bool)

    options = parser.parse_args()

    Trainer(time_steps=options.time_steps, show_plot=options.show_plot, algorithm_name=options.name).train()
