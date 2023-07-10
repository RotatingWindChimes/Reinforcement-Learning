import sys
from argparse import ArgumentParser
sys.path.append("..")
from utils.env import CliffWalkingEnv
from utils.DynaQ import DynaQ
from utils import config


class Trainer:
    def __init__(self, time_steps, show_plot):
        """训练类

        :param time_steps: 训练步数
        :param show_plot: 是否可视化
        """
        self.time_steps = time_steps
        self.show_plot = show_plot
        self.env = CliffWalkingEnv(nrow=4, ncol=12)

    def train(self):
        model = DynaQ(env=self.env, **config.__dict__["DYNAQ_PARAMS"])

        model.run(time_steps=self.time_steps)

        if self.show_plot:
            model.plot_return()


if __name__ == "__main__":
    parser = ArgumentParser(description="Model Parameters")

    parser.add_argument("--time_steps", "-ts", default=500, type=int)
    parser.add_argument("--show_plot", "-show", default=True, type=bool)

    options = parser.parse_args()

    Trainer(time_steps=options.time_steps, show_plot=options.show_plot).train()
