import sys
import gym
from argparse import ArgumentParser
sys.path.append("..")
from utils import config
from utils.dqn import DQN, DoubleDQN
from utils.buffer import Buffer


class Trainer:
    def __init__(self, time_steps, env_name, show_plot):
        """ 训练类

        :param time_steps: 训练步数
        :param env_name: 环境名
        :param show_plot: 是否可视化奖励
        """
        self.time_steps = time_steps
        self.env_name = env_name
        self.show_plot = show_plot

    def train(self):
        env = gym.make(self.env_name)

        # model = DQN(env=env, **config.__dict__["DQN_PARAMS"])
        model = DoubleDQN(env=env, **config.__dict__["DQN_PARAMS"])
        exp_buffer = Buffer(**config.__dict__["BUFFER_PARAMS"])

        model.run(time_steps=self.time_steps, exp_buffer=exp_buffer, **config.__dict__["TRAIN_PARAMS"])

        if self.show_plot:
            model.plot_return()


if __name__ == "__main__":
    parser = ArgumentParser(description="Params for training.")

    parser.add_argument("--time_steps", "-ts", default=500, type=int)
    parser.add_argument("--env_name", "-n", default="CartPole-v0", type=str)
    parser.add_argument("--show_plot", "-show", default=True, type=bool)

    options = parser.parse_args()

    Trainer(time_steps=options.time_steps, env_name=options.env_name, show_plot=options.show_plot).train()
