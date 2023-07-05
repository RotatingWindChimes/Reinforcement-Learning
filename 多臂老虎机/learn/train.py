from argparse import ArgumentParser
import sys
sys.path.append("..")
from method.epsilon_greedy import EpsilonGreedy, DecayEpsilonGreedy
from method.ucb import UCB
from method.thompson_sampling import ThompsonSampling
from utils.env import MultiArmMachine
from utils import config


# 算法名称和对应模型
ALGORITHMS = {"epsilon_greedy": EpsilonGreedy, "decay_epsilon_greedy": DecayEpsilonGreedy, "ucb": UCB,
              "thompson": ThompsonSampling}


class Trainer:
    """训练类

    Attributes:
        name: 算法名称
        time_steps: 训练次数
        show_regret: 是否可视化累积误差
    """
    def __init__(self, name, time_steps, show_regret):
        self.name = name
        self.time_steps = time_steps
        self.show_regret = show_regret
        self.machine = MultiArmMachine(arms=10)

    def train(self):
        model = ALGORITHMS[self.name](machine=self.machine, **config.__dict__["{}_PARAMS".format(self.name.upper())])

        model.run(num_steps=self.time_steps)

        if self.show_regret:
            model.plot_regrets(solver_name=self.name)


if __name__ == "__main__":
    parser = ArgumentParser(description="Settings for model.")

    parser.add_argument("--name", "-n", default="epsilon_greedy", type=str)
    parser.add_argument("--time_steps", "-ts", default=500, type=int)
    parser.add_argument("--show_regret", "-show", default=True, type=bool)

    options = parser.parse_args()

    Trainer(name=options.name, time_steps=options.time_steps, show_regret=options.show_regret).train()
