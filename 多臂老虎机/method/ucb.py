import numpy as np
import sys
sys.path.append("..")
from utils.solver import Solver


class UCB(Solver):
    """UCB策略

    Attributes:
        machine: 多臂老虎机模型
        coefficient: UCB加权系数
        init_prob: 初始概率估计
    """
    def __init__(self, machine, coefficient, init_prob):
        super(UCB, self).__init__(machine=machine)
        self.coefficient = coefficient
        self.init_prob = init_prob
        self.estimates = [self.init_prob] * self.machine.arms
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1

        ucb = np.sqrt(np.log(self.total_count)) / 2 / (1 + self.count) + self.estimates

        action = np.argmax(ucb)
        reward = self.machine.get_reward(action=action)

        self.estimates[action] += 1.0 / (1.0 + self.count[action]) * (reward - self.estimates[action])

        return action
