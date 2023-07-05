import numpy as np
import sys
sys.path.append("..")
from utils.solver import Solver


class EpsilonGreedy(Solver):
    """Epsilon Greedy策略

    Attributes:
        machine: 多臂老虎机模型
        epsilon: 阈值
        init_prob: 初始概率估计
    """
    def __init__(self, machine, epsilon, init_prob):
        super(EpsilonGreedy, self).__init__(machine=machine)
        self.epsilon = epsilon
        self.init_prob = init_prob
        self.estimates = [self.init_prob] * self.machine.arms

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.machine.arms)
        else:
            action = np.argmax(self.estimates)

        reward = self.machine.get_reward(action=action)

        self.estimates[action] += 1.0 / (self.count[action]+1) * (reward - self.estimates[action])

        return action


class DecayEpsilonGreedy(Solver):
    """Epsilon Greedy策略

    Attributes:
        machine: 多臂老虎机模型
        init_prob: 初始概率估计
    """
    def __init__(self, machine, init_prob):
        super(DecayEpsilonGreedy, self).__init__(machine=machine)
        self.init_prob = init_prob
        self.estimates = [self.init_prob] * self.machine.arms
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1

        if np.random.random() < 1.0 / self.total_count:
            action = np.random.randint(0, self.machine.arms)
        else:
            action = np.argmax(self.estimates)

        reward = self.machine.get_reward(action=action)

        self.estimates[action] += 1.0 / (self.count[action] + 1) * (reward - self.estimates[action])

        return action
