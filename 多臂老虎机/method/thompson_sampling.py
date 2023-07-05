import numpy as np
import sys
sys.path.append("..")
from utils.solver import Solver


class ThompsonSampling(Solver):
    """Thompson Sample策略

    Attributes:
        machine: 多臂老虎机模型
    """
    def __init__(self, machine):
        super(ThompsonSampling, self).__init__(machine=machine)
        self._a = np.ones(self.machine.arms)
        self._b = np.ones(self.machine.arms)

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)

        action = np.argmax(samples)

        reward = self.machine.get_reward(action=action)

        self._a[action] += reward
        self._b[action] += 1 - reward

        return action
