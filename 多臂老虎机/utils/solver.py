import os
import numpy as np
import matplotlib.pyplot as plt


class Solver:
    """解决多臂老虎机问题的算法类

    Attributes:
        machine: 多臂老虎机模型
    """
    def __init__(self, machine):
        self.machine = machine
        self.regret = 0
        self.regrets = []
        self.actions = []
        self.count = np.zeros(self.machine.arms)

    # 选择一个动作, 依赖于具体算法
    def run_one_step(self):
        raise NotImplementedError

    # 更新累积懊悔
    def upgrade(self, action):
        regret = self.machine.best_prob - self.machine.probs[action]  # 当前步的懊悔

        self.regret += regret               # 累积懊悔
        self.regrets.append(self.regret)

    # 执行若干步
    def run(self, num_steps):
        for _ in range(num_steps):
            action = self.run_one_step()
            self.actions.append(action)
            self.count[action] += 1
            self.upgrade(action=action)

    # 可视化累积懊悔
    def plot_regrets(self, solver_name):
        if not os.path.exists(os.path.join("..", "Images")):
            os.makedirs(os.path.join("..", "Images"))

        img_path = os.path.join("..", "Images")

        x_range = range(len(self.regrets))
        plt.plot(x_range, self.regrets, label=solver_name)
        plt.savefig(os.path.join(img_path, "{}.jpg").format(solver_name))
        plt.show()
