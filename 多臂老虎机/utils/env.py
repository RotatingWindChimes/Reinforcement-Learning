import numpy as np


class MultiArmMachine:
    """ 多臂老虎机类

    Attributes:
        arms: 臂数
    """
    def __init__(self, arms):
        self.arms = arms
        self.probs = np.random.random(self.arms)
        self.best_prob_id = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_prob_id]

    # 返回动作奖励
    def get_reward(self, action):
        if np.random.rand() < self.probs[action]:
            return 1
        else:
            return 0


if __name__ == "__main__":
    machine = MultiArmMachine(arms=10)
    print("这是一个{}臂老虎机，各臂获奖概率为{}".format(machine.arms, machine.probs))

    print("选择1号杆，获得奖励为{}".format(machine.get_reward(action=1)))
