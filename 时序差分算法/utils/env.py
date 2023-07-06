class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        """悬崖漫步环境

        :param ncol: 列数
        :param nrow: 行数
        """
        self.ncol = ncol
        self.nrow = nrow
        self.x = 0
        self.y = nrow - 1

    def step(self, action):
        """step函数

        :param action: 动作的索引
        :return: 更改后的状态、奖励、是否结束
        """
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]

        self.x = min(self.ncol - 1, max(self.x + change[action][0], 0))
        self.y = min(self.nrow - 1, max(self.y + change[action][1], 0))

        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False

        if self.y == self.nrow - 1 and self.x > 0:
            done = True

            if self.x != self.ncol - 1:
                reward = -100

        return next_state, reward, done

    def reset(self):
        """重置环境

        :return: 初始环境
        """
        self.x = 0
        self.y = self.nrow - 1

        return self.y * self.ncol + self.x
