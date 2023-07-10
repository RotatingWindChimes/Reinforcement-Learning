class CliffWalkingEnv:
    def __init__(self, nrow, ncol):
        """悬崖漫步环境

        :param nrow: 行数
        :param ncol: 列数
        """
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0
        self.y = self.nrow - 1

    def step(self, action):
        """执行动作

        :param action: 动作索引
        :return: 状态、奖励、flag
        """
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))

        reward = -1
        state = self.y * self.ncol + self.x
        done = False

        if self.y == self.nrow - 1 and self.x > 0:
            done = True

            if self.x != self.ncol - 1:
                reward = -100

        return state, reward, done

    def reset(self):
        """初始化

        :return: 初始状态
        """
        self.x = 0
        self.y = self.nrow - 1

        return self.y * self.ncol + self.x
