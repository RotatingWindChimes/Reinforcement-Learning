import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


class QLearning:
    def __init__(self, env, epsilon, alpha, gamma, n_action=4):
        """QLearning

        :param env: 环境
        :param epsilon: epsilon-greedy策略
        :param alpha: 计算TD error时用到
        :param gamma: 计算TD error时用到
        :param n_action: 动作空间大小
        """
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.n_action = n_action

        self.state_size = self.env.nrow * self.env.ncol
        self.Q_table = np.zeros([self.state_size, self.n_action])
        self.return_list = []

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.n_action)
        else:
            action = np.argmax(self.Q_table[state])

        return action

    def update(self, s0, a0, r, s1):
        td_error = r + self.gamma * np.max(self.Q_table[s1]) - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

    def best_action(self, state):
        q_max = np.max(self.Q_table[state])
        a = np.zeros(self.n_action)

        for i in range(self.n_action):
            if self.Q_table[state, i] == q_max:
                a[i] = 1

        return a

    def run(self, timesteps):
        for _ in tqdm(range(timesteps)):

            # 一轮训练的总回报
            episode_return = 0

            # 环境初始化
            state = self.env.reset()
            action = self.get_action(state=state)
            done = False

            while not done:
                next_state, reward, done = self.env.step(action=action)
                episode_return += reward

                next_action = self.get_action(next_state)
                self.update(s0=state, a0=action, s1=next_state, r=reward)
                state = next_state
                action = next_action

            self.return_list.append(episode_return)

    def plot_return(self):
        if not os.path.exists(os.path.join("..", "Images")):
            os.makedirs(os.path.join("..", "Images"))

        pic_path = os.path.join("..", "Images")

        x_range = range(len(self.return_list))

        plt.plot(x_range, self.return_list, label="Q_Learning")
        plt.savefig(os.path.join(pic_path, "Q_Learning.jpg"))

        plt.show()
