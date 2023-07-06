import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


class Sarsa:
    def __init__(self, env, alpha, epsilon, gamma, n_action=4):
        """Sarsa算法类

        :param env: 环境
        :param alpha: TD target计算时用到
        :param epsilon: epsilon-Greedy
        :param gamma: TD target计算时用到
        :param n_action: 动作空间大小
        """
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_action = n_action

        self.nrow = self.env.nrow
        self.ncol = self.env.ncol
        self.state_size = self.nrow * self.ncol
        self.Q_table = np.zeros((self.state_size, self.n_action))
        self.return_list = []

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.n_action)
        else:
            action = np.argmax(self.Q_table[state])

        return action

    def update(self, s0, a0, s1, a1, r):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

    def best_action(self, state):
        q_max = np.max(self.Q_table[state])
        a = [0] * self.n_action

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
                self.update(s0=state, a0=action, s1=next_state, a1=next_action, r=reward)
                state = next_state
                action = next_action

                if done:
                    break

            self.return_list.append(episode_return)

    def plot_return(self):
        if not os.path.exists(os.path.join("..", "Images")):
            os.makedirs(os.path.join("..", "Images"))

        pic_path = os.path.join("..", "Images")

        x_range = range(len(self.return_list))

        plt.plot(x_range, self.return_list, label="Sarsa")
        plt.savefig(os.path.join(pic_path, "Sarsa.jpg"))

        plt.show()
