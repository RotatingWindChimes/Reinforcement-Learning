import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os


class DynaQ:
    def __init__(self, env, alpha, gamma, epsilon, n_planning, n_action=4):
        """Dyna-Q算法类

        :param env: 环境
        :param alpha: TD target计算用到
        :param gamma: 折扣因子
        :param epsilon: epsilon-Greedy策略
        :param n_planning: Q-planning次数
        :param n_action: 动作数
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning = n_planning
        self.n_action = n_action

        self.state_size = self.env.nrow * self.env.ncol
        self.Q_table = np.zeros([self.state_size, self.n_action])
        self.model = {}
        self.return_list = []

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.n_action)
        else:
            action = np.argmax(self.Q_table[state])

        return action

    def q_learning(self, s0, a0, r0, s1):
        td_error = r0 + self.gamma * np.max(self.Q_table[s1]) - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

    def update(self, s0, a0, r0, s1):
        self.q_learning(s0, a0, r0, s1)
        self.model[(s0, a0)] = r0, s1

        for _ in range(self.n_planning):
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            self.q_learning(s, a, r, s_)

    def run(self, time_steps):
        for _ in tqdm(range(time_steps)):
            episode_return = 0
            state = self.env.reset()
            done = False

            while not done:
                action = self.get_action(state)
                next_state, reward, done = self.env.step(action=action)

                episode_return += reward
                self.update(state, action, reward, next_state)

                state = next_state

            self.return_list.append(episode_return)

    def plot_return(self):
        if not os.path.exists(os.path.join("..", "Images")):
            os.makedirs(os.path.join("..", "Images"))

        pic_path = os.path.join("..", "Images")

        x_range = range(len(self.return_list))

        plt.plot(x_range, self.return_list, label="Dyna-Q")
        plt.savefig(os.path.join(pic_path, "Dyna-Q.jpg"))
        plt.show()
