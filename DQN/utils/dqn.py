import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


class QNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(in_features=state_dim, out_features=hidden_dim, bias=True)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=action_dim, bias=True)

    def forward(self, x):
        y = f.relu(self.fc1(x))

        return self.fc2(y)


class DQN:
    def __init__(self, env, hidden_dim, learning_rate, epsilon, gamma, device, target_update):
        self.env = env

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        self.target_q_net = QNet(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim)
        self.q_net = QNet(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim)

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = device
        self.target_update = target_update

        self.count = 0
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.return_list = []

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            state = torch.tensor(np.array(state), dtype=torch.float32).to(device=self.device)
            action = torch.argmax(self.q_net(state)).item()

        return action

    def update(self, transition_dict):
        self.count += 1

        states = torch.tensor(transition_dict["states"], dtype=torch.float32).to(device=self.device)
        actions = torch.tensor(transition_dict["actions"], dtype=torch.int64).reshape(-1, 1).to(device=self.device)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float32).reshape(-1, 1).to(device=self.device)
        next_states = torch.tensor(transition_dict["next_states"], dtype=torch.float32).to(device=self.device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float32).reshape(-1, 1).to(device=self.device)

        q_values = self.q_net(states).gather(1, actions)

        max_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_q_values * (1 - dones)

        dqn_loss = f.mse_loss(q_values, q_targets)

        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    def run(self, time_steps, exp_buffer, minimal_size, batch_size):
        for i in tqdm(range(time_steps)):

            # 本轮回报
            episode_return = 0

            # 初始化本轮环境
            state = self.env.reset()
            done = False

            while not done:
                action = self.take_action(state=state)
                next_state, reward, done, _ = self.env.step(action=action)

                exp_buffer.add(state=state, reward=reward, next_state=next_state, action=action, done=done)

                state = next_state
                episode_return += reward

                if exp_buffer.size() > minimal_size:
                    states, actions, rewards, next_states, dones = exp_buffer.sample(batch_size=batch_size)
                    transitions = {"states": states, "actions": actions, "rewards": rewards,
                                   "next_states": next_states, "dones": dones}

                    self.update(transition_dict=transitions)

            self.return_list.append(episode_return)

            if i % 50 == 0:
                print("Episode {}: reward {}".format(i, self.return_list[-1]))

    def plot_return(self):
        if not os.path.exists(os.path.join("..", "Images")):
            os.makedirs(os.path.join("..", "Images"))

        img_path = os.path.join("..", "Images")

        x_range = range(len(self.return_list))

        plt.plot(x_range, self.return_list, label="DQN")
        plt.savefig(os.path.join(img_path, "DQN.jpg"))

        plt.show()


class DoubleDQN:
    def __init__(self, env, hidden_dim, learning_rate, epsilon, gamma, device, target_update):
        self.env = env

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        self.target_q_net = QNet(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim)
        self.q_net = QNet(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim)

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = device
        self.target_update = target_update

        self.count = 0
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.return_list = []

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            state = torch.tensor(np.array(state), dtype=torch.float32).to(device=self.device)
            action = torch.argmax(self.q_net(state)).item()

        return action

    def update(self, transition_dict):
        self.count += 1

        states = torch.tensor(transition_dict["states"], dtype=torch.float32).to(device=self.device)
        actions = torch.tensor(transition_dict["actions"], dtype=torch.int64).reshape(-1, 1).to(device=self.device)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float32).reshape(-1, 1).to(device=self.device)
        next_states = torch.tensor(transition_dict["next_states"], dtype=torch.float32).to(device=self.device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float32).reshape(-1, 1).to(device=self.device)

        q_values = self.q_net(states).gather(1, actions)

        best_actions = self.q_net(next_states).max(1)[1].view(-1, 1)

        q_targets = rewards + self.gamma * self.target_q_net(next_states).gather(1, best_actions) * (1 - dones)

        dqn_loss = f.mse_loss(q_values, q_targets)

        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    def run(self, time_steps, exp_buffer, minimal_size, batch_size):
        for i in tqdm(range(time_steps)):

            # 本轮回报
            episode_return = 0

            # 初始化本轮环境
            state = self.env.reset()
            done = False

            while not done:
                action = self.take_action(state=state)
                next_state, reward, done, _ = self.env.step(action=action)

                exp_buffer.add(state=state, reward=reward, next_state=next_state, action=action, done=done)

                state = next_state
                episode_return += reward

                if exp_buffer.size() > minimal_size:
                    states, actions, rewards, next_states, dones = exp_buffer.sample(batch_size=batch_size)
                    transitions = {"states": states, "actions": actions, "rewards": rewards,
                                   "next_states": next_states, "dones": dones}

                    self.update(transition_dict=transitions)

            self.return_list.append(episode_return)

            if i % 50 == 0:
                print("Episode {}: reward {}".format(i, self.return_list[-1]))

    def plot_return(self):
        if not os.path.exists(os.path.join("..", "Images")):
            os.makedirs(os.path.join("..", "Images"))

        img_path = os.path.join("..", "Images")

        x_range = range(len(self.return_list))

        plt.plot(x_range, self.return_list, label="DQN")
        plt.savefig(os.path.join(img_path, "DQN.jpg"))

        plt.show()
