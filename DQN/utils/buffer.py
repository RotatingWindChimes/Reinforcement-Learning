from collections import deque
import random
import numpy as np


class Buffer:
    def __init__(self, capacity):
        """经验回放池类

        :param capacity: 回放池大小
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)

        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
