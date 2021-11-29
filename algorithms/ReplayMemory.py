from collections import deque
import pickle
import os
import random
import numpy as np


class ReplayMemory:
    def __init__(self, memory_size):
        self.buffer = deque()
        self.sucBuffer = deque()
        self.memory_size = memory_size

    def append(self, pre_state, action, reward, post_state, terminal, info):
        if info == "success":
            self.sucBuffer.append((pre_state, action, reward, post_state, terminal))
        else:
            self.buffer.append((pre_state, action, reward, post_state, terminal))
        if len(self.buffer) >= self.memory_size:
            self.buffer.popleft()
        if len(self.sucBuffer) >= self.memory_size:
            self.buffer.popleft()

    def sample(self, size):
        size_1 = int(size * 0.875)
        size_2 = int(size * 0.125)

        minibatch_1 = random.sample(self.buffer, size_1)
        #states_1 = np.array([data[0][0] for data in minibatch_1])
        states_1 = np.array([data[0] for data in minibatch_1])
        actions_1 = np.array([data[1] for data in minibatch_1])
        rewards_1 = np.array([data[2] for data in minibatch_1])
        next_states_1 = np.array([data[3] for data in minibatch_1])
        terminals_1 = np.array([data[4] for data in minibatch_1])

        if len(self.sucBuffer) >= size_2:
            minibatch_2 = random.sample(self.sucBuffer, size_2)
        else:
            minibatch_2 = random.sample(self.buffer, size_2)
        #states_2 = np.array([data[0][0] for data in minibatch_2])
        states_2 = np.array([data[0] for data in minibatch_2])
        actions_2 = np.array([data[1] for data in minibatch_2])
        rewards_2 = np.array([data[2] for data in minibatch_2])
        next_states_2 = np.array([data[3] for data in minibatch_2])
        terminals_2 = np.array([data[4] for data in minibatch_2])

        states = np.concatenate((states_1, states_2), axis=0)
        actions = np.concatenate((actions_1, actions_2), axis=0)
        rewards = np.concatenate((rewards_1, rewards_2), axis=0)
        next_states = np.concatenate((next_states_1, next_states_2), axis=0)
        terminals = np.concatenate((terminals_1, terminals_2), axis=0)

        return states, actions, rewards, next_states, terminals

    def save(self, dir):
        file = os.path.join(dir, 'replaymemory.pickle')
        print("save buffer")
        with open(file, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, dir):
        file = os.path.join(dir, 'replaymemory.pickle')
        with open(file, 'rb') as f:
            memory = pickle.load(f)
        return memory