from collections import deque
import pickle
import os
import random
import numpy as np


class UnetMemory:
    def __init__(self, memory_size):
        self.buffer = deque()
        self.memory_size = memory_size

    def append(self, s_, out_s):

        self.buffer.append((s_, out_s))
        if len(self.buffer) >= self.memory_size:
            self.buffer.popleft()


    def sample(self, size):


        minibatch = random.sample(self.buffer, size)
        s_ = np.array([data[0] for data in minibatch])
        out_s = np.array([data[1] for data in minibatch])
        #labdepth = np.array([data[1][0] for data in minibatch])



        return s_, out_s

    def save(self, dir):
        file = os.path.join(dir, 'unetmemory.pickle')
        with open(file, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, dir):
        file = os.path.join(dir, 'unetmemory.pickle')
        with open(file, 'rb') as f:
            memory = pickle.load(f)
        return memory