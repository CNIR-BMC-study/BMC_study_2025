import numpy as np
import random

class NumberGame:
    def __init__(self,threshold = 0.5):
        self._threshold = threshold
        self._concept = self.make_concept()

    def make_concept(self):
        num_list = np.linspace(1, 100, 100, dtype=np.int64)
        idx = [random.random() >= self._threshold for _ in range(100)]
        return num_list[idx]

    def show(self):
        return self._concept

    def reset(self):
        self._concept = self.make_concept()

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, x):
        self._threshold = x
