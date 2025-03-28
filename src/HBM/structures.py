import numpy as np

class Feature:
    def __init__(self, label, values: list):
        self._label = label
        self._values = np.array(values)
        self._n = len(values)

    @property
    def n(self):
        return self._n

    @property
    def values(self):
        return self._values

    @property
    def label(self):
        return self._label

class Category:
    def __init__(self) -> None:
        self._n = 0
        self._items = []

    @property
    def n(self):
        return self._n

    @property
    def items(self):
        return self._items

    def add(self, features: Feature):
        self._n += 1
        self._items.append(features)
