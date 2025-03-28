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
        self._kappa0 = 1e-3

    @property
    def n(self):
        return self._n

    @property
    def items(self):
        return self._items

    @property
    def kappa0(self):
        return self._kappa0

    @property
    def nu0(self):
        return self._nu0

    @property
    def psi0(self):
        return self._psi0

    @property
    def mean_prior(self):
        return self._mean_prior

    def add(self, feature: Feature):
        self._n += 1
        self._items.append(feature)
        self._nu0 = feature.n + 1
        self._psi0 = np.eye(feature.n)
        self._mean_prior = np.zeros(feature.n)

    def mean(self):
        return np.mean([i.values for i in self._items], axis=0)

    def cov(self):
        return np.cov([i.values for i in self._items], rowvar=False)
