import numpy as np
import scipy.special as sp
from .structures import Feature, Category

class CRP:
    def __init__(
        self,
        alpha = 1.0,
        delta = 1.0,
        strict: bool = False
    ) -> None:
        self._alpha = alpha
        self._delta = delta
        self._strict = strict
        self._n = 0
        self._clusters = []

    @property
    def clusters(self) -> list:
        return self._clusters

    def add(self, feature: Feature) -> int:
        self._n += 1
        ll = np.array(
            [self.ll(cluster, feature, "beta") for cluster in self._clusters]
        )
        prior = self.prior()
        p = np.append(
            prior * ll,
            self._alpha / (self._n + self._alpha - 1)
        )
        p /= np.sum(p)
        choice = np.random.choice(len(p), p=p)
        if choice == len(self._clusters):
            new_clusters = Category()
            new_clusters.add(feature)
            self._clusters.append(new_clusters)
        else:
            self._clusters[choice].add(feature)
        return choice

    def ll(self, clusters, feature: Feature, type: str = "beta"):
        if type == "beta":
            return self.beta_ll(clusters, feature)

    def beta_ll(self, cluster, feature: Feature):
        if not self._clusters:
            return 1
        n_class_features = 0
        for i in cluster.items:
            if self._strict:
                n_class_features += np.all(i.values == feature)
            else:
                n_class_features += np.mean(i.values == feature)
        return sp.beta(
            n_class_features + self._delta,
            cluster.n - n_class_features + self._delta
        ) / sp.beta(self._delta, self._delta)

    def prior(self) -> np.ndarray:
        return np.array([
            i.n / (self._n + self._alpha -1) for i in self._clusters
        ])

class StickyCRP(CRP):
    def __init__(
        self,
        alpha = 1.0,
        delta = 1.0,
        strict: bool = False,
        stickiness = 1.0
    ) -> None:
        super().__init__(alpha, delta, strict)
        self._stickiness = stickiness
        self._memory = 0

    def add(self, feature: Feature) -> int:
        self._memory = super().add(feature)
        return self._memory

    def piror(self) -> np.ndarray:
        p = super().prior()
        p[self._memory] += self._stickiness / (self._n + self._alpha - 1)
        return p
