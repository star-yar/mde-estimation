from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..duration_estimator import Groups


@dataclass
class NormalDistributionWithConstantDailyGrowth(object):
    mean: float
    std: float
    new_observations_per_day: int
    random_gen: np.random.RandomState = np.random.RandomState()

    def __call__(
        self,
        n_days: int,
    ) -> Groups[np.ndarray]:
        if n_days <= 0:
            raise ValueError("Can't simulate for less then one day")
        sample_size = self.new_observations_per_day * n_days
        return Groups(
            self.random_gen.normal(self.mean, self.std, size=sample_size),
            self.random_gen.normal(self.mean, self.std, size=sample_size),
        )

