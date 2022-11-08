from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..duration_estimator import Groups, SampleParams


@dataclass
class NormalWithConstantRateSampleParams(SampleParams):
    mean: float
    std: float
    n_users_per_day: int


def get_groups_from_normal_with_constant_new_users_rate(
        n_days: int, sample_params: NormalWithConstantRateSampleParams,
) -> Groups[np.ndarray]:
    sample_size = sample_params.n_users_per_day * n_days
    return Groups(
        np.random.normal(sample_params.mean, sample_params.std, size=sample_size),
        np.random.normal(sample_params.mean, sample_params.std, size=sample_size),
    )

