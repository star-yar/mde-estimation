from __future__ import annotations

import typing as tp

import numpy as np

from .bootstrap import Bootstrap
from ..duration_estimator import Groups

TMeansSample = np.ndarray
MeansGroups = Groups[np.ndarray]


class BootstrapForMeans(Bootstrap[TMeansSample]):
    @staticmethod
    def estimate_metric(
        groups: MeansGroups, sample_size_axis: int = 0, **kwargs: tp.Any,
    ) -> Groups[tp.Union[float, np.ndarray]]:
        return Groups(
            np.mean(groups.control, axis=sample_size_axis),
            np.mean(groups.pilot, axis=sample_size_axis),
        )

    @staticmethod
    def bootstrap_sample(
        bootstrap_size: int, groups: MeansGroups, **kwargs: tp.Any,
    ) -> Groups[tp.Union[float, np.ndarray]]:
        return Groups(
            np.random.choice(groups.control, (groups.control.size, bootstrap_size)),
            np.random.choice(groups.pilot, (groups.pilot.size, bootstrap_size)),
        )
