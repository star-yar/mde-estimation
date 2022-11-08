from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import typing as tp

import numpy as np

from ..duration_estimator import Effect, FoundEffect, Groups

TSample = tp.TypeVar('TSample')


@dataclass
class ConfInterval:
    left: float
    right: float

    def contains(self, value: float) -> bool:
        return self.left <= value <= self.right


def get_ci_bootstrap_pivotal(
        bootstraped_estimations: np.ndarray, pointwise_estimation: float, alpha: float = 0.05,
) -> ConfInterval:
    """
    Estimates central conf interval

    Args:
        bootstraped_estimations: metric estimation using bootstrap
        pointwise_estimation: pointwise metric estimation
        alpha: conf level
    """
    alpha_tail = alpha / 2
    left_quantile = np.quantile(bootstraped_estimations, alpha_tail)
    right_quantile = np.quantile(bootstraped_estimations, 1 - alpha_tail)
    return ConfInterval(
        2 * pointwise_estimation - right_quantile,
        2 * pointwise_estimation - left_quantile,
    )


class Bootstrap(ABC, tp.Generic[TSample]):
    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        self._metric_kwargs = {}
        self._bootstrap_sample_kwargs = {}

    @staticmethod
    @abstractmethod
    def estimate_metric(
            groups: Groups[TSample],
            sample_size_axis: int = 0,
            **metric_kwargs: tp.Any,
    ) -> Groups[TSample]:
        pass

    @staticmethod
    @abstractmethod
    def bootstrap_sample(
            bootstrap_size: int,
            groups: Groups[TSample],
            **kwargs: tp.Any,
    ) -> Groups[TSample]:
        pass

    def __call__(
            self,
            groups: Groups[TSample],
            effect: Effect,
            boostrap_size: int = 1000,
    ) -> FoundEffect:
        metrics = self.estimate_metric(groups, **self._metric_kwargs)
        boostrap_samples = self.bootstrap_sample(boostrap_size, groups, **self._bootstrap_sample_kwargs)
        sampled_metrics = self.estimate_metric(boostrap_samples, **self._metric_kwargs)
        assert len(sampled_metrics.control) == boostrap_size
        assert len(sampled_metrics.pilot) == boostrap_size
        conf_interval_no_effect_test = get_ci_bootstrap_pivotal(
            bootstraped_estimations=sampled_metrics.pilot - sampled_metrics.control,
            pointwise_estimation=metrics.pilot - metrics.control,
        )
        conf_interval_injected_effect_test = get_ci_bootstrap_pivotal(
            bootstraped_estimations=effect.inject(sampled_metrics.pilot) - sampled_metrics.control,
            pointwise_estimation=effect.inject(metrics.pilot) - metrics.control,
        )
        return FoundEffect(
            given_effect=not conf_interval_injected_effect_test.contains(0),
            given_no_effect=not conf_interval_no_effect_test.contains(0),
        )
