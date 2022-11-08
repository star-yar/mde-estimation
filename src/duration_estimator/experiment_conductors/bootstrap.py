from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import typing as tp

import numpy as np

from duration_estimator import Effect, FoundEffect, Groups, TSingleGroup


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


class ConductUsingBootstrap(ABC):
    @staticmethod
    @abstractmethod
    def metric_estimator(sample: TSingleGroup, axis: int = 0) -> tp.Union[float, np.ndarray]:
        pass

    @staticmethod
    @abstractmethod
    def sample_bootstrapper(bootstrap_size: int, groups: Groups) -> Groups:
        pass

    def __call__(
            self,
            groups: Groups[tp.Any],
            effect: Effect,
            boostrap_size: int = 1000,
    ) -> FoundEffect:
        metric_pilot = self.metric_estimator(groups.pilot)
        metric_control = self.metric_estimator(groups.control)
        boostrap_samples = self.sample_bootstrapper(boostrap_size, groups)
        sampled_metric_control = self.metric_estimator(boostrap_samples.control)
        assert len(sampled_metric_control) == boostrap_size
        sampled_metric_pilot = self.metric_estimator(boostrap_samples.pilot)
        assert len(sampled_metric_pilot) == boostrap_size
        conf_interval_no_effect_test = get_ci_bootstrap_pivotal(
            bootstraped_estimations=sampled_metric_pilot - sampled_metric_control,
            pointwise_estimation=metric_pilot - metric_control,
        )
        conf_interval_injected_effect_test = get_ci_bootstrap_pivotal(
            bootstraped_estimations=effect.inject(sampled_metric_pilot) - sampled_metric_control,
            pointwise_estimation=effect.inject(metric_pilot) - metric_control,
        )
        return FoundEffect(
            given_effect=not conf_interval_injected_effect_test.contains(0),
            given_no_effect=not conf_interval_no_effect_test.contains(0),
        )
