from abc import ABC
import typing as tp
from dataclasses import dataclass

import numpy as np


@dataclass
class SampleParams(ABC):
    pass


@dataclass
class Groups:
    control: np.ndarray
    pilot: np.ndarray


TMetricEstimator = tp.Callable[[np.ndarray], float]
TSampleGenerator = tp.Callable[[int, SampleParams], Groups]


@dataclass
class ConfInterval:
    left: float
    right: float

    def contains(self, value: float) -> bool:
        return self.left <= value <= self.right


@dataclass
class TestErrors:
    n_false_positive: int
    n_false_negative: int
    n_experiments_per_each: int

    @property
    def false_positive_rate(self) -> float:
        return self.n_false_positive / self.n_experiments_per_each

    @property
    def false_negative_rate(self) -> float:
        return self.n_false_negative / self.n_experiments_per_each

    @property
    def total_rate(self):
        return self.false_positive_rate + self.false_negative_rate


@dataclass
class NormalDistributionSampleParams:
    mean: float
    std: float


@dataclass
class ExperimentResults:
    no_effect_test: bool
    injected_effect_test: bool


TExperimentConductor = tp.Callable[[Groups, float, bool], ExperimentResults]


def calculate_error_rates(
        effect: float,
        users_per_day: int,  # todo: make this a function of time
        sample_params: SampleParams,
        sample_generator: TSampleGenerator,
        max_days: int = 30,
        is_additive_effect: bool = False,
) -> tp.List[TestErrors]:
    """Returns error rates based on duration from 1 to max_days

    Args:
        effect: effect size in percents, possible values > 0,
            let's say 0.1 is passed then metric has increased on 10%.
        users_per_day: users increment per day
        max_days: max experiment duration
        sample_params:
        sample_generator:
        is_additive_effect: states if affect is added or multiplied
    """
    return [
        measure_error_rate(
            sample_size=n_days * users_per_day,
            effect=effect,
            sample_params=sample_params,
            sample_generator=sample_generator,
            is_additive_effect=is_additive_effect,
        )
        for n_days in range(1, max_days + 1)
    ]


def get_groups_samples_from_normal(
        sample_size: int, sample_params: NormalDistributionSampleParams,
) -> Groups:
    return Groups(
        np.random.normal(sample_params.mean, sample_params.std, size=sample_size),
        np.random.normal(sample_params.mean, sample_params.std, size=sample_size),
    )


def measure_error_rate(
        sample_size: int,
        effect: float,
        sample_params: SampleParams,
        sample_generator: TSampleGenerator,
        is_additive_effect: bool,
        n_iterations: int = 250,
        experiment_conductor: TExperimentConductor = None,
) -> TestErrors:
    if experiment_conductor is None:
        experiment_conductor = conduct_experiment_using_bootstrap

    n_false_positive_effects = 0
    n_undiscovered_effects = 0
    for _ in range(n_iterations):
        groups = sample_generator(sample_size, sample_params)
        found_effect_aa_test, found_effect_ab_test = experiment_conductor(
            groups, effect, is_additive_effect,
        )
        n_false_positive_effects += found_effect_aa_test
        n_undiscovered_effects += not found_effect_ab_test
    return TestErrors(n_false_positive_effects, n_undiscovered_effects, n_iterations)


def conduct_experiment_using_bootstrap(
        groups: Groups,
        effect: float,
        is_additive_effect: bool,
        metric_estimator: TMetricEstimator = np.mean,
        boostrap_size: int = 1000,
) -> ExperimentResults:
    metric_pilot = metric_estimator(groups.pilot)
    metric_control = metric_estimator(groups.control)
    boostrap_samples = bootstrap_samples(boostrap_size, groups)
    sampled_metric_control = np.apply_along_axis(metric_estimator, axis=1, arr=boostrap_samples.control)
    sampled_metric_pilot = np.apply_along_axis(metric_estimator, axis=1, arr=boostrap_samples.pilot)
    conf_interval_no_effect_test = get_ci_bootstrap_pivotal(
        bootstraped_estimations=sampled_metric_pilot - sampled_metric_control,
        pointwise_estimation=metric_pilot - metric_control
    )
    conf_interval_injected_effect_test = get_ci_bootstrap_pivotal(
        bootstraped_estimations=(
                _inject_effect(sampled_metric_pilot, effect, is_additive_effect)
                - sampled_metric_control
        ),
        pointwise_estimation=(
                _inject_effect(metric_pilot, effect, is_additive_effect)
                - metric_control
        )
    )
    return ExperimentResults(
        _is_significant_diff(conf_interval_no_effect_test),
        _is_significant_diff(conf_interval_injected_effect_test),
    )


def _inject_effect(metric: float, effect: float, is_additive: bool) -> float:
    if is_additive:
        return metric + effect
    else:
        return metric * (1 + effect)


def bootstrap_samples(boostrap_size: int, groups: Groups) -> Groups:
    return Groups(
        np.random.choice(groups.control, (boostrap_size, groups.control.size)),
        np.random.choice(groups.pilot, (boostrap_size, groups.pilot.size)),
    )


def _is_significant_diff(confidence_interval: ConfInterval) -> bool:
    return not confidence_interval.contains(0)


def get_ci_bootstrap_pivotal(
        bootstraped_estimations: np.ndarray, pointwise_estimation: float, alpha: float = 0.05,
) -> ConfInterval:
    """
    Estimates central conf interval

    Args:
        bootstraped_estimations: значения метрики, полученные с помощью бутстрепа
        pointwise_estimation: точечная оценка метрики
        alpha: уровень значимости
    """
    alpha_tail = alpha / 2
    left_quantile = np.quantile(bootstraped_estimations, alpha_tail)
    right_quantile = np.quantile(bootstraped_estimations, 1 - alpha_tail)
    return ConfInterval(
        2 * pointwise_estimation - right_quantile,
        2 * pointwise_estimation - left_quantile,
    )


def find_optimal_duration(error_rates: tp.List[TestErrors], error_rate_threshold: float) -> int:
    """
    Args:
        error_rates:
        error_rate_threshold: FP + FN error rates from interval [0.05, 0.5].

    Returns:

    """
    total_error_rates = [x.total_rate for x in error_rates]
    less_than_threshold = np.sum(total_error_rates, axis=1) <= error_rate_threshold
    if not less_than_threshold.any():
        return len(error_rates)
    return 1 + np.where(less_than_threshold)[0].min()


if __name__ == '__main__':
    pass
