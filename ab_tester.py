from __future__ import annotations
from abc import ABC
import typing as tp
from dataclasses import dataclass

import numpy as np
from tqdm.auto import trange


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
    n_false_positive: int = 0
    n_false_negative: int = 0
    n_experiments_per_each: int = 0

    @property
    def false_positive_rate(self) -> float:
        return self.n_false_positive / self.n_experiments_per_each

    @property
    def false_negative_rate(self) -> float:
        return self.n_false_negative / self.n_experiments_per_each

    @property
    def total_rate(self):
        return self.false_positive_rate + self.false_negative_rate

    def __add__(self, other: object) -> TestErrors:
        if not isinstance(other, TestErrors):
            raise NotImplementedError(
                f'Can\'t add type{type(other).__name__} '
                f'to {self.__class__.__name__}'
            )
        return TestErrors(
            self.n_false_positive + other.n_false_positive,
            self.n_false_negative + other.n_false_negative,
            self.n_experiments_per_each + other.n_experiments_per_each
        )


@dataclass
class NormalDistributionSampleParams(SampleParams):
    mean: float
    std: float


@dataclass
class FoundEffect:
    given_effect: bool
    given_no_effect: bool

    def get_test_errors(self) -> TestErrors:
        return TestErrors(
            self.given_no_effect,
            not self.given_effect,
            n_experiments_per_each=1,
        )


TExperimentConductor = tp.Callable[[Groups, float, bool], FoundEffect]


def calculate_error_rates(
        effect: float,
        users_per_day: int,  # todo: make this a function of time
        sample_params: SampleParams,
        sample_generator: TSampleGenerator,
        max_days: int = 30,
        n_experiment_runs_per_day_simulation: int = 250,
        is_additive_effect: bool = False,
        experiment_conductor: TExperimentConductor = None,
        verbose: bool = False,
) -> tp.List[TestErrors]:
    """Returns error rates based on duration from 1 to max_days

    Args:
        effect: effect size in percents, possible values > 0,
            let's say 0.1 is passed then metric has increased on 10%.
        users_per_day: users increment per day
        max_days: max experiment duration
        sample_params: params passed to `sample_generator`
        sample_generator: generator of observation samples
        is_additive_effect: states if affect is added or multiplied
        n_experiment_runs_per_day_simulation: n_observations per day we get to estimate error rates
        experiment_conductor:
        verbose:
    """
    if experiment_conductor is None:
        experiment_conductor = conduct_experiments_using_bootstrap

    days_range = trange(1, max_days + 1) if verbose else range(1, max_days + 1)
    return [
        measure_error_rate(
            sample_size=n_days * users_per_day,
            effect=effect,
            sample_params=sample_params,
            sample_generator=sample_generator,
            is_additive_effect=is_additive_effect,
            n_iterations=n_experiment_runs_per_day_simulation,
            experiment_conductor=experiment_conductor,
        )
        for n_days in days_range
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
        experiment_conductor: TExperimentConductor,
        n_iterations: int = 250,
) -> TestErrors:
    test_errors = TestErrors()
    for _ in range(n_iterations):
        groups = sample_generator(sample_size, sample_params)
        experiment_results = experiment_conductor(
            groups, effect, is_additive_effect,
        )
        test_errors += experiment_results.get_test_errors()
    return test_errors


def conduct_experiments_using_bootstrap(
        groups: Groups,
        effect: float,
        is_additive_effect: bool,
        metric_estimator: TMetricEstimator = np.mean,
        boostrap_size: int = 1000,
) -> FoundEffect:
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
                inject_effect(sampled_metric_pilot, effect, is_additive_effect)
                - sampled_metric_control
        ),
        pointwise_estimation=(
                inject_effect(metric_pilot, effect, is_additive_effect)
                - metric_control
        )
    )
    return FoundEffect(
        given_effect=conf_interval_injected_effect_test.contains(0),
        given_no_effect=conf_interval_no_effect_test.contains(0),
    )


def inject_effect(metric: float, effect: float, is_additive: bool) -> float:
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
    total_error_rates = np.array([x.total_rate for x in error_rates])
    less_than_threshold = total_error_rates <= error_rate_threshold
    if not less_than_threshold.any():
        return len(error_rates)
    return 1 + np.where(less_than_threshold)[0].min()


if __name__ == '__main__':
    experiment_effect = 0.1
    experiment_users_per_day = 10
    experiment_max_days = 30
    experiment_error_rates = calculate_error_rates(
        effect=experiment_effect,
        users_per_day=experiment_users_per_day,
        sample_params=NormalDistributionSampleParams(100, 10),
        sample_generator=get_groups_samples_from_normal,
        max_days=experiment_max_days,
        is_additive_effect=False,
        verbose=True,
    )
    find_optimal_duration(experiment_error_rates, 0.2)
