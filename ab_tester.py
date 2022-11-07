from __future__ import annotations
from abc import ABC
from pathlib import Path
import typing as tp
from dataclasses import dataclass
from logging import getLogger

import numpy as np
from tqdm.auto import trange
from joblib import Parallel, delayed


logger = getLogger(__file__)


@dataclass
class Effect:
    size: float
    is_additive: bool

    def inject(self, metric: float) -> float:
        return metric + self.size if self.is_additive else metric * (1 + self.size)


@dataclass
class SampleParams(ABC):
    pass


@dataclass
class Groups:
    control: np.ndarray
    pilot: np.ndarray


TMetricEstimator = tp.Callable[[np.ndarray], float]
TSampleGenerator = tp.Callable[[int, SampleParams], Groups]
TSampleBootstraper = tp.Callable[[int, Groups], Groups]


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

    def __repr__(self) -> str:
        return (
            f"FP: {self.n_false_positive}, "
            f"FN: {self.n_false_negative}, "
            f"total: {self.total_rate}"
        )


@dataclass
class NormalDistributionWithConstantRateSampleParams(SampleParams):
    mean: float
    std: float
    n_users_per_day: int


@dataclass
class FoundEffect:
    given_effect: bool
    given_no_effect: bool

    def get_test_errors(self) -> TestErrors:
        return TestErrors(
            n_false_positive=self.given_no_effect,
            n_false_negative=not self.given_effect,
            n_experiments_per_each=1,
        )


TExperimentConductor = tp.Callable[[Groups, Effect], FoundEffect]


def calculate_error_rates(
        effect: Effect,
        sample_params: SampleParams,
        sample_generator: TSampleGenerator,
        max_days: int = 30,
        n_experiment_runs_per_day_simulation: int = 250,
        experiment_conductor: TExperimentConductor = None,
        verbose: bool = False,
) -> tp.List[TestErrors]:
    """Returns error rates based on duration from 1 to max_days

    Args:
        effect: effect size in percents, possible values > 0,
            let's say 0.1 is passed then metric has increased on 10%.
        max_days: max experiment duration
        sample_params: params passed to `sample_generator`
        sample_generator: generator of observation samples
        n_experiment_runs_per_day_simulation: n_observations per day we get to estimate error rates
        experiment_conductor:
        verbose:
    """
    if experiment_conductor is None:
        experiment_conductor = conduct_experiments_using_bootstrap

    days_range = trange(1, max_days + 1) if verbose else range(1, max_days + 1)
    error_rates = []
    for n_days in days_range:
        error_rates.append(
            measure_error_rate(
                n_days=n_days,
                effect=effect,
                sample_params=sample_params,
                sample_generator=sample_generator,
                n_iterations=n_experiment_runs_per_day_simulation,
                experiment_conductor=experiment_conductor,
                verbose=verbose,
            )
        )
        if verbose:
            logger.info(f'Current error rates estimation: {error_rates}.')
    return error_rates


def get_groups_for_normal_with_constant_users_per_day_rate(
        n_days: int, sample_params: NormalDistributionWithConstantRateSampleParams,
) -> Groups:
    sample_size = sample_params.n_users_per_day * n_days
    return Groups(
        np.random.normal(sample_params.mean, sample_params.std, size=sample_size),
        np.random.normal(sample_params.mean, sample_params.std, size=sample_size),
    )


def measure_error_rate(
        n_days: int,
        effect: Effect,
        sample_params: SampleParams,
        sample_generator: TSampleGenerator,
        experiment_conductor: TExperimentConductor,
        n_iterations: int = 250,
        verbose: bool = False,
        n_jobs: int = -1,
) -> TestErrors:
    test_errors = Parallel(n_jobs=n_jobs)(
        delayed(_measure_one_error)(
            effect, experiment_conductor, n_days, sample_generator, sample_params,
        )
        for _ in (trange(n_iterations, leave=False) if verbose else range(n_iterations))
    )
    return sum(test_errors, TestErrors())


def _measure_one_error(
        effect: Effect,
        experiment_conductor: TExperimentConductor,
        n_days: int,
        sample_generator: TSampleGenerator,
        sample_params: SampleParams,
) -> TestErrors:
    groups = sample_generator(n_days, sample_params)
    experiment_results = experiment_conductor(groups, effect)
    return experiment_results.get_test_errors()


def conduct_experiments_using_bootstrap(
        groups: Groups,
        effect: Effect,
        metric_estimator: TMetricEstimator = np.mean,
        boostrap_size: int = 1000,
        sample_bootstrapper: TSampleBootstraper = None,
        estimator_works_with_bootstrap_sample: bool = False,
) -> FoundEffect:
    """
    Args:
        groups: pilot & control samples to use for estimation
        effect:
        metric_estimator:
        boostrap_size:
        sample_bootstrapper:
        estimator_works_with_bootstrap_sample:
            just call `metric_estimator` if true,
            use `metric_estimator` with `np.apply_along_axis` otherwise
    """
    if sample_bootstrapper is None:
        sample_bootstrapper = bootstrap_samples
    metric_pilot = metric_estimator(groups.pilot)
    metric_control = metric_estimator(groups.control)
    boostrap_samples = sample_bootstrapper(boostrap_size, groups)
    if estimator_works_with_bootstrap_sample:
        sampled_metric_control = metric_estimator(boostrap_samples.control)
        sampled_metric_pilot = metric_estimator(boostrap_samples.pilot)
    else:
        sampled_metric_control = np.apply_along_axis(metric_estimator, axis=1, arr=boostrap_samples.control)
        sampled_metric_pilot = np.apply_along_axis(metric_estimator, axis=1, arr=boostrap_samples.pilot)
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
    experiment = dict(
        effect=Effect(0.1, is_additive=False),
        max_days=30,
        _sample_params=NormalDistributionWithConstantRateSampleParams(
            mean=100, std=10, n_users_per_day=10,
        ),
    )
    experiment['error_rates'] = calculate_error_rates(
        effect=experiment['effect'],
        sample_params=experiment['sample_params'],
        sample_generator=get_groups_for_normal_with_constant_users_per_day_rate,
        max_days=experiment['max_days'],
        verbose=True,
    )
    find_optimal_duration(experiment['error_rates'], 0.2)
