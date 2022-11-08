from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from logging import getLogger
import typing as tp

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm.auto import trange

logger = getLogger(__file__)

Vector = tp.Union[tp.List[tp.Union[int, float]], np.ndarray, pd.Series]
T = tp.TypeVar('T')


@dataclass
class Groups(tp.Generic[T]):
    control: T
    pilot: T


class ExperimentConductor(tp.Protocol):
    def __call__(self, groups: Groups, effect: Effect) -> FoundEffect:
        ...


class SampleGenerator(tp.Protocol):
    def __call__(self, sample_size: int, sample_params: SampleParams) -> Groups:
        ...


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
class FoundEffect:
    given_effect: bool
    given_no_effect: bool

    def to_test_errors(self) -> TestErrors:
        return TestErrors(
            n_false_positive=self.given_no_effect,
            n_false_negative=not self.given_effect,
            n_experiments_per_each=1,
        )


class ExperimentDurationEstimator:
    def __init__(
            self,
            effect: Effect,
            experiment_conductor: ExperimentConductor,
            sample_generator: SampleGenerator,
            sample_params: SampleParams,
            max_days: int = 30,
    ) -> None:
        """
        todo

        Args:
            effect: effect size in percents, possible values > 0,
                let's say 0.1 is passed then metric has increased on 10%.
            experiment_conductor:
            sample_generator:
            sample_params: params passed to `self.sample_generator`
            max_days: max experiment duration
        """

        self._error_rates = None
        self.effect = effect
        self.sample_params = sample_params
        self.max_days = max_days

        self.experiment_conductor = experiment_conductor
        self.sample_generator = sample_generator

    @property
    def error_rates(self) -> tp.List[TestErrors]:
        if self._error_rates is None:
            raise ValueError("Please call `fit` method first")
        return self._error_rates

    def fit(
            self,
            n_experiment_runs_per_day_simulation: int = 250,
            verbose: bool = False,
    ) -> None:
        """Returns error rates based on duration from 1 to max_days

        Args:
            n_experiment_runs_per_day_simulation: n_observations per day we get to estimate error rates
            verbose:
        """
        self._error_rates = []
        days_range = trange(1, self.max_days + 1) if verbose else range(1, self.max_days + 1)
        for n_days in days_range:
            self._error_rates.append(
                self._measure_error_rate_for_given_duration(
                    n_days=n_days,
                    n_iterations=n_experiment_runs_per_day_simulation,
                    verbose=verbose
                )
            )
            if verbose:
                logger.info(f'Current error rates estimation: {self.error_rates}.')

    def _measure_error_rate_for_given_duration(
            self,
            n_days: int,
            n_iterations: int = 250,
            verbose: bool = False,
            n_jobs: int = -1,
    ) -> TestErrors:
        iterator = (
            trange(n_iterations, leave=False)
            if verbose
            else range(n_iterations)
        )
        test_errors = Parallel(n_jobs=n_jobs)(
            delayed(self._measure_one_error)(n_days)
            for _ in iterator
        )
        return sum(test_errors, TestErrors())

    def _measure_one_error(self, n_days: int) -> TestErrors:
        groups = self.sample_generator(n_days, self.sample_params)
        experiment_results = self.experiment_conductor(groups, self.effect)
        return experiment_results.to_test_errors()

    def find_optimal_duration(
            self, error_rate_threshold: float,
    ) -> int:
        """
        Returns number of days needed to be below the specified error threshold

        Args:
            error_rate_threshold: FP + FN error rates from interval [0.05, 0.5].
        """
        total_error_rates = np.array([x.total_rate for x in self.error_rates])
        less_than_threshold = total_error_rates <= error_rate_threshold
        if not less_than_threshold.any():
            return len(self.error_rates)
        return 1 + np.where(less_than_threshold)[0].min()
