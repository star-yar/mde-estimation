import typing as tp

import numpy as np
import pandas as pd

from duration_estimator.experiment_conductors import Bootstrap
from ..historical_data_sampler import (
    HistoricBasedSampleParams,
    STRATA_COLUMN,
    StratifiedGroups,
    HistoricalDataSampler,
)

_TSample = pd.DataFrame
StratifiedSessions = StratifiedGroups[_TSample]


class HistoricalSessionsSampler(HistoricalDataSampler[_TSample]):
    @staticmethod
    def _sample(
            df_user_sessions: pd.DataFrame,
            n_unique_users_for_period: pd.Series,
            sample_params: HistoricBasedSampleParams,
    ) -> StratifiedSessions:
        sample_pilot = {}
        sample_control = {}
        sample_sizes = sample_params.get_sample_size(n_unique_users_for_period)
        for strata_name, strata_data in df_user_sessions.groupby(STRATA_COLUMN):
            strata_sample_size = sample_sizes[strata_name]
            general_sample = (
                strata_data[['sessions', 'conversions']]
                .sample(strata_sample_size)
            )
            groups_sizes = sample_params.get_groups_sizes(strata_sample_size)
            sample_pilot[strata_name] = general_sample.head(groups_sizes.pilot)
            sample_control[strata_name] = general_sample.tail(groups_sizes.control)
        return StratifiedGroups(pilot=pd.Series(sample_pilot), control=pd.Series(sample_control))


class SessionsBootstrap(Bootstrap[_TSample]):
    def __init__(
            self,
            strats_weights: tp.Mapping[str, float],
            **kwargs: tp.Any,
    ) -> None:
        super().__init__(**kwargs)
        self._metric_kwargs["strats_weights"] = strats_weights

    @staticmethod
    def estimate_metric(
            groups: StratifiedSessions,
            sample_size_axis: int = 0,
            strats_weights: tp.Mapping[str, float] = None,
    ) -> StratifiedSessions:
        if strats_weights is None:
            raise ValueError('Please provide `strats_weights`')
        return StratifiedSessions(
            pilot=_estimate_metric(groups.pilot, strats_weights, sample_size_axis),
            control=_estimate_metric(groups.control, strats_weights, sample_size_axis),
        )

    @staticmethod
    def bootstrap_sample(
            bootstrap_size: int,
            groups: StratifiedSessions,
            **kwargs: tp.Any,
    ) -> StratifiedSessions:
        return StratifiedGroups(
            control={
                name: _bootstrap_strata_conversions(data, bootstrap_size)
                for name, data in groups.control.items()
            },
            pilot={
                name: _bootstrap_strata_conversions(data, bootstrap_size)
                for name, data in groups.pilot.items()
            },
        )


def _estimate_metric(
        group: tp.Mapping[str, _TSample],
        strats_weights: tp.Mapping[str, float],
        sample_size_axis: int,
) -> float:
    group_mean = {
        strata_name: (
            _calc_metric_for_initial_sample(strata_data)
            if isinstance(strata_data, pd.DataFrame)
            else _calc_metric_for_boot_sample(strata_data, sample_size_axis)
        )
        for strata_name, strata_data in group.items()
    }
    return sum(
        group_weight * group_mean[group_id]
        for group_id, group_weight in strats_weights.items()
    )


def _calc_metric_for_initial_sample(strata_data: pd.DataFrame) -> float:
    return strata_data['conversions'].sum() / strata_data['sessions'].sum()


def _calc_metric_for_boot_sample(
        strata_data: pd.DataFrame, sample_size_axis: int,
) -> np.ndarray:
    return (
            strata_data.sum(axis=sample_size_axis)[:, 0]
            / strata_data.sum(axis=sample_size_axis)[:, 1]
    )


def _bootstrap_strata_conversions(
        strat_data: pd.DataFrame, bootstrap_size: int,
) -> np.array:
    sample_size = strat_data.shape[0]
    sampled_strat = strat_data.sample(
        bootstrap_size * sample_size, replace=True,
    )
    assert len(strat_data.columns) == 2
    return (
        sampled_strat[['conversions', 'sessions']]
        .values.reshape(sample_size, bootstrap_size, -1)
    )
