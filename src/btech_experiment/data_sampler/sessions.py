import typing as tp

import numpy as np
import pandas as pd

from .historical_data_sampler import HistoricBasedSampleParams, StratifiedGroups, HistoricalDataSampler


class HistoricalUsersConversionsSampler(HistoricalDataSampler):
    @staticmethod
    def _sample(
            df_user_sessions: pd.DataFrame,
            n_unique_users_for_period: pd.Series,
            sample_params: HistoricBasedSampleParams,
    ) -> StratifiedGroups:
        sample_pilot = {}
        sample_control = {}
        sample_sizes = sample_params.get_sample_size(n_unique_users_for_period)
        for strata_name, strata_data in df_user_sessions.groupby('platform'):
            strata_sample_size = sample_sizes[strata_name]
            general_sample = (
                strata_data[['sessions', 'conversions']]
                .sample(strata_sample_size)
            )
            groups_sizes = sample_params.get_groups_sizes(strata_sample_size)
            sample_pilot[strata_name] = general_sample.head(groups_sizes.pilot)
            sample_control[strata_name] = general_sample.tail(groups_sizes.control)
        return StratifiedGroups(pilot=pd.Series(sample_pilot), control=pd.Series(sample_control))


def stratified_metric_estimator_for_sessions(
        group: tp.Mapping[str, tp.Union[np.ndarray, pd.DataFrame]],
        strats_weights: pd.Series,
) -> float:
    group_mean = {
        strata_name: (
            strata_data['conversions'].sum() / strata_data['sessions'].sum()
            if isinstance(strata_data, pd.DataFrame)
            else strata_data.sum(axis=0)[:, 0] / strata_data.sum(axis=0)[:, 1]
        )
        for strata_name, strata_data in group.items()
    }
    return sum(
        group_weight * group_mean[group_id]
        for group_id, group_weight in strats_weights.items()
    )


def bootstrap_strata_conversions(
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


def stratified_sample_bootstrapper_for_sessions(
        boostrap_size: int, groups: StratifiedGroups,
) -> StratifiedGroups:
    return StratifiedGroups(
        control={
            name: bootstrap_strata_conversions(data, boostrap_size)
            for name, data in groups.control.items()
        },
        pilot={
            name: bootstrap_strata_conversions(data, boostrap_size)
            for name, data in groups.pilot.items()
        },
    )
