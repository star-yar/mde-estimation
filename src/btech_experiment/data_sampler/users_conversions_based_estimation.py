import typing as tp

import numpy as np
import pandas as pd

from duration_estimator.experiment_conductors import Bootstrap
from .historical_data_sampler import HistoricBasedSampleParams, HistoricalDataSampler, StratifiedGroups

_TSample = np.ndarray
StratifiedUserConversions = StratifiedGroups[_TSample]


class HistoricalUsersConversionsSampler(HistoricalDataSampler[_TSample]):
    @staticmethod
    def _sample(
            df_user_sessions: pd.DataFrame,
            n_unique_users_for_period: pd.Series,
            sample_params: HistoricBasedSampleParams,
    ) -> StratifiedUserConversions:
        sample_pilot = {}
        sample_control = {}
        sample_sizes = sample_params.get_sample_size(
            n_unique_users_for_period
        )
        for strata_name, strata_data in df_user_sessions.groupby('platform'):
            strata_sample_size = sample_sizes[strata_name]
            general_sample = (
                strata_data['conversion_proba']
                .sample(strata_sample_size)
            )
            groups_sizes = sample_params.get_groups_sizes(strata_sample_size)
            sample_pilot[strata_name] = general_sample.head(
                groups_sizes.pilot
            ).values
            sample_control[strata_name] = general_sample.tail(
                groups_sizes.control
            ).values
        return StratifiedGroups(
            pilot=pd.Series(sample_pilot),
            control=pd.Series(sample_control),
        )


class UsersConversionsBootstrap(Bootstrap[_TSample]):
    def __init__(
            self,
            strats_weights: tp.Mapping[str, float],
            **kwargs: tp.Any,
    ) -> None:
        super().__init__(**kwargs)
        self._metric_kwargs["strats_weights"] = strats_weights

    @staticmethod
    def estimate_metric(
            groups: StratifiedUserConversions,
            sample_size_axis: int = 0,
            strats_weights: tp.Mapping[str, float] = None,
    ) -> StratifiedUserConversions:
        if strats_weights is None:
            raise ValueError('Please provide `strats_weights`')
        return StratifiedUserConversions(
            pilot=_estimate_metric(groups.pilot, strats_weights, sample_size_axis),
            control=_estimate_metric(groups.control, strats_weights, sample_size_axis),
        )

    @staticmethod
    def bootstrap_sample(
            bootstrap_size: int,
            groups: StratifiedUserConversions,
            **kwargs: tp.Any,
    ) -> StratifiedUserConversions:
        return StratifiedUserConversions(
            groups.control.apply(
                lambda x: np.random.choice(x, (x.size, bootstrap_size))
            ),
            groups.pilot.apply(
                lambda x: np.random.choice(x, (x.size, bootstrap_size))
            ),
        )


def _estimate_metric(
        group: tp.Mapping[str, _TSample],
        strats_weights: tp.Mapping[str, float],
        sample_size_axis: int,
) -> float:
    group_mean = {
        strata: np.mean(data, axis=sample_size_axis)
        for strata, data in group.items()
    }
    return sum(
        group_weight * group_mean[group_id]
        for group_id, group_weight in strats_weights.items()
    )
