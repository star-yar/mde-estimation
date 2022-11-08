from dataclasses import dataclass
import typing as tp

import numpy as np
import pandas as pd

from btech_experiment.data_retrieval import _get_n_unique_users_for_period
from duration_estimator import Groups, SampleParams, Vector


class GroupsSizes(Groups[int]):
    pass


class StratifiedGroups(Groups[tp.Mapping[str, Vector]]):
    pass


@dataclass
class HistoricBasedSampleParams(SampleParams):
    share_of_all_users: float
    share_of_samlpe_for_pilot: float

    @property
    def share_of_samlpe_for_control(self) -> float:
        return 1 - self.share_of_samlpe_for_pilot

    def get_groups_sizes(
            self, experiment_sample_size: tp.Union[int, pd.Series],
    ) -> GroupsSizes:
        pilot_size = np.floor(
            self.share_of_samlpe_for_pilot
            * experiment_sample_size
        )
        pilot_size = (
            pilot_size.astype(int)
            if isinstance(pilot_size, pd.Series)
            else int(pilot_size)
        )
        return GroupsSizes(
            pilot=pilot_size,
            control=experiment_sample_size - pilot_size,
        )

    def get_sample_size(
            self, whole_sample_size: tp.Union[int, pd.Series],
    ) -> pd.Series:
        size = np.floor(
            self.share_of_all_users * whole_sample_size
        )
        return (
            size.astype(int)
            if isinstance(size, pd.Series)
            else int(size)
        )

    def get_groups_sizes_from_general_sample_size(
            self, whole_sample_size: tp.Union[int, pd.Series],
    ) -> GroupsSizes:
        experiment_sample_size = self.get_sample_size(whole_sample_size)
        return self.get_groups_sizes(experiment_sample_size)


class Sampler(tp.Protocol):
    def __call__(
            self,
            df_user_sessions: pd.DataFrame,
            n_unique_users_for_period: pd.Series,
            sample_params: HistoricBasedSampleParams,
            **kwargs: tp.Any,
    ) -> StratifiedGroups:
        ...


def sample_from_historical_data(
        n_days: int,
        sample_params: HistoricBasedSampleParams,
        df_daily_users: pd.DataFrame,
        df_user_sessions: pd.DataFrame,
        sampler: Sampler,
        **sampler_kwargs: tp.Any,
) -> StratifiedGroups:
    n_unique_users_for_period = _get_n_unique_users_for_period(
        df_daily_users, n_days,
    )
    return sampler(
        df_user_sessions,
        n_unique_users_for_period,
        sample_params,
        **sampler_kwargs,
    )


def eval_strats_weights(df_daily_users: pd.DataFrame) -> pd.Series:
    return (
            df_daily_users.groupby('platform')['new_users'].sum()
            / df_daily_users['new_users'].sum()
    )
