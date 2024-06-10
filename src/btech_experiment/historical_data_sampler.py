from abc import abstractmethod
from dataclasses import dataclass
from datetime import timedelta
import typing as tp

import numpy as np
import pandas as pd

from duration_estimator import Groups, SampleParams

NEW_USERS_COLUMN = 'new_users'
STRATA_COLUMN = 'platform'
T = tp.TypeVar('T')
StratifiedGroups = Groups[tp.Mapping[str, T]]


class GroupsSizes(Groups[int]):
    pass


@dataclass
class HistoricBasedSampleParams(SampleParams):
    share_of_all_users: float
    share_of_sample_for_pilot: float

    @property
    def share_of_sample_for_control(self) -> float:
        return 1 - self.share_of_sample_for_pilot

    def get_groups_sizes(
        self, experiment_sample_size: tp.Union[int, pd.Series],
    ) -> GroupsSizes:
        pilot_size = np.floor(
            self.share_of_sample_for_pilot
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


class HistoricalDataSampler(tp.Generic[T]):
    def __init__(
        self,
        df_daily_users: pd.DataFrame,
        df_user_sessions: pd.DataFrame,
        **sampler_kwargs: tp.Any,
    ) -> None:
        self.df_user_sessions = df_user_sessions
        self.df_daily_users = df_daily_users
        self.sampler_kwargs = sampler_kwargs

    @staticmethod
    @abstractmethod
    def _sample(
        df_user_sessions: pd.DataFrame,
        n_unique_users_for_period: pd.Series,
        sample_params: HistoricBasedSampleParams,
    ) -> StratifiedGroups:
        pass

    @staticmethod
    def _get_n_unique_users_for_period(
        df_daily_users: pd.DataFrame, n_days: int,
    ) -> pd.Series:
        starting_date = df_daily_users.index.min()
        is_selected_day = df_daily_users.index == starting_date + timedelta(n_days - 1)
        return df_daily_users[is_selected_day].set_index(STRATA_COLUMN)['unique_users_cumcount']

    def __call__(self, n_days: int, sample_params: HistoricBasedSampleParams) -> StratifiedGroups:
        n_unique_users_for_period = self._get_n_unique_users_for_period(
            self.df_daily_users, n_days,
        )
        return self._sample(
            self.df_user_sessions,
            n_unique_users_for_period,
            sample_params,
        )


def eval_strats_weights(df_daily_users: pd.DataFrame) -> pd.Series:
    return (
        df_daily_users.groupby(STRATA_COLUMN)[NEW_USERS_COLUMN].sum()
        / df_daily_users[NEW_USERS_COLUMN].sum()
    )
