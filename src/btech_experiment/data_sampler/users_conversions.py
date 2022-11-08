import numpy as np
import pandas as pd

from .historical_data_sampler import HistoricBasedSampleParams, StratifiedGroups


def sample_user_conversions_from_user_sessions(
    df_user_sessions: pd.DataFrame,
    n_unique_users_for_period: pd.Series,
    sample_params: HistoricBasedSampleParams,
) -> StratifiedGroups:
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


def stratified_sample_bootstrapper_for_users(
    boostrap_size: int, groups: StratifiedGroups,
) -> StratifiedGroups:
    return StratifiedGroups(
        groups.control.apply(
            lambda x: np.random.choice(x, (x.size, boostrap_size))
        ),
        groups.pilot.apply(
            lambda x: np.random.choice(x, (x.size, boostrap_size))
        ),
    )


def stratified_metric_estimator_for_users(
    group: pd.Series, strats_weights: pd.Series,
) -> float:
    group_mean = group.apply(np.mean, axis=0)
    return sum(
        group_weight * group_mean[group_id]
        for group_id, group_weight in strats_weights.items()
    )
