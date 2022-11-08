import pandas as pd

from btech_experiment.data_sampler import (
    HistoricBasedSampleParams,
    eval_strats_weights,
    HistoricalUsersConversionsSampler,
    stratified_metric_estimator_for_users,
    stratified_sample_bootstrapper_for_users,
)
from duration_estimator.experiment_conductors.bootstrap import BootstrapForMeans


def test_sampling_sessions(
        df_daily_users: pd.DataFrame,
        df_user_sessions: pd.DataFrame,
) -> None:
    sampler = HistoricalUsersConversionsSampler(df_daily_users, df_user_sessions)
    group = sampler(n_days=1, sample_params=HistoricBasedSampleParams(0.15, 0.2))
    assert isinstance(
        stratified_metric_estimator_for_users(
            group.pilot, eval_strats_weights(df_daily_users),
        ),
        float,
    )
    assert (
            BootstrapForMeans().bootstrap_sample(5, group).pilot.shape == (2, 5)
    )

    # test on bootstrap
    strat_wt = eval_strats_weights(df_daily_users)
    strat_gr = stratified_sample_bootstrapper_for_users(100, group)
    boot_metric = stratified_metric_estimator_for_users(strat_gr.control, strat_wt)
    assert len(boot_metric) == 100
