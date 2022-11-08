import pandas as pd

from btech_experiment.data_sampler import (
    HistoricBasedSampleParams,
    eval_strats_weights,
    HistoricalDataSampler,
    bootstrap_strata_conversions,
    sample_sessions_from_user_sessions,
    stratified_metric_estimator_for_sessions,
    stratified_sample_bootstrapper_for_sessions,
)


def test_sampling_user_conversions(
        df_daily_users: pd.DataFrame,
        df_user_sessions: pd.DataFrame,
) -> None:
    sampler = HistoricalDataSampler(
        sample_sessions_from_user_sessions,
        df_daily_users,
        df_user_sessions,
    )
    group = sampler(n_days=1, sample_params=HistoricBasedSampleParams(0.1, 0.2))
    assert isinstance(
        stratified_metric_estimator_for_sessions(
            group.pilot, eval_strats_weights(df_daily_users),
        ),
        float,
    )
    assert (
            bootstrap_strata_conversions(group.pilot['ANDROID'], 5).shape
            == (group.pilot['ANDROID'].shape[0], 5, 2)
    )

    # test on bootstrap
    strat_wt = eval_strats_weights(df_daily_users)
    strat_gr = stratified_sample_bootstrapper_for_sessions(100, group)
    boot_metric = stratified_metric_estimator_for_sessions(strat_gr.pilot, strat_wt)
    assert len(boot_metric) == 100
