import pandas as pd

from btech_experiment.data_sampler import (
    HistoricBasedSampleParams,
    HistoricalUsersConversionsSampler,
    SessionsBootstrap,
    eval_strats_weights,
)
from duration_estimator import Effect


def test_sampling_sessions(
        df_daily_users: pd.DataFrame,
        df_user_sessions: pd.DataFrame,
) -> None:
    # test sampler
    sampler = HistoricalUsersConversionsSampler(df_daily_users, df_user_sessions)
    groups = sampler(n_days=1, sample_params=HistoricBasedSampleParams(0.15, 0.2))
    assert 'ANDROID' in groups.pilot.keys()
    assert 'IOS' in groups.pilot.keys()
    assert groups.pilot['ANDROID'].shape == (2073,)
    assert groups.pilot['IOS'].shape == (418,)

    # test metric on initial sample
    st_wt = eval_strats_weights(df_daily_users)
    metrics = SessionsBootstrap.estimate_metric(groups, strats_weights=st_wt)
    assert isinstance(metrics.pilot, float)
    assert isinstance(metrics.control, float)

    # test bootstrapping
    bootstrapped_samples = SessionsBootstrap.bootstrap_sample(5, groups)
    assert 'ANDROID' in groups.pilot.keys()
    assert 'IOS' in groups.pilot.keys()
    assert (
            bootstrapped_samples.pilot['ANDROID'].shape
            == (bootstrapped_samples.pilot['ANDROID'].shape[0], 5)
    )
    assert (
            bootstrapped_samples.pilot['IOS'].shape
            == (bootstrapped_samples.pilot['IOS'].shape[0], 5)
    )

    # test metric for boot sample
    boot_metric = SessionsBootstrap.estimate_metric(
        bootstrapped_samples, strats_weights=st_wt,
    )
    assert len(boot_metric.pilot) == 5

    # try end-to_end
    found_effect = SessionsBootstrap(st_wt)(groups, Effect(0.5, is_additive=False))
    assert found_effect.given_effect
    assert not found_effect.given_no_effect
