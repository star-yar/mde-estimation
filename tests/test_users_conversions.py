import pandas as pd
import pytest

from btech_experiment.data_sampler import (
    HistoricBasedSampleParams,
    HistoricalUsersConversionsSampler,
    SessionsBootstrap,
    eval_strats_weights,
)
from duration_estimator import Effect


class TestUserSessionsCase:
    @pytest.fixture
    def strats_weights(self, df_daily_users: pd.DataFrame) -> pd.Series:
        return eval_strats_weights(df_daily_users)

    @pytest.fixture
    def sampler(
            self,
            df_daily_users: pd.DataFrame,
            df_user_sessions: pd.DataFrame,
    ) -> HistoricalUsersConversionsSampler:
        return HistoricalUsersConversionsSampler(df_daily_users, df_user_sessions)

    def test_sampling(
            self, sampler: HistoricalUsersConversionsSampler,
    ) -> None:
        groups = sampler(n_days=1, sample_params=HistoricBasedSampleParams(0.15, 0.2))
        assert 'ANDROID' in groups.pilot.keys()
        assert 'IOS' in groups.pilot.keys()
        assert groups.pilot['ANDROID'].shape == (2073,)
        assert groups.pilot['IOS'].shape == (418,)

    def test_evaluating_metric_on_initial_sample(
            self,
            sampler: HistoricalUsersConversionsSampler,
            strats_weights: pd.Series,
    ) -> None:
        groups = sampler(n_days=1, sample_params=HistoricBasedSampleParams(0.15, 0.2))

        # test metric on initial sample
        metrics = SessionsBootstrap.estimate_metric(groups, strats_weights=strats_weights)
        assert isinstance(metrics.pilot, float)
        assert isinstance(metrics.control, float)

    def test_bootstrapping_sample(
            self,
            sampler: HistoricalUsersConversionsSampler,
    ) -> None:
        groups = sampler(n_days=1, sample_params=HistoricBasedSampleParams(0.15, 0.2))

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

    def test_evaluating_metric_on_bootstrapped_sample(
            self,
            sampler: HistoricalUsersConversionsSampler,
            strats_weights: pd.Series,
    ) -> None:
        groups = sampler(n_days=1, sample_params=HistoricBasedSampleParams(0.15, 0.2))
        bootstrapped_samples = SessionsBootstrap.bootstrap_sample(5, groups)

        boot_metric = SessionsBootstrap.estimate_metric(
            bootstrapped_samples, strats_weights=strats_weights,
        )
        assert len(boot_metric.pilot) == 5

    def test_end_to_end(
            self,
            sampler: HistoricalUsersConversionsSampler,
            strats_weights: pd.Series,
    ) -> None:
        groups = sampler(n_days=1, sample_params=HistoricBasedSampleParams(0.15, 0.2))

        found_effect = SessionsBootstrap(strats_weights)(groups, Effect(0.5, is_additive=False))
        assert found_effect.given_effect
        assert not found_effect.given_no_effect
