import pandas as pd
import pytest

from btech_experiment.data_sampler import (
    HistoricBasedSampleParams,
    HistoricalSessionsSampler,
    SessionsBootstrap,
    StratifiedSessions,
    eval_strats_weights,
)
from duration_estimator import Effect


class TestSessionsCase:
    @pytest.fixture
    def sample_params(self) -> HistoricBasedSampleParams:
        return HistoricBasedSampleParams(0.5, 0.2)

    @pytest.fixture
    def strats_weights(self, df_daily_users: pd.DataFrame) -> pd.Series:
        return eval_strats_weights(df_daily_users)

    @pytest.fixture
    def groups(
            self,
            sampler: HistoricalSessionsSampler,
            sample_params: HistoricBasedSampleParams,
    ) -> StratifiedSessions:
        return sampler(n_days=1, sample_params=sample_params)

    @pytest.fixture
    def sampler(
            self,
            df_daily_users: pd.DataFrame,
            df_user_sessions: pd.DataFrame,
    ) -> HistoricalSessionsSampler:
        return HistoricalSessionsSampler(df_daily_users, df_user_sessions)

    def test_sampling(self, groups: StratifiedSessions) -> None:
        assert 'ANDROID' in groups.pilot.keys()
        assert 'IOS' in groups.pilot.keys()
        assert groups.pilot['ANDROID'].shape == (10, 2)
        assert groups.pilot['IOS'].shape == (5, 2)

    def test_evaluating_metric_on_initial_sample(
            self,
            strats_weights: pd.Series,
            groups: StratifiedSessions,
    ) -> None:
        metrics = SessionsBootstrap.estimate_metric(groups, strats_weights=strats_weights)
        assert isinstance(metrics.pilot, float)
        assert isinstance(metrics.control, float)

    def test_bootstrapping_sample(self, groups: StratifiedSessions) -> None:
        bootstrapped_samples = SessionsBootstrap.bootstrap_sample(5, groups)
        assert 'ANDROID' in groups.pilot.keys()
        assert 'IOS' in groups.pilot.keys()
        assert (
                bootstrapped_samples.pilot['ANDROID'].shape
                == (bootstrapped_samples.pilot['ANDROID'].shape[0], 5, 2)
        )
        assert (
                bootstrapped_samples.pilot['IOS'].shape
                == (bootstrapped_samples.pilot['IOS'].shape[0], 5, 2)
        )

    def test_evaluating_metric_on_bootstrapped_sample(
            self,
            strats_weights: pd.Series,
            groups: StratifiedSessions,
    ) -> None:
        bootstrapped_samples = SessionsBootstrap.bootstrap_sample(5, groups)
        boot_metric = SessionsBootstrap.estimate_metric(
            bootstrapped_samples, strats_weights=strats_weights,
        )
        assert len(boot_metric.pilot) == 5

    def test_end_to_end(
            self,
            strats_weights: pd.Series,
            groups: StratifiedSessions,
    ) -> None:
        boot = SessionsBootstrap(strats_weights)
        found_effect = boot(groups, Effect(1000.0, is_additive=True))
        assert found_effect.given_effect
        assert not found_effect.given_no_effect
