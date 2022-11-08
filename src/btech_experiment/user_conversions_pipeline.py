from btech_experiment import (
    HistoricBasedSampleParams,
    eval_strats_weights,
)
from btech_experiment.data_retrieval import (
    get_daily_users,
    get_period,
    get_user_sessions,
)
from btech_experiment.diagnostics import show_diagnostics
from btech_experiment.estimators import (
    HistoricalUsersConversionsSampler,
    UsersConversionsBootstrap,
)
from duration_estimator import (
    Effect,
    ExperimentDurationEstimator,
)
from duration_estimator.utils import save_experiment_result

if __name__ == '__main__':
    VERBOSE = True
    period = get_period(
        # last_available_period_date='2021-12-01', n_month_from_last_date=1,
        last_available_period_date='2022-10-01', n_month_from_last_date=1,
    )
    print(f'{period = }')

    # load data
    df_daily_users = get_daily_users(*period)
    df_user_sessions = get_user_sessions(*period)

    # experiment setup
    expected_effect = Effect(0.05, is_additive=False)
    sample_params = HistoricBasedSampleParams(
        share_of_all_users=0.5,
        share_of_sample_for_pilot=0.9,
    )
    max_days = 30
    print(f'{sample_params = }')

    # diagnostics
    show_diagnostics(df_daily_users, sample_params)

    # components set up
    sample_generator = HistoricalUsersConversionsSampler(
        df_daily_users=df_daily_users,
        df_user_sessions=df_user_sessions,
    )
    experiment_conductor = UsersConversionsBootstrap(
        strats_weights=eval_strats_weights(df_daily_users)
    )

    # duration estimator
    duration_estimator = ExperimentDurationEstimator(
        effect=expected_effect,
        sample_generator=sample_generator,
        experiment_conductor=experiment_conductor,
        sample_params=sample_params,
        max_days=max_days,
    ).fit(VERBOSE)

    save_experiment_result(period, duration_estimator, is_user_based_metric=True)