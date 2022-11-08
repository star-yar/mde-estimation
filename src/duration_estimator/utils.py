import json
import typing as tp

from . import ExperimentDurationEstimator


def save_experiment_result(
    period: tp.Any,
    duration_estimator: ExperimentDurationEstimator,
    is_user_based_metric: bool,
) -> None:
    mode = 'user' if is_user_based_metric else 'sessions'
    filename = (
        f'./data/experiments/error_rates_{mode}_{period}_{duration_estimator}.json'
    )
    with open(filename, 'w') as f:
        json.dump(
            dict(
                error_rates=[
                    dict(
                        n_false_negative=x.n_false_negative,
                        n_false_positive=x.n_false_positive,
                        n_experiments_per_each=x.n_experiments_per_each,
                    )
                    for x in duration_estimator.error_rates
                ],
            ),
            f,
        )
