from dataclasses import asdict
import json
import typing as tp

from .duration_estimator import Effect, SampleParams, TestErrors


def save_experiment_result(
    period: tp.Any,
    expected_effect: Effect,
    sample_params: SampleParams,
    error_rates: tp.List[TestErrors],
    is_user_based_metric: bool,
) -> None:
    mode = 'user' if is_user_based_metric else 'sessions'
    with open(
        f'./data/experiments/error_rates_'
        f'{mode}_'
        f'{period}_'
        f'{asdict(expected_effect)}_'
        f'{asdict(sample_params)}.json',
        'w',
    ) as f:
        json.dump(
            dict(
                error_rates=[
                    dict(
                        n_false_negative=x.n_false_negative,
                        n_false_positive=x.n_false_positive,
                        n_experiments_per_each=x.n_experiments_per_each,
                    )
                    for x in error_rates
                ],
            ),
            f,
        )
