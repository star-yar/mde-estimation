import json
from pathlib import Path
import typing as tp

from . import ExperimentDurationEstimator, TestErrors

EXPERIMENTS_PATH = Path(__file__).parents[2] / 'data'


def save_experiment_result(
    experiment_name: str,
    duration_estimator: ExperimentDurationEstimator,
    experiments_dir: str,
) -> None:
    filename = f'{experiments_dir}/{experiment_name}.json'
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


def load_experiment_result(
    experiment_name: str, experiments_dir: str,
) -> tp.List[TestErrors]:
    with open(f'{experiments_dir}/{experiment_name}.json', 'r') as f:
        results = json.load(f)

    return [
        TestErrors(
            n_false_negative=record['n_false_negative'],
            n_false_positive=record['n_false_positive'],
            n_experiments_per_each=record['n_experiments_per_each'],
        )
        for record in results['error_rates']
    ]
