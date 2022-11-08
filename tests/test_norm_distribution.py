import numpy as np

from duration_estimator import (
    Effect, ExperimentDurationEstimator,
)
from duration_estimator.experiment_conductors import BootstrapForMeans
from duration_estimator.sample_generators.normal_distribution import (
    NormalWithConstantRateSampleParams,
    get_groups_from_normal_with_constant_new_users_rate,
)


def test_normal_distribution_duration_is_correct() -> None:
    experiment_simulator = ExperimentDurationEstimator(
        effect=Effect(0.1, is_additive=False),
        sample_params=NormalWithConstantRateSampleParams(
            mean=100, std=18, n_users_per_day=10, random_gen=np.random.RandomState(42),
        ),
        sample_generator=get_groups_from_normal_with_constant_new_users_rate,
        experiment_conductor=BootstrapForMeans(),
        max_days=10,
    )
    experiment_simulator.fit()
    assert experiment_simulator.find_optimal_duration(0.2) == 4
