import numpy as np

from duration_estimator import (
    Effect, ExperimentDurationEstimator,
)
from duration_estimator.experiment_conductors import BootstrapForMeans
from duration_estimator.sample_generators.normal_distribution import (
    NormalDistributionWithConstantDailyGrowth,
)


def test_normal_distribution_duration_is_correct() -> None:
    experiment_simulator = ExperimentDurationEstimator(
        expected_effect=Effect(0.1, is_additive=False),
        sample_generator=NormalDistributionWithConstantDailyGrowth(
            mean=100,
            std=18,
            new_observations_per_day=10,
            random_gen=np.random.RandomState(42),
        ),
        experiment_conductor=BootstrapForMeans(),
        max_days=10,
    )
    experiment_simulator.fit()
    assert experiment_simulator.find_optimal_duration(0.2) == 3
