import typing as tp

import numpy as np

from duration_estimator import (
    Effect, ExperimentDurationEstimator, Groups, TSingleGroup,
)
from duration_estimator.experiment_conductors.bootstrap import ConductUsingBootstrap
from duration_estimator.sample_generators.normal_distribution import (
    NormalWithConstantRateSampleParams,
    get_groups_from_normal_with_constant_new_users_rate,
)


class BootstrapForMeans(ConductUsingBootstrap):
    @staticmethod
    def metric_estimator(sample: TSingleGroup, axis: int = 0) -> tp.Union[float, np.ndarray]:
        return np.mean(sample, axis=axis)

    @staticmethod
    def sample_bootstrapper(bootstrap_size: int, groups: Groups) -> Groups:
        return Groups(
            np.random.choice(groups.control, (groups.control.size, bootstrap_size)),
            np.random.choice(groups.pilot, (groups.pilot.size, bootstrap_size)),
        )


def test_normal_distribution_duration_is_correct():
    experiment_simulator = ExperimentDurationEstimator(
        effect=Effect(0.1, is_additive=False),
        sample_params=NormalWithConstantRateSampleParams(
            mean=100, std=10, n_users_per_day=10,
        ),
        sample_generator=get_groups_from_normal_with_constant_new_users_rate,
        experiment_conductor=BootstrapForMeans(),
        max_days=10,
    )
    experiment_simulator.fit()
    assert experiment_simulator.find_optimal_duration(0.2) == 2
