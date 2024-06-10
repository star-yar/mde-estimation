import numpy as np

from duration_estimator import sample_generators


def test_normal_with_constant_rate():
    new_per_day = 1000
    mean = 0
    std = 1
    generator = sample_generators.NormalDistributionWithConstantDailyGrowth(mean, std, new_per_day)
    for days in range(1, 10):
        sample = generator(days)
        assert sample.pilot.size == sample.control.size == days * new_per_day
        assert round(sample.pilot.mean(), 0) == round(sample.control.mean(), 0) == mean
        assert round(sample.pilot.std(), 1) == round(sample.control.std(), 1) == std
