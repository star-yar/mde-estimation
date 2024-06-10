import numpy as np

from duration_estimator import sample_generators


def test_normal_with_constant_rate():
    new_per_day = 1000
    generator = sample_generators.NormalDistributionWithConstantDailyGrowth(0, 1, new_per_day)
    for days in range(1, 10):
        sample = generator(days)
        assert sample.pilot.size == sample.control.size == days * new_per_day
        np.testing.assert_approx_equal(sample.pilot.mean(), 0)
        np.testing.assert_approx_equal(sample.pilot.std(), 1)
        np.testing.assert_approx_equal(sample.pilot.mean(), sample.control.mean())
        np.testing.assert_approx_equal(sample.pilot.std(), sample.control.std())
