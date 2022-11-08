from datetime import date

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def df_daily_users():
    return pd.DataFrame(
        {
            'date_': [
                date(2021, 11, 1),
                date(2021, 11, 1),
                date(2021, 11, 2),
                date(2021, 11, 2),
            ],
            'platform': ['ANDROID', 'IOS', 'ANDROID', 'IOS'],
            'new_users': [100, 50, 35, 27],
            'unique_users_cumcount': [100, 50, 135, 77],
        }
    ).set_index('date_')


@pytest.fixture
def df_user_sessions():
    return pd.DataFrame(
        {
            'user_pseudo_id': np.arange(0, 1500),
            'platform': [
                *['ANDROID'] * 1000,
                * ['IOS'] * 500,
            ],
            'sessions': [
                *np.random.normal(7, 13, 1000),
                *np.random.normal(17, 20, 500),
            ],
            'conversions': [
                *np.random.normal(0.29, 2.2, 1000),
                *np.random.normal(0.28, 1.9, 500),
            ],
            'conversion_proba': [
                *np.random.normal(0.015, 0.08, 1000),
                *np.random.normal(0.01, 0.06, 500),
            ],
        }
    )
