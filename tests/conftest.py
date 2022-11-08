# todo: fix
import pandas as pd
import pytest


@pytest.fixture
def df_daily_users():
    return pd.read_pickle('../data/df_daily_users.pkl')


@pytest.fixture
def df_user_sessions():
    return pd.read_pickle('../data/df_user_sessions.pkl')
