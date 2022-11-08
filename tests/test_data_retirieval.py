from btech_experiment.data_retrieval import (
    get_data_from_query,
    DATA_SOURCE_RAW_SESSIONS,
)


def test_data_loads():
    df = get_data_from_query(
        f'select * from {DATA_SOURCE_RAW_SESSIONS} limit 10'
    )
    assert not df.empty
