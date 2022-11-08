from datetime import date
import typing as tp

from google.auth.credentials import Credentials
from google.oauth2 import service_account
import pandas as pd
import pandas_gbq

TDate = tp.Union[str, date]

DATA_SOURCE_RAW_SESSIONS = '`btech-dwh.ga_orbis_generaloverview.Sessions_raw_data`'


def load_credentials(path: str) -> Credentials:
    return (
        service_account
        .Credentials
        .from_service_account_file(path)
    )


def get_data_from_query(
        query: str, credentials: Credentials,
) -> pd.DataFrame:
    return pandas_gbq.read_gbq(query, credentials=credentials)


def get_daily_users(from_: TDate, to: TDate, credentials: Credentials) -> pd.DataFrame:
    df_daily_user_conversions = get_data_from_query(
        f"""
        with first_occurance as (
            select user_pseudo_id, min(date_) date_
            from {DATA_SOURCE_RAW_SESSIONS}
            where date_ between '{from_}' and '{to}'
            group by user_pseudo_id
        ),
        first_conversion as (
            select user_pseudo_id, min(date_) date_
            from {DATA_SOURCE_RAW_SESSIONS}
            where date_ between '{from_}' and '{to}' and Purchase_flag='True'
            group by user_pseudo_id
        ),
        additional_user_counts as (
            select
                date_,
                platform,
                -- region, city,
                count(distinct user_pseudo_id) users,
            from {DATA_SOURCE_RAW_SESSIONS}
            inner join first_occurance using(user_pseudo_id, date_)
            where date_ between '{from_}' and '{to}'
            group by
                date_, platform
            order by date_ asc
        ),
        additional_user_conversion as (
            select
                date_,
                platform,
                -- region, city,
                sum(cast(cast(Purchase_flag as bool) as int)) converted
            from {DATA_SOURCE_RAW_SESSIONS}
            inner join first_conversion using(user_pseudo_id, date_)
            where date_ between '{from_}' and '{to}'
            group by
                date_, platform
            order by date_ asc
        )
        select
            date_, platform, users as new_users,
            sum(users) over (partition by platform order by date_) as unique_users_cumcount
        from additional_user_counts
        left join additional_user_conversion using(date_, platform)
        order by date_, platform
        """,
        credentials,
    )
    df_daily_user_conversions['date_'] = pd.to_datetime(
        df_daily_user_conversions['date_']
    )
    df_daily_user_conversions = df_daily_user_conversions.set_index('date_')
    return df_daily_user_conversions


def get_user_sessions(from_: TDate, to: TDate, credentials: Credentials) -> pd.DataFrame:
    return get_data_from_query(
        f"""
        with per_user_sessions as (
          select
            user_pseudo_id,
            platform,
            count(session_id) sessions,
            sum(cast(cast(Purchase_flag as bool) as int)) conversions,
            (
                sum(cast(cast(Purchase_flag as bool) as int))
                / count(session_id)
            ) conversion_proba
          from {DATA_SOURCE_RAW_SESSIONS}
          where date_ between '{from_}' and '{to}'
          group by user_pseudo_id, platform
        )
        select * 
        from per_user_sessions
        """,
        credentials,
    )


def get_period(
        last_available_period_date: str,
        n_month_from_last_date: int,
        date_fmt: str = '%Y-%m-%d',
) -> tp.Tuple[str, str]:
    to = pd.to_datetime(last_available_period_date, format=date_fmt)
    from_ = date(to.year, to.month - n_month_from_last_date, to.day)
    from_ = from_.strftime(date_fmt)
    to = to.strftime(date_fmt)
    return from_, to
