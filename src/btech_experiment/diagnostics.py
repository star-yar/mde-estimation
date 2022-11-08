import numpy as np
import pandas as pd
from plotly import express as px, graph_objects as go
from plotly.subplots import make_subplots

from btech_experiment.custom import HistoricBasedSampleParams

PLATFORM = 'platform'
CONTROL = 'control'
PILOT = 'pilot'
COLORS = px.colors.qualitative.T10


def show_diagnostics(
        df_daily_users: pd.DataFrame,
        sample_params: HistoricBasedSampleParams,
) -> None:
    experiment_data = evaluate_experiment_data(df_daily_users, sample_params)

    # plotting diagnostics
    fig = make_subplots(
        rows=2, cols=2,
        shared_yaxes=True,
        subplot_titles=(
            'Pilot Total users', 'Pilot New users',
            'Control Total users', 'Control New users',
        ),
    )

    day_index = np.arange(experiment_data['date_'].nunique()) + 1
    for (platform_id, platform_data), color in zip(experiment_data.groupby(PLATFORM), COLORS):
        show_legend = True
        for group_index, group_name in enumerate((PILOT, CONTROL), start=1):
            fig.add_trace(
                go.Scatter(
                    x=day_index,
                    y=platform_data[
                        f'unique_users_cumcount_{group_name}'
                    ],
                    name=platform_id,
                    line=dict(color=color),
                    legendgroup=platform_id,
                    showlegend=show_legend,
                ),
                row=1, col=group_index,
            )
            show_legend = False
            fig.add_trace(
                go.Scatter(
                    x=day_index,
                    y=platform_data[
                        f'new_users_{group_name}'
                    ],
                    name=platform_id,
                    line=dict(color=color),
                    legendgroup=platform_id,
                    showlegend=show_legend,
                ),
                row=2, col=group_index,
            )

    fig.update_layout(
        height=800,
        title_text='Users daily during experiment',
    )
    fig.show()


def evaluate_experiment_data(
        df_daily_users: pd.DataFrame, sample_params: HistoricBasedSampleParams,
) -> pd.DataFrame:
    df_daily_users = df_daily_users.copy()
    get_size = lambda size_column, group_name: getattr(
        sample_params.get_groups_sizes_from_general_sample_size(
            df_daily_users[size_column]
        ),
        group_name,
    )
    for group_name in (PILOT, CONTROL):
        df_daily_users[f'unique_users_cumcount_{group_name}'] = (
            get_size('unique_users_cumcount', group_name)
        )
        df_daily_users[f'new_users_{group_name}'] = (
            get_size('new_users', group_name)
        )
    df_daily_user_conversions_total = (
        df_daily_users
        .groupby('date_')
        .sum()
        .reset_index()
        .assign(platform='total')
    )
    df_daily_user_conversions_with_total = (
        df_daily_users.reset_index()
        .append(df_daily_user_conversions_total)
    )
    return df_daily_user_conversions_with_total
