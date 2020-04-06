import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config
import opt_helpers as opt

SUSCEPTIBLE_COLOR = '#0099d1'
INFECTED_COLOR = '#fa7a2a'
HOSPITALIZED_COLOR = '#bab400'
RECOVERED_COLOR = '#009e45'
DEAD_COLOR = '#9e00ba'
LOCKDOWN_COLOR = '#dbdbdb'

LOCKDOWN_TH = 0.01
DEFAULT_THEME = 'simple_white'


def generate_data_from_scenario():
    scenario_path = os.path.join(config.PATH, 'app_data', 'scenario', 'two_lockdowns.csv')
    data_set = pd.read_csv(scenario_path)
    return data_set


def generate_data_from_model(hospital_beds, icus_penalty, lockdown_penalty):
    print(f"You are calling the model with {hospital_beds} beds, {icus_penalty} "
          f"icu penalty and {lockdown_penalty} lockdown penalty")
    time.sleep(2)
    lockdown_penalty /= 100
    icus_penalty /= 100
    hospital_beds /= 1000
    data_set, lockdown, params = opt.opt_strategy(weight_eps=icus_penalty, bed_ratio=hospital_beds,
                                                  weight_goout=lockdown_penalty)

    data = pd.DataFrame(data_set, index=['S', 'E', 'A', 'I', 'H', 'D', 'R']).transpose()
    data['beds'] = hospital_beds
    data['lockdown'] = np.append(0, (-1) / 12.3 * lockdown + 13.3 / 12.3)  # np.append((lockdown-1)/(params['k']-1),0)
    data['day'] = np.arange(len(lockdown) + 1)

    return data


def get_sir_plot(data_set: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data_set.day, y=(data_set.S) * 100,
        mode='lines',
        line=dict(width=0.5, color=SUSCEPTIBLE_COLOR),
        stackgroup='one',
        name='susceptible'
    ))
    fig.add_trace(go.Scatter(
        x=data_set.day, y=(data_set.E + data_set.I + data_set.A + data_set.H) * 100,
        mode='lines',
        line=dict(width=0.5, color=INFECTED_COLOR),
        stackgroup='one',
        name='infected'
    ))
    fig.add_trace(go.Scatter(
        x=data_set.day, y=(data_set.R + data_set.D) * 100,
        hoverinfo='x+y',
        mode='lines',
        line=dict(width=0.5, color=RECOVERED_COLOR),
        stackgroup='one',
        name='recovered'
    ))
    fig.update_layout(
        title="Infection curve",
        xaxis_title="days",
        yaxis_title="% of population",
        showlegend=True,
        template=DEFAULT_THEME,
        yaxis=dict(
            type='linear',
            range=[0, 100],
            ticksuffix='%'))
    return fig


def get_hospital_plot(data_set: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data_set.day,
        y=data_set.H * 100,
        mode='lines',
        name='hospitalized',
        line={'color': HOSPITALIZED_COLOR}))
    fig.add_trace(go.Scatter(
        x=data_set.day,
        y=data_set.D * 100,
        mode='lines',
        name='dead',
        line={'color': DEAD_COLOR}))
    fig.add_trace(go.Scatter(
        x=data_set.day,
        y=data_set.beds * 100,
        mode='lines',
        name='hospital beds',
        line={'dash': 'dot', 'color': '#000000'}))
    fig.update_layout(
        title="Hospitalized and dead people",
        xaxis_title="days",
        yaxis_title="% of population",
        showlegend=True,
        template=DEFAULT_THEME,
        yaxis=dict(
            type='linear',
            range=[0, max(data_set.H.max(), data_set.D.max()) * 110],
            ticksuffix='%'))

    data_set['lockdown_prev'] = data_set.lockdown.shift(1)
    data_set['lockdown_start'] = (data_set.lockdown > LOCKDOWN_TH) & (data_set.lockdown_prev < LOCKDOWN_TH)
    data_set['lockdown_end'] = (data_set.lockdown < LOCKDOWN_TH) & (data_set.lockdown_prev > LOCKDOWN_TH)

    starts = list(data_set[data_set.lockdown_start == 1].day)
    ends = list(data_set[data_set.lockdown_end == 1].day)

    if len(starts) < len(ends):
        starts.insert(0, 1)

    if len(ends) < len(starts):
        ends.append(data_set.day.max())

    # Add shape regions
    fig.update_layout(
        shapes=[
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=s,
                y0=0,
                x1=e,
                y1=1,
                fillcolor=LOCKDOWN_COLOR,
                opacity=0.5,
                layer="below",
                line_width=0,
            ) for s, e in zip(starts, ends)]
    )
    return fig


def get_lockdown_plot(data_set: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data_set.day,
        y=data_set.lockdown * 100,
        marker_color=LOCKDOWN_COLOR))
    fig.update_layout(
        title="Population in lockdown",
        xaxis_title="days",
        yaxis_title="% of population",
        showlegend=False,
        template=DEFAULT_THEME,
        yaxis=dict(
            type='linear',
            range=[0, 100],
            ticksuffix='%'))
    return fig
