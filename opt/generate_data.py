import itertools
import logging
import pathlib
import pickle
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from opt import opt_helpers as opt

SUSCEPTIBLE_COLOR = '#0099d1'
INFECTED_COLOR = '#fa7a2a'
HOSPITALIZED_COLOR = '#bab400'
RECOVERED_COLOR = '#009e45'
DEAD_COLOR = '#9e00ba'
LOCKDOWN_COLOR = '#dbdbdb'

LOCKDOWN_TH = 0.01
DEFAULT_THEME = 'simple_white'


def precompute_solution(*iterables, path_pickle='app_data/solutions.pkl', reset=False):
    ''' gets iterables and computes solutions'''
    if reset:
        dict_solutions = {}
    else:
        if pathlib.Path(path_pickle).is_file():
            with open(path_pickle, 'rb') as f:
                dict_solutions = pickle.load(f)
        else:
            dict_solutions = {}
    i = 0
    for param_combination in itertools.product(*iterables):
        if param_combination in dict_solutions:
            continue

        I, hospital_beds, icus_penalty, lockdown_penalty = param_combination
        print(f"You are calling the model with {hospital_beds} beds, {icus_penalty} "
              f"icu penalty and {lockdown_penalty} lockdown penalty")
        I /= 100
        lockdown_penalty /= 100
        icus_penalty /= 1000
        hospital_beds /= 10000

        try:
            data_set, lockdown, params = opt.opt_strategy(weight_eps=icus_penalty, bed_ratio=hospital_beds,
                                                          weight_goout=lockdown_penalty, initial_infect=I)
            data = pd.DataFrame(data_set, index=['S', 'E', 'A', 'I', 'H', 'D', 'R']).transpose()
            data['beds'] = hospital_beds
            data['lockdown'] = np.append(0, (-1) / 12.3 * lockdown + 13.3 / 12.3)
            data['day'] = np.arange(len(lockdown) + 1)
            dict_solutions[param_combination] = data
        except:
            logging.exception('No solution found with {}'.format(param_combination))

        if i % 100 == 0:
            with open(path_pickle, 'wb') as f:
                pickle.dump(dict_solutions, f)
        i += 1


def generate_data_from_model(initial_infected, hospital_beds, icus_penalty, lockdown_penalty):
    print(f"You are calling the model with {hospital_beds} beds, {icus_penalty} "
          f"icu penalty and {lockdown_penalty} lockdown penalty")
    time.sleep(2)
    initial_infected /= 100
    lockdown_penalty /= 100
    icus_penalty /= 1000
    hospital_beds /= 10000
    print('parameters: {},{},{},{}'.format(initial_infected, hospital_beds, icus_penalty, lockdown_penalty))
    data_set, lockdown, params = opt.opt_strategy(weight_eps=icus_penalty, bed_ratio=hospital_beds,
                                                  weight_goout=lockdown_penalty, initial_infect=initial_infected)

    data = pd.DataFrame(data_set, index=['S', 'E', 'A', 'I', 'H', 'D', 'R']).transpose()
    data['beds'] = hospital_beds
    data['lockdown'] = np.append(0, (-1) / 12.3 * lockdown + 13.3 / 12.3)
    data['day'] = np.arange(len(lockdown) + 1)

    return data.iloc[:81, :]


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
        showlegend=True,
        title="Infection curve",
        xaxis_title="day",
        yaxis_title="% of population",
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
        name='Hospitalizations',
        line={'color': HOSPITALIZED_COLOR}))
    fig.add_trace(go.Scatter(
        x=data_set.day,
        y=data_set.D * 100,
        mode='lines',
        name='Deaths',
        line={'color': DEAD_COLOR}))
    fig.add_trace(go.Scatter(
        x=data_set.day,
        y=data_set.beds * 100,
        mode='lines',
        name='Hospital beds',
        line={'dash': 'dot', 'color': '#000000'}))
    fig.update_layout(
        title="Hospitalized and dead people",
        xaxis_title="day",
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
        xaxis_title="day",
        yaxis_title="% of population",
        showlegend=False,
        template=DEFAULT_THEME,
        yaxis=dict(
            type='linear',
            range=[0, 100],
            ticksuffix='%'))
    return fig


if __name__ == "__main__":
    precompute_solution([5, 10], [5, 10], [10, 20], [10, 20])
