''' Here all data generation takes place'''
from datetime import datetime
import math
import pathlib
import copy

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

import config
import template_model


def generate_fake_data():
    n_days = 100
    day_start = datetime(2020,2,1)
    
    
    date_range = pd.date_range(day_start, periods = n_days, freq = '1D')
    x = np.arange(n_days)
    sigma_square = 50
    data = dict()
    data['infected'] = np.exp(-(x - 50)**2/2/sigma_square)/2
    delay = 15
    data['recovered'] = 0.8*np.concatenate([np.zeros(delay), data['infected'][:-delay]])
    data['dead'] = 0.2*np.concatenate([np.zeros(delay), data['infected'][:-delay]])
    data['unaffected'] = 1 - data['infected'] - data['dead'] - data['recovered'] 
    df = pd.DataFrame(data, index=date_range)
    return df

def load_data():
    # df = pd.read_csv('data_generated/data.txt').transpose()
    df = pd.read_csv(config.DATA_PATH / 'data.txt').transpose()
    n_days = df.shape[0]
    day_start = datetime(2020,2,1)
    date_range = pd.date_range(day_start, periods = n_days, freq = '1D')
    df.index = date_range
    return df

def test_plot(): 
    # df = generate_fake_data()
    df = load_data()
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col],
                            name=col))
    fig.show()

    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df[col],
                            name=col))

# def return_plot():
#     ''' return plot from model'''
#     # df = load_data()
    
#     #GEt trajectories
#     x = template_model.simulate()
    
#     # prepare date_range
#     time_start = config.date_start_simulations
#     date_range = pd.date_range(time_start, periods = x.shape[1], freq = 'D')
#     fig = go.Figure()
#     for name, traj in x.iterrows():
#         fig.add_trace(go.Bar(x=date_range, y=traj,
#                             name=name))
#     fig.update_layout(barmode='stack',
#                     #   autosize= False,
#                     #   height=500,
#                       )
#     return fig

def return_plot_from_dates(dates):
    ''' return plot from model'''
    # df = load_data()
    params = copy.deepcopy(template_model.params)
    params['tr'] = dates
    #GEt trajectories
    x = template_model.simulate(params)
    
    # prepare date_range
    time_start = config.date_start_simulations
    date_range = pd.date_range(time_start, periods = x.shape[1], freq = 'D')
    fig = go.Figure()
    for name, traj in x.iterrows():
        fig.add_trace(go.Bar(x=date_range, y=traj,
                            name=name))
    fig.update_layout(barmode='stack',
                    #   autosize= False,
                    #   height=500,
                      )
    return fig


# Change the bar mode
    
    fig.show()

if __name__ == "__main__":
    fig = return_plot_from_dates([20, 30, 40, 50])
    fig.show()