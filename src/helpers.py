# help functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

states = ['S', 'E', 'A', 'I', 'H', 'D', 'R']


def f(x, params):
    return 2 - np.exp(-params['eps']*x)


    
def get_ch(x, params):
    return np.power((x[0] + x[4]),params['sigma'])


def innovate(x, params, t):
    # evolve one step
    Gamma = infect_rate(x, params, t)
    if t == params['tc']:
        CH = get_ch(x, params)
        trans = np.array([[(1-CH)*(1-Gamma), 0, 0, 0, 0, 0, 0],
                          [1-(1-CH)*(1-Gamma), 1-params['eta'], 0, 0, 0, 0, 0],
                          [0, params['eta'], 1-params['alpha'], 0, 0, 0, 0],
                          [0, 0, params['alpha'], 1-params['mu'], 0, 0, 0],
                          [0, 0, 0, params['mu']*params['gamma'], params['w']*(1-params['phi']) + (1-params['w'])*(1-params['xi']), 0, 0],
                          [0, 0, 0, params['mu']*(1-params['gamma']), (1-params['w'])*params['xi'], 1, 0],
                          [0, 0, 0, 0, params['w']*params['phi'], 0, 1]])
    else:
        trans = np.array([[(1-Gamma), 0, 0, 0, 0, 0, 0],
                         [Gamma, 1-params['eta'], 0, 0, 0, 0, 0],
                         [0, params['eta'], 1-params['alpha'], 0, 0, 0, 0],
                         [0, 0, params['alpha'], 1-params['mu'], 0, 0, 0],
                         [0, 0, 0, params['mu']*params['gamma'], params['w']*(1-params['phi']) + (1-params['w'])*(1-params['xi']), 0, 0],
                         [0, 0, 0, params['mu']*(1-params['gamma']), (1-params['w'])*params['xi'], 1, 0],
                         [0, 0, 0, 0, params['w']*params['phi'], 0, 1]])

    return np.dot(trans,x.reshape(-1,1))
    

def graph(x, params):
    plt.subplot(2,1,1)
    for state in states:
        plt.plot(x.loc[state,:], label=state)
    plt.plot([params['tc'], params['tc']], [0, 1], color='black')
    for tr in params['tr']:
        plt.plot([tr, tr], [0, 1], color='black')
    plt.legend()
    plt.title('Full model')

    plt.subplot(2,1,2)
    plt.plot(x.loc['S',:], label='S')
    plt.plot(x.loc[['E','A','I','H'],:].sum(axis=0), label='I')
    plt.plot(x.loc[['R','D'],:].sum(axis=0), label='R')
    plt.plot([params['tc'], params['tc']], [0, 1], color='black')
    for tr in params['tr']:
        plt.plot([tr, tr], [0, 1], color='black')
    plt.legend()
    plt.title('SIR adaptation')
    plt.show()