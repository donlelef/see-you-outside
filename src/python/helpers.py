# help functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

states = ['S', 'E', 'A', 'I', 'H', 'D', 'R', 'C']


def f(x, params):
    return 2 - np.exp(-params['eps']*x)


def infect_rate(x, params, t):
    z = 1/(f(params['n_eff']/params['s'], params))
    x_a = x[2]
    x_i = x[3]
    kappa0 = 1
    
    if t < params['tc']:
        k = params['k']
    else:
        indices = np.where(np.array(params['tr']) > t)[0]
        if len(indices) == len(params['tr']):
            kappa0 = params['kappa0']
        else:
            #k = params['kappa'][min(ind)] * (params['sigma']-1)
            if len(indices) == 0:
                ind = -1
            else:
                ind = min(indices)-1
            kappa0 = 1/params['kappa'][ind]
        k = params['k'] * (1-kappa0) + kappa0 * (params['sigma']-1)

    return 1 - np.power(1-params['beta_a'], k*params['C']*x_a) * np.power(1-params['beta_i'], k*params['C']*x_i), kappa0
    
    
def get_ch(x, params):
    return np.power(x[0]+x[6],params['sigma'])


def innovate(x, params, t):
    # evolve one step
    Gamma, kappa0 = infect_rate(x, params, t)
    # print(t)
    # print(params['tr'])
    if t == params['tc']:
        CH = kappa0 * get_ch(x, params)
        # s e a i h d r
        trans = np.array([[(1-CH)*(1-Gamma), 0, 0, 0, 0, 0, 0, 0],
                          [(1-CH)*Gamma, 1-params['eta'], 0, 0, 0, 0, 0, 0],
                          [0, params['eta'], 1-params['alpha'], 0, 0, 0, 0, 0],
                          [0, 0, params['alpha'], 1-params['mu'], 0, 0, 0, 0],
                          [0, 0, 0, params['mu']*params['gamma'], params['w']*(1-params['phi']) + (1-params['w'])*(1-params['xi']), 0, 0, 0],
                          [0, 0, 0, 0, params['w']*params['phi'], 1, 0, 0],
                          [0, 0, 0, params['mu']*(1-params['gamma']), (1-params['w'])*params['xi'], 0, 1, 0],
                          [CH, 0, 0, 0, 0, 0, 0, 0]])
        #print(np.dot(trans,x.reshape(-1,1)).sum())
        return np.dot(trans,x.reshape(-1,1)), params
                          
    elif t in params['tr']:
        CH = kappa0 * get_ch(x, params)
        ch = x[7]
        #tot = params['S_CH']+params['R_CH']
        # s e a i h d r
        trans = np.array([[1-Gamma, 0, 0, 0, 0, 0, 0, 0],
                          [Gamma, 1-params['eta'], 0, 0, 0, 0, 0, 0],
                          [0, params['eta'], 1-params['alpha'], 0, 0, 0, 0, 0],
                          [0, 0, params['alpha'], 1-params['mu'], 0, 0, 0, 0],
                          [0, 0, 0, params['mu']*params['gamma'], params['w']*(1-params['phi']) + (1-params['w'])*(1-params['xi']), 0, 0, 0],
                          [0, 0, 0, 0, params['w']*params['phi'], 1, 0, 0],
                          [0, 0, 0, params['mu']*(1-params['gamma']), (1-params['w'])*params['xi'], 0, 1, 0],
                          [0 , 0, 0, 0, 0, 0, 0, 0]])
        #print((np.dot(trans,x.reshape(-1,1)) + np.array([ch-CH, 0, 0, 0, 0, 0, 0, CH]).reshape(-1,1)).sum())
        return np.dot(trans,x.reshape(-1,1)) + np.array([ch-CH, 0, 0, 0, 0, 0, 0, CH]).reshape(-1,1), params
                          
    else:
        trans = np.array([[(1-Gamma), 0, 0, 0, 0, 0, 0, 0],
                         [Gamma, 1-params['eta'], 0, 0, 0, 0, 0, 0],
                         [0, params['eta'], 1-params['alpha'], 0, 0, 0, 0, 0],
                         [0, 0, params['alpha'], 1-params['mu'], 0, 0, 0, 0],
                         [0, 0, 0, params['mu']*params['gamma'], params['w']*(1-params['phi']) + (1-params['w'])*(1-params['xi']), 0, 0, 0],
                         [0, 0, 0, 0, params['w']*params['phi'], 1, 0, 0],
                         [0, 0, 0, params['mu']*(1-params['gamma']), (1-params['w'])*params['xi'], 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1]])
        return np.dot(trans,x.reshape(-1,1)), params
    
def simulate(x_init, params):
    x = pd.DataFrame(index=states)
    x[1] = x_init

    for t in range(2, params['t_max']+1):
       x[t], params = innovate(x[t-1].values, params, t)
       #print(x.loc['D',t])
    print("Max deaths: {:.2f}%".format(100*np.max(x.loc['D'].values)))
    print("Max hospitalizaitons: {:.2f}%".format(100*np.max(x.loc['H'].values)))
    graph(x, params)

def int_to_date(int, start):
    return start + int*timedelta(days=1)


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
    plt.plot(x.loc[['S', 'C'],:].sum(axis=0), label='S')
    plt.plot(x.loc[['E','A','I','H'],:].sum(axis=0), label='I')
    plt.plot(x.loc[['R','D'],:].sum(axis=0), label='R')
    plt.plot([params['tc'], params['tc']], [0, 1], color='black')
    for tr in params['tr']:
        plt.plot([tr, tr], [0, 1], color='black')
    plt.legend()
    plt.title('SIR adaptation')
    plt.show()