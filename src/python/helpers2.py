# help functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping
import time

states = ['S', 'E', 'A', 'I', 'H', 'D', 'R', 'C']


def f(x, params):
    return 2 - np.exp(-params['eps']*x)


def infect_rate(x, params, t):
    x_a = x[2]
    x_i = x[3]
    kappa0 = 0
    
    if t < params['tr'][0]:
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
    return np.power(x[0]+x[7],params['sigma'])


def innovate(x, params, t):
    # evolve one step
    Gamma, kappa0 = infect_rate(x, params, t)
    if t == params['tr'][0]:
        CH = kappa0 * get_ch(x, params)
        print(CH)
        # s e a i h d r
        trans = np.array([[(1-CH)*(1-Gamma), 0, 0, 0, 0, 0, 0, 0],
                          [(1-CH)*Gamma, 1-params['eta'], 0, 0, 0, 0, 0, 0],
                          [0, params['eta'], 1-params['alpha'], 0, 0, 0, 0, 0],
                          [0, 0, params['alpha'], 1-params['mu'], 0, 0, 0, 0],
                          [0, 0, 0, params['mu']*params['gamma'], params['w']*(1-params['phi']) + (1-params['w'])*(1-params['xi']), 0, 0, 0],
                          [0, 0, 0, 0, params['w']*params['phi'], 1, 0, 0],
                          [0, 0, 0, params['mu']*(1-params['gamma']), (1-params['w'])*params['xi'], 0, 1, 0],
                          [CH, 0, 0, 0, 0, 0, 0, 0]])
        return np.dot(trans,x.reshape(-1,1))
                          
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
        return np.dot(trans,x.reshape(-1,1)) + np.array([ch-CH, 0, 0, 0, 0, 0, 0, CH]).reshape(-1,1)
                          
    else:
        trans = np.array([[(1-Gamma), 0, 0, 0, 0, 0, 0, 0],
                         [Gamma, 1-params['eta'], 0, 0, 0, 0, 0, 0],
                         [0, params['eta'], 1-params['alpha'], 0, 0, 0, 0, 0],
                         [0, 0, params['alpha'], 1-params['mu'], 0, 0, 0, 0],
                         [0, 0, 0, params['mu']*params['gamma'], params['w']*(1-params['phi']) + (1-params['w'])*(1-params['xi']), 0, 0, 0],
                         [0, 0, 0, 0, params['w']*params['phi'], 1, 0, 0],
                         [0, 0, 0, params['mu']*(1-params['gamma']), (1-params['w'])*params['xi'], 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1]])
        return np.dot(trans,x.reshape(-1,1))
    
    
def simulate(params, save=False, name=None):
    x = pd.DataFrame(index=states)
    x[1] = params['x_init']

    for t in range(2, params['t_max']+1):
        x[t] = innovate(x[t-1].values, params, t)
       
    if name != None:
        print("Max deaths {}: {:.2f}%".format(name, 100*np.max(x.loc['D'].values)))
        print("Max hospitalizations: {:.2f}%".format(100*np.max(x.loc['H'].values)))
    else:
        print("Max deaths {}: {:.2f}%".format(name, 100*np.max(x.loc['D'].values)))
        print("Max hospitalizations: {:.2f}%".format(100*np.max(x.loc['H'].values)))
    graph(x, params, save, name)
    

def graph(x, params, save=False, name=None):
    plt.subplot(2,1,1)
    for state in states:
        plt.plot(x.loc[state,:], label=state)
    for tr in params['tr']:
        plt.plot([tr, tr], [0, 1], color='black')
    plt.legend()
    plt.title('Full model')

    plt.subplot(2,1,2)
    plt.plot(x.loc[['S', 'C'],:].sum(axis=0), label='S')
    plt.plot(x.loc[['E','A','I','H'],:].sum(axis=0), label='I')
    plt.plot(x.loc[['R','D'],:].sum(axis=0), label='R')
    for tr in params['tr']:
        plt.plot([tr, tr], [0, 1], color='black')
    plt.legend()
    plt.title('SIR adaptation')
    if save:
        plt.savefig(name+'.png')
    plt.show()
    
    plt.subplot(2,1,2)
    plt.plot(x.loc[['I'],:].sum(axis=0), label='I')
    plt.plot(x.loc[['H'],:].sum(axis=0), label='H')
    plt.plot(x.loc[['D'],:].sum(axis=0), label='D')
    for tr in params['tr']:
        plt.plot([tr, tr], [0, 1], color='black')
    plt.legend()
    plt.title('SIR adaptation')
    if save:
        plt.savefig(name+'.png')
    plt.show()
    

def get_costcd(tr, params):
    x = pd.DataFrame(index=states)
    x[1] = params['x_init']
    tr.sort()
    params['tr'] = np.array([int(tr[i]) if i%2==0 else int(tr[i])+1 for i in range(len(tr))]).clip(1,params['t_max']-1)

    for t in range(2, params['t_max']+1):
       x[t] = innovate(x[t-1].values, params, t)

    num_i = x.loc[['D','R'],params['t_max']].sum() # total infected cases(normalized)
    num_h = num_i*params['gamma'] # number of icu cases
    num_d = x.loc['D',params['t_max']] # number of death
    
    tr = list(tr)
    tr.append(params['t_max'])
    gaps = [tr[i+1]-tr[i] for i in range(len(tr)-1)]
    # total days of lockdown
    lockdowns = np.array([gaps[i] if i%2==0 else 0 for i in range(len(gaps))]).sum()

    #t_gap = np.max(gaps) # days without intervention(normalized)
    # t_gap for economic, num_h for medication cost, num_d to repect life

    return -np.dot(params['weights'], np.array([num_i, num_h, num_d, lockdowns]))
    

def optimize(params):
    params_org = params.copy()
    
    tic = time.time()
    tr_opt = basinhopping(get_cost, x0=params['tr'], minimizer_kwargs={'args':params, 'method':'Nelder-Mead'}, stepsize=10, niter=10)
    tr_opt = tr_opt.x
    print(time.time()-tic)
    
    params['tr'] = tr_opt.clip(1,params['t_max']-1)
    
    simulate(params_org, save=True, name='Original')
    print(tr_opt)
    simulate(params, save=True, name='Optimized')
