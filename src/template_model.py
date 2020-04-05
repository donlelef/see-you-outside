from helpers import *
import argparse

# define parameters(only consider one group and one region)
params = {}
params['mobility'] = 0 # only one region
params['eta'] = 1/2.34 # from exposed to asymptomatic
params['alpha'] = 1/2.86 # from asymptomatic to infected
params['mu'] = 1/3.2 # prob leaving infected
params['gamma'] = 0.05 # conditional prob to icu
params['phi'] = 1/7.0 # death rate (inverse of time in icu)
params['w'] = 0.42 # prob death
params['xi'] = 0.1 # prob recover from ICU


params['beta_a'] = 0.06 # infectivity of asymptomatic
params['beta_i'] = 0.06 # infectivity of infected
params['k'] = 13.3 # average number of contact
params['C'] = 0.721 # contact rate
params['eps'] = 0.01 # density factor
params['sigma']= 2.5 # household size
params['n_eff'] = 10000 # effecitve population
params['s'] = 0.0001 # area of the region(guess)


# parameters for control
params['t_max'] = 100
# starting of intervention fully-lockdown
params['tc'] = 20
# time line of removing different intervention(increasig oder)
# last one is remove all interventions and kappa = k again
params['tr'] = [30, 60]
# confinement factor from the strictest one to the slightest one(modeling average contact)
# first entry is lock down
params['kappa'] = [300000, 3/2, 100000, 3/2, 100000]
params['kappa0'] = 2/3
params['k_old'] = params['k']

x_init = [0.99, 0.01, 0, 0, 0, 0, 0, 0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--x_init', type=float, nargs='+', default=x_init)
    parser.add_argument('-tc', '--tc', type=int, default=params['tc'])
    parser.add_argument('-tr', '--tr', type=float, nargs='+', default=params['tr'])
    parser.add_argument('-kappa', '--kappa', type=float, nargs='+', default=params['kappa'])
    parser.add_argument('-tm', '--t_max', type=int, default=params['t_max'])
    args = parser.parse_args()
    
    params['t_max'] = args.t_max
    params['tc'] = args.tc # starting of intervention fully-lockdown
    # time line of removing different intervention(increasig oder)
    # last one is remove all interventions and kappa = k again
    params['tr'] = args.tr
    # confinement factor from the strictest one to the slightest one(modeling average contact)
    # first entry is lock down
    params['kappa'] = args.kappa
    
    x_init = np.array(args.x_init)
    
    simulate(x_init, params)
