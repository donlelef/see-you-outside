import casadi as cs
import pandas as pd
import matplotlib.pyplot as plt

def f(x, params):
    return 2 - cs.exp(-params['eps']*x)


def infect_rate(x, k,params):
    z = 1/(f(params['n_eff']/params['s'], params))
    x_a = x[2]
    x_i = x[3]

    return 1 - cs.power(1-params['beta_a'], k*params['C']*x_a) *cs.power(1-params['beta_i'], k*params['C']*x_i)
    


def calculate_trans(x, params,k, t):
    # evolve one step
    Gamma = infect_rate(x, k, params)
    # print(t)
    # print(params['tr'])

    trans = cs.MX.zeros(7,7)
    trans[0,0] = (1-Gamma)
    trans[1,0] = Gamma
    trans[1,1] = 1-params['eta']
    trans[2,1] = params['eta']
    trans[2,2] = 1-params['alpha']
    trans[3,2] = params['alpha']
    trans[3,3] = 1-params['mu']
    trans[4,3] = params['mu']*params['gamma']
    trans[4,4] = params['w']*(1-params['phi']) + (1-params['w'])*(1-params['xi'])
    trans[5,4] = params['w']*params['phi']
    trans[5,5] = 1
    trans[6,3] = params['mu']*(1-params['gamma'])
    trans[6,4] = (1-params['w'])*params['xi']
    trans[6,6] = 1
    
    # trans = [[(1-Gamma), 0, 0, 0, 0, 0, 0],
    #                 [Gamma, 1-params['eta'], 0, 0, 0, 0, 0],
    #                 [0, params['eta'], 1-params['alpha'], 0, 0, 0, 0 ],
    #                 [0, 0, params['alpha'], 1-params['mu'], 0, 0, 0 ],
    #                 [0, 0, 0, params['mu']*params['gamma'], params['w']*(1-params['phi']) + (1-params['w'])*(1-params['xi']), 0, 0],
    #                 [0, 0, 0, 0, params['w']*params['phi'], 1, 0],
    #                 [0, 0, 0, params['mu']*(1-params['gamma']), (1-params['w'])*params['xi'], 0, 1]]
    return trans


def opt_strategy(weight_eps, bed_ratio, weight_goout, initial_state = [0.8,0.2,0]):
    # args:
    #   weight_eps: weights on overloading hospital: [0,1]
    #   bed_ratio: bed per person
    #   weight_goout: weights on going out/ economy: [0,1]
    #   initial state: SIR, d not consider because control cannot change it
    params = {}
    params['mobility'] = 0 # only one region
    params['eta'] = 1/2.34 # from exposed to asymptomatic
    params['alpha'] = 1/2.86 # from asymptomatic to infected
    params['mu'] = 1/3.2 # prob leaving infected
    params['gamma'] = 0.13 # conditional prob to icu
    params['phi'] = 1/7.0 # death rate (inverse of time in icu)
    params['w'] = 0.2 # prob death
    params['xi'] = 0.1 # prob recover from ICU


    params['beta_a'] = 0.07 # infectivity of asymptomatic
    params['beta_i'] = 0.07 # infectivity of infected
    params['k'] = 13.3 # average number of contact
    params['C'] = 0.721 # contact rate
    params['eps'] = 0.01 # density factor
    params['sigma']= 2.5 # household size
    params['n_eff'] = 8570000 # effecitve population
    params['s'] = 39133 # area of the region

    eps_penalty = weight_eps*1e5 # penalty parameter for soft constraints,upper bound 1e5
    lockdown_penalty =  weight_goout*8e-2 # upper bound 8e-2 
    death_penalty = weight_eps*5e3   # upper bound 5e3
    bed_per_person = bed_ratio # upper bound 5e-2
    final_infect_penalty = 5e6
    opti = cs.Opti()

    T = 100 # horizon
    x = opti.variable(7,T+1)
    k = opti.variable(1,T)
    eps_soft = opti.variable(1,T)
    loss = opti.variable(1,T+1)
    x_init = opti.parameter(7,1)

    # boundery condition
    opti.subject_to(loss[1]==0)

    # multiple shooting (dynamics)
    for i in range(T):
        trans = calculate_trans(x[:,i], params,k[i], i)
        opti.subject_to(x[:,i+1]==trans@x[:,i])
        #opti.subject_to(loss[i+1]==loss[i]-k[i])#**2+10000*(x[3,i]+x[5,i])**2)
        opti.subject_to(loss[i+1]==loss[i]+lockdown_penalty*(params['k']-k[i])**2+eps_penalty*(eps_soft[i]))

        # control constraints
        opti.subject_to(k[i]<=params['k'])
        opti.subject_to(k[i]>=1)
        opti.subject_to(eps_soft[i]>=0)

        opti.subject_to(eps_soft[i]<=0.1) # reasonable upper bound on available beds
        #opti.subject_to(x[4,i]<=0.01)
        opti.subject_to(x[4,i]<=bed_per_person + eps_soft[i])

        # initialization of value
        opti.set_initial(eps_soft[i],0.1)
        opti.set_initial(k[i],1)

    # boundary conditions
    opti.subject_to(x[:,0]==x_init)
    opti.subject_to(k[0]==1)

    opti.minimize(loss[-1]+death_penalty*x[6,T]*x[6,T]+final_infect_penalty*x[4,T]*x[4,T])
    p_opts = {"expand":True}
    s_opts = {"max_iter": 1e4}
    opti.solver('ipopt',p_opts,s_opts)


    # initial state
    temp = cs.DM(7,1)
    temp[0] = initial_state[0]   # s
    temp[1] = 0.5*initial_state[1]   # e
    temp[2] = 0.4*initial_state[1]  # a
    temp[3] = 0.09*initial_state[1] # i
    temp[4] = 0.01*initial_state[1] # h
    temp[5] = initial_state[2]   # r
    temp[6] = 0.0   # d
    opti.set_value(x_init,temp)

    sol = opti.solve()

    return sol.value(x), sol.value(k), params

# plt.figure(1) 
# plt.clf()
# plt.plot(sol.value(k))

# plt.figure(2)
# plt.clf()
# plt.plot(sol.value(eps_soft))

# plt.figure(3)
# plt.clf()
# plt.plot(sol.value(x)[3,:],label='infected')
# plt.plot(sol.value(x)[4,:],label='hospitalized')
# plt.plot(sol.value(x)[5,:],label='death')
# plt.legend()
# plt.show()

#pd.DataFrame(sol.value(x), index=['S','E','A','I','H','D','R']).to_csv('For_Emanuele.csv')
