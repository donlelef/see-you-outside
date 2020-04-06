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

eps_penalty = 1 # penalty parameter for soft constraints
opti = cs.Opti()

T = 100 # horizon
x = opti.variable(7,T+1)
k = opti.variable(1,T)
eps_soft = opti.variable(1,T)
loss = opti.variable(1,T+1)
x_init = opti.parameter(7,1)
num_habitant = 8570000

# boundery condition
opti.subject_to(loss[1]==0)

# multiple shooting (dynamics)
for i in range(T):
    trans = calculate_trans(x[:,i], params,k[i], i)
    opti.subject_to(x[:,i+1]==trans@x[:,i])
    #opti.subject_to(loss[i+1]==loss[i]-k[i])#**2+10000*(x[3,i]+x[5,i])**2)
    opti.subject_to(loss[i+1]==loss[i]-k[i]+eps_penalty*(eps_soft[i])**2 )#**2+10000*(x[3,i]+x[5,i])**2)

    # control constraints
    opti.subject_to(k[i]<=params['k'])
    opti.subject_to(k[i]>=1)
    opti.subject_to(eps_soft[i]>=0)
    #opti.subject_to(x[4,i]<=0.01)
    opti.subject_to(x[4,i]<=0.01 + eps_soft[i])
opti.subject_to(x[:,0]==x_init)

opti.minimize(loss[-1])#+10000*x[3,-1]**2)
p_opts = {"expand":True}
s_opts = {"max_iter": 1e4}
opti.solver('ipopt',p_opts,s_opts)


# initial state
temp = cs.DM(7,1)
temp[0] = 0.95
temp[1] = 0.05
temp[2] = 0.0
temp[3] = 0.0
temp[4] = 0.0
temp[5] = 0.0
temp[6] = 0.0
opti.set_value(x_init,temp)

sol = opti.solve()

plt.plot(sol.value(k))
plt.show()

plt.plot(sol.value(x)[3,:])
plt.show()

#pd.DataFrame(sol.value(x), index=['S','E','A','I','H','D','R']).to_csv('For_Emanuele.csv')
