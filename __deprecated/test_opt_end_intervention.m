% in this script I will optimize the end time of intervention given a
% fixed starting date of intervention
clear all;
close all;
clc;

%% define parameters(only consider one group and one region)
params = struct;
params.mobility = 0; % only one region
params.eta = 1/2.34; ... from exposed to asymptomatic
params.alpha = 1/2.86; % from asymptomatic to infected
params.mu = 1/3.2; % prob leaving infected
params.gamma = 0.05; % conditional prob to icu
params.phi = 1/7.0; % death rate (inverse of time in icu)
params.w = 0.42; %prob death
params.xi = 0.1; % prob recover from ICU

% parameters for control
params.beta_a = 0.06; % infectivity of asymptomatic
params.beta_i = 0.06; % infectivity of infected
params.k = 13.3; % average number of contact
params.C = 0.721; % contact rate
params.eps = 0.01; % density factor
params.sigma = 2.5; % household size
params.kappa = 1; %confinement factor
params.n_eff = 10000; % effecitve population
params.s = 1e4; % area of the region(guess)


x_init = [0.9;0.1;zeros(5,1)];
sim = struct;
sim.x = x_init;
sim.x_c = x_init;
params.tc= 10; % starting of intervention

% simulated anealing
options = anneal();
loss_func = @(tr) get_cost(tr,x_init,params); % loss function handle
options.Generator = @(x) center_gen(x,params.tc);   % custom sampler
options.MaxTries = 200; options.MaxSuccess = 30;
tr = 40; % initial guess
[optimal_tr,optimal_cost] = anneal(loss_func,tr,options);


%% help function
function x_next = innovate(x,params,i)
    % evolve one step
    Gamma = infect_rate(x,params,i);
    if i == params.tc
        CH = get_ch(x,params);
                % s, e, a, i, h, r, d
        trans = [(1-CH)*(1-Gamma), 0, 0, 0, 0, 0, 0;
                1-(1-CH)*(1-Gamma), 1-params.eta, 0, 0, 0, 0, 0;
                0,params.eta,1-params.alpha,0,0,0, 0;
                0,0,params.alpha,1-params.mu,0,0, 0;
                0,0, 0, params.mu*params.gamma, params.w*(1-params.phi)+(1-params.w)*(1-params.xi),0,0;
                0,0,0,params.mu*(1-params.gamma),(1-params.w)*params.xi,1,0;
                0,0,0,0,params.w*params.phi,0,1];
       
    else
                 % s, e, a, i, h, r, d
        trans = [1-Gamma, 0, 0, 0, 0, 0, 0;
                Gamma, 1-params.eta, 0, 0, 0, 0, 0;
                0,params.eta,1-params.alpha,0,0,0, 0;
                0,0,params.alpha,1-params.mu,0,0, 0;
                0,0, 0, params.mu*params.gamma, params.w*(1-params.phi)+(1-params.w)*(1-params.xi),0,0;
                0,0,0,params.mu*(1-params.gamma),(1-params.w)*params.xi,1,0;
                0,0,0,0,params.w*params.phi,0,1];
        
    end
    x_next = trans*x;

end

function P = infect_rate(x,params,i)
   f = @(x) 2-exp(-params.eps*x);
   z = 1/(f(params.n_eff/params.s));
   x_a = x(3)*params.n_eff;
   x_i = x(4)*params.n_eff;
   if i<params.tc
       k = params.k;
   elseif i>=params.tr
       k = params.k;
   else
       k = params.kappa*(params.sigma-1);
   end
   P = 1-(1-params.beta_a)^(z*k*f(params.n_eff/params.s)*params.C*x_a/params.n_eff)...
            *(1-params.beta_i)^(z*k*f(params.n_eff/params.s)*params.C*x_i/params.n_eff);

end

function CH = get_ch(x,params)
    CH =(x(1)+x(5))^params.sigma;
end

function loss = get_cost(tr,x_init,params)
    params.tr = tr;
    x = x_init;
    sim_end = 100;
    for i = 1:sim_end
        x = innovate(x(:,end),params,i);
    end
    % estimate loss
    num_i = sum(x(end-1:end)); % total infected cases(normalized)
    num_h = num_i*params.gamma; % number of icu cases
    num_d = x(end); % number of death
    t_gap = (sim_end-tr+1)/sim_end; % days without intervention(normalized)
    % t_gap for economic, num_h for medication cost, num_d to repect life
    loss = -0.01*t_gap+num_h*0.01+num_d;
end

function x = center_gen(x,tc)
    % sampling function
    x = x+3*randn(1)+30*(rand(1)-1);
    x = max(1+tc,ceil(x)); % at least one day after intervention
end
    
