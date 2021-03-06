% template code for optimizing intervention removals from fully lock-down
clear all; close all; clc;

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
params.n_eff = 10000; % effecitve population
params.s = 1e4; % area of the region(guess)

params.tc= 10; % starting of intervention fully-lockdown
% time line of removing different intervention(increasig oder)
% last one is remove all interventions and kappa = k again
params.tr = [20,30,40,50]; 
%confinement factor from the strictest one to the slightest one(modeling average contact)
% first entry is lock down
params.kappa = [1,3,5,7]; 
params.k_old = params.k;

% simulated anealing
x_init = [0.9;0.1;zeros(5,1)];
sim_end = 100; % end time of simulation
options = anneal();
loss_func = @(tr) get_cost(tr,x_init,params,sim_end); % loss function handle
options.Generator = @(x) center_gen(x,params.tc,size(params.kappa,2),sim_end);   % custom sampler
options.MaxTries = 200; options.MaxSuccess = 30;
tr = [24    25    97   101]; % initial guess
[optimal_tr,optimal_cost] = anneal(loss_func,tr,options);

% simulation for the optimal solution
params.tr = optimal_tr;
sim = struct;
sim.x = x_init;
for i = 2:sim_end
    sim.x = [sim.x,innovate(sim.x(:,end),params,i)];
end
figure(1);clf; hold on
for i = 1:7
    plot([1:size(sim.x,2)],sim.x(i,:));
end
% critical time points
temp = [params.tc,params.tr(find(params.tr<=100))]; 
plot(temp,sim.x(:,temp),'*');
title('prediction')
legend('s','e','a','i','h','r','d','critical time stamps');


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
   else
       index=min(find(params.tr>=i));
       if isempty(index) % full recover remove all interventions
           k = params.k;
       else
           k = params.kappa(index)*(params.sigma-1);
       end
   end
   P = 1-(1-params.beta_a)^(z*k*f(params.n_eff/params.s)*params.C*x_a/params.n_eff)...
            *(1-params.beta_i)^(z*k*f(params.n_eff/params.s)*params.C*x_i/params.n_eff);

end

function CH = get_ch(x,params)
    CH =(x(1)+x(5))^params.sigma;
end

function loss = get_cost(tr,x_init,params,sim_end)
    params.tr = tr;
    x = x_init;
    for i = 2:sim_end
        x = innovate(x(:,end),params,i);
    end
    % estimate loss
    num_i = sum(x(end-1:end)); % total infected cases(normalized)
    num_h = num_i*params.gamma; % number of icu cases
    num_d = x(end); % number of death
    t_gap = max((sim_end-tr+1)/sim_end,0); % days without intervention(normalized)
    % t_gap for economic, num_h for medication cost, num_d to repect life
    weights = -[0.06,0.08,0.09,0.1]; % benefits weights for removing one more intervention
    loss = -weights*t_gap'+num_h*0.01+num_d;
end

function x = center_gen(x,tc,dim,sim_end)
    % sampling function
    % Args:
    %    dim: number of interventions that can be romoved
    
    x = x+3*randn(1,dim)+20*(rand(1,dim)-1);
    % sort to increasing and ensure at least one day of fully lockdown
    x = min(max(1+tc,ceil(sort(x))),sim_end+1); 
end


