% data from github covid19-cases-switzerland
% Here there is also data. We can se if it's the same or we can
% complemented https://covid-19-schweiz.bagapps.ch/fr-1.html

%fname = 'covid19_cases_switzerland_openzh.json';
%val = jsondecode(fileread(fname));

% starting date: 25feb2020 - day1
% Using data up to Friday 13mar2020 - day18

T_cases = readtable('covid19_cases_switzerland_openzh.csv');
cases_CH = table2array(T_cases(1:18,28));

T_fat = readtable('covid19_fatalities_switzerland_openzh.csv');
fat_CH = table2array(T_fat(1:18,28));

T_hos = readtable('covid19_hospitalized_switzerland_openzh.csv');
hos_CH = table2array(T_hos(1:18,28));

T_rel = readtable('covid19_released_switzerland_openzh.csv');
rel_CH = table2array(T_rel(1:18,28));

days = linspace(18,18+18,18);

figure(3)
plot(days, cases_CH)
hold on
plot(days, fat_CH)
plot(days, hos_CH)

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

params2 = [0; 1/2.34 ; 1/2.86 ; 1/3.2 ; 0.05 ; 1/7.0 ; 0.42 ; 0.1 ; 0.06 ; 0.06 ; 13.3 ; 0.721 ; 0.01 ; 2.5 ; 1 ];
% parameters for initial infected
iniI = 1/8.57e6;

x_init = [1-iniI ; iniI ; zeros(5,1)];

sim = struct;
sim.x = x_init;

params.tc= 101; % starting of intervention(not intervented in this case)
for i = 1:100
    sim.x = [sim.x,innovate(sim.x(:,end),params,i)];
end

figure(1);clf; hold on
for i = 1:7
    plot([1:size(sim.x,2)],sim.x(i,:)*8.57e6);
end
axis([17,37,1,1200])
legend('s','e','a','i','h','r','d');
title('uncontrolled case in Spain')

data = [cases_CH ; hos_CH ; fat_CH];

loss = @(param) 0.5*(data'*simu(x_init,param)).^2;

ac= loss(params2);
[minimum,fval] = anneal(loss, params2);

params2 = minimum(:,15);

params.mobility = params2(1); % only one region
params.eta = params2(2); ... from exposed to asymptomatic
params.alpha = params2(3); % from asymptomatic to infected
params.mu = params2(4); % prob leaving infected
params.gamma = params2(5); % conditional prob to icu
params.phi = params2(6); % death rate (inverse of time in icu)
params.w = params2(7); %prob death
params.xi = params2(8); % prob recover from ICU

% parameters for control
params.beta_a = params2(9); % infectivity of asymptomatic
params.beta_i = params2(10); % infectivity of infected
params.k = params2(11); % average number of contact
params.C = params2(12); % contact rate
params.eps = params2(13); % density factor
params.sigma = params2(14); % household size
params.kappa = params2(15); %confinement factor

sim = struct;
sim.x = x_init;

params.tc= 101; % starting of intervention(not intervented in this case)
for i = 1:100
    sim.x = [sim.x,innovate(sim.x(:,end),params,i)];
end

figure(4);clf; hold on
for i = 1:7
    plot([1:size(sim.x,2)],sim.x(i,:)*8.57e6);
end
axis([17,37,1,1200])
legend('s','e','a','i','h','r','d');
title('uncontrolled case Switzerland ?')



%% Innovate funciton
function x_out = simu(x_init,params2)
    sim = struct;
    sim.x = x_init;
    
    params = struct;
    params.mobility = params2(1); % only one region
    params.eta = params2(2); ... from exposed to asymptomatic
    params.alpha = params2(3); % from asymptomatic to infected
    params.mu = params2(4); % prob leaving infected
    params.gamma = params2(5); % conditional prob to icu
    params.phi = params2(6); % death rate (inverse of time in icu)
    params.w = params2(7); %prob death
    params.xi = params2(8); % prob recover from ICU

    % parameters for control
    params.beta_a = params2(9); % infectivity of asymptomatic
    params.beta_i = params2(10); % infectivity of infected
    params.k = params2(11); % average number of contact
    params.C = params2(12); % contact rate
    params.eps = params2(13); % density factor
    params.sigma = params2(14); % household size
    params.kappa = params2(15); %confinement factor
    params.n_eff = 8.57e6; % effecitve population
    params.s = 8.57e6; % area of the region(guess)

    params.tc= 101; % starting of intervention(not intervented in this case)
    for i = 1:100
        sim.x = [sim.x,innovate(sim.x(:,end),params,i)];
    end    
    x_out = [sim.x(4,18:35)' ; sim.x(5,18:35)' ; sim.x(7,18:35)'];
    
end

function x_next = innovate(x,params,i)
    % evolve one step
    Gamma = infect_rate(x,params,i);
             % s, e, a, i, h, r, d
    trans = [1-Gamma, 0, 0, 0, 0, 0, 0;
            Gamma, 1-params.eta, 0, 0, 0, 0, 0;
            0,params.eta,1-params.alpha,0,0,0, 0;
            0,0,params.alpha,1-params.mu,0,0, 0;
            0,0, 0, params.mu*params.gamma, params.w*(1-params.phi)+(1-params.w)*(1-params.xi),0,0;
            0,0,0,params.mu*(1-params.gamma),(1-params.w)*params.xi,1,0;
            0,0,0,0,params.w*params.phi,0,1];
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
       k = params.kappa*(params.sigma-1);
   end
   P = 1-(1-params.beta_a)^(z*k*f(params.n_eff/params.s)*params.C*x_a/params.n_eff)...
            *(1-params.beta_i)^(z*k*f(params.n_eff/params.s)*params.C*x_i/params.n_eff);

end