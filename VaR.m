%% Statistical characterisation of high-frequency time series
% Parametric & non-parametric VaR/CVaR forecasting

clear all;
close all;

% Inputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

crypto = 'TRX';
tau = 1; % timescale
start_time = '02-Jan-2018 11:00:00';
end_time = '14-Jun-2018 17:00:00';
sets = 10; % k-folds
train = .7; % fraction of dataset for training
val = .3; % fraction of training dataset for validation
thresh = [.15 .1 .05]; % var thresholds

% Import & format data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (strcmp(crypto, 'EURUSD'))
    data = readtable(strcat(crypto,'_merged.txt'));
    table = table2timetable(data);
else
    TR = timerange(start_time,end_time);
    data = importdata(strcat(crypto,'_merged.txt'));
    posix = data.data(:,2);
    time = datetime(posix, 'ConvertFrom', 'Posixtime');
    price = data.data(:, 12);
    table = timetable(time, price);
    table = table(TR,:);
end
table.Properties.VariableNames = {crypto};

% Compute log-returns %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

r = log(table{2:end,1:end}./table{1:end-1,1:end});

% Plot price evolution %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(1); clf; hold on
plot(table.time, table{:,1})
%title([crypto ' price, January - June 2018'])
ylabel(['Price'])
xlim(datetime(2018,[1 6],[2 14]))
legend(crypto)

% Accumulate log-returns for different time horizon %%%%%%%%%%%%%%%%%%%%%%%

aux = [];
for t = 0:tau:length(r)-tau
    
    aux = [aux; sum(r(t+1:t+tau))];  
    
end
r = aux;

% Compute rolling VaR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 72; % length of window

rolling_var = zeros(1, length(r)-n+1);
for t = 1:(length(r)-n+1)
    rolling_var(t) = quantile(r(t:(t+n-1)), .05, 1);
end
rolling_var = [zeros(1, length(r) - length(rolling_var)), rolling_var];

% Plot log-returns & 3 days rolling 95% VaR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(2); clf; hold on;
plot(table.time(2:end), r)
hold on;
plot(table.time(2:end), rolling_var, 'r', 'LineWidth', 1.5)
%title([crypto ' hourly log-returns, January - June 2018'])
xlim(datetime(2018,[1 6],[2 14]))
legend('log-return', 'rolling 95% VaR')
set(gca,'FontSize', 12)

% Compute the first four moments %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = length(r(:,1));
m = mean(r); % Compute mean and store value in variable
std_ = std(r, 0, 1); % Compute standard deviation
sk = skewness(r); % Compute skew
kurt = kurtosis(r); % Compute kurtosis
ex_kurt = kurt - 3; % Compute excess kurtosis
stats = [m', std_', sk', kurt' ex_kurt'];
stats = array2table(stats);
stats.Properties.VariableNames = {'Mean', 'Std', 'Skew', 'Kurtosis', 'Excess_K'};
stats = addvars(stats, {crypto}, 'Before', 'Mean');
stats.Properties.VariableNames = {'Cryptocurrency','Mean', 'Std', 'Skew', 'Kurtosis', 'Excess_K'};
stats

% Plot of empirical PDF vs Gaussian PDF with same mean/std %%%%%%%%%%%%%%%%

minr = min(r);
maxr = max(r);
x = linspace(minr, maxr, 100);
g = exp(-(x-m).^2/(2*std_^2))/sqrt(2*pi*std_^2);
NB = 50;
[b,a] = histnorm(r, NB); % Normalized histogram of returns with NB bins
figure(3);
semilogy(a, b,'ob','MarkerSize',6,'MarkerFaceColor','b')
hold on;
semilogy(x,g,'r','LineWidth',2)
xlim(1.2*[minr maxr])
ylim([0.03 100])
%title(crypto)
legend('Empirical PDF', 'Normal PDF')
set(gca,'FontSize', 12)
xlabel('log-return')

% Plot of Empirical CCDF vs Gaussian %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pos_r = r(r>0);
neg_r = abs(r(r<0));

% positive
xp = sort(pos_r); % Returns sorted in ascending order
yp = 1:1:length(pos_r); 
yp = 1 - yp/(length(pos_r)+1); % Calculating CCDF

% negative
xn = sort(neg_r); % Returns sorted in ascending order
yn = 1:1:length(neg_r); 
yn = 1 - yn/(length(neg_r)+1); % Calculating CCDF

% normal
c = (1 - erf((xp-m)/(std_*sqrt(2))));

% plot
figure(4)
loglog(xp,yp,'o','MarkerSize', 2, 'MarkerEdgeColor','b')
hold on
loglog(xn,yn,'o', 'MarkerSize', 2, 'MarkerEdgeColor', 'r')
loglog(xp, c, 'green', 'LineWidth', 2)
ylim([1e-4 1])
xlim([min(abs(r)) 1])
xlabel('log-return')
ylabel('complementary cumulative distribution')
legend({'positive' 'negative' 'normal'})
set(gca,'FontSize', 12)

%%  Fit distributions to log-returns

% 1) Split dataset into train-validate-test sets

r_fit = r(1:round(length(r)*train)); % fraction of dataset set for train/val
r_train = r_fit(1:round(length(r_fit)*train)); % fraction of train/val set for training
r_val = r_fit(round(length(r_fit)*train)+1:end); % fraction of train/val set for validation
r_test = r(round(length(r)*train)+1:end); % fraction of data set for testing

% 2) Fit distributions to training set using MLE and MOM

%%% Normal distribution
mu = mean(r_train);
sigma = std(r_train);
pd_normal = makedist('Normal', 'mu', mu, 'sigma', sigma);

%%% Student T distribution

%%% MOM
nu_mom_s = 4 + 1/(kurtosis(r_train) - 3);
sigma_mom_s = sqrt((var(r_train)*nu_mom_s - var(r_train)*2)/nu_mom_s);
pd_t_mom = makedist('tLocationScale', 'mu', mu, 'sigma', sigma_mom_s, 'nu', nu_mom_s);

%%% MLE
params_mle_s = fitdist(r_train, 'tLocationScale');
mu_mle_s = params_mle_s.mu;
sigma_mle_s = params_mle_s.sigma;
nu_mle_s = params_mle_s.nu;
pd_t_mle = makedist('tLocationScale', 'mu', mu_mle_s, 'sigma', sigma_mle_s, 'nu', nu_mle_s);

% 3) fit distributions with bootstrap method to test parameter
% robustness

bts = 0.8; % Fraction of data to be retained in each bootstrap sample
Nbts = 1000; % Number of bootstrap samples
alpha = 0.9; % Significance level

%%% Normal distribution (Note: MoM parameter estimators are the same as the
%%% for MLE).

mu_mom = zeros(1, Nbts); % Vector to collect bootstrap estimates for MOM mean
sigma_mom = zeros(1, Nbts); % Vector to collect bootstrap estimates for MOM s.d.
mu_mle = zeros(1, Nbts); % Vector to collect bootstrap estimates for MLE mu
sigma_mle = zeros(1, Nbts); % Vector to collect bootstrap estimates for MLE s.d.

for i = 1:Nbts
    r_bts = r_train(randperm(length(r_train))); % Random permutation of returns
    r_bts = r_bts(1:round(bts*length(r_bts))); % Bootstrapping bts% of returns 
    mu_mom(i) = mean(r_bts); % Method of Moments calculation of mu
    sigma_mom(i) = std(r_bts); % Method of Moments calculation of sigma
    params_mle = fitdist(r_bts, 'Normal'); % MLE fit of bootstrap sample
    mu_mle(i) = params_mle.mu;
    sigma_mle(i) = params_mle.sigma;
end

% sort estimates
mu_mom = sort(mu_mom); 
sigma_mom = sort(sigma_mom);
mu_mle = sort(mu_mle);
sigma_mle = sort(sigma_mle);

% compute confidence interval for each parameter estimator
fprintf('MOM mu CI at %3.2f CL: [%4.3f; %4.3f] \n',alpha, mu_mom(round(0.5*(1-alpha)*Nbts)), mu_mom(round(0.5*(1+alpha)*Nbts)));
fprintf('MOM sigma CI at %3.2f CL: [%4.3f; %4.3f] \n',alpha, sigma_mom(round(0.5*(1-alpha)*Nbts)), sigma_mom(round(0.5*(1+alpha)*Nbts)));
fprintf('MLE mu CI at %3.2f CL: [%4.3f; %4.3f] \n',alpha, mu_mle(round(0.5*(1-alpha)*Nbts)), mu_mle(round(0.5*(1+alpha)*Nbts)));
fprintf('MLE sigma CI at %3.2f CL: [%4.3f; %4.3f] \n',alpha, sigma_mle(round(0.5*(1-alpha)*Nbts)), sigma_mle(round(0.5*(1+alpha)*Nbts)));

%%% Student-t distribution 

mu_mom_st = zeros(1, Nbts); % Vector to collect bootstrap estimates for MOM mean
sigma_mom_st = zeros(1, Nbts); % Vector to collect bootstrap estimates for MOM s.d.
nu_mom_st = zeros(1, Nbts); % Vector to collect bootstrap estimates for MOM nu
mu_mle_st = zeros(1, Nbts); % Vector to collect bootstrap estimates for MLE mu
sigma_mle_st = zeros(1, Nbts); % Vector to collect bootstrap estimates for MLE s.d.
nu_mle_st = zeros(1, Nbts); % Vector to collect bootstrap estimates for MLE nu

for i = 1:Nbts
    r_bts = r_train(randperm(length(r_train))); % Random permutation of returns
    r_bts = r_bts(1:round(bts*length(r_bts))); % Bootstrapping bts% of returns 
    mu_mom_st(i) = mean(r_bts); % Method of Moments calculation of mu
    nu_mom_st(i) = 4 + 1/(kurtosis(r_bts) - 3); % Method of Moments calculation of nu
    sigma_mom_st(i) = sqrt((var(r_bts)*nu_mom_st - var(r_bts)*2)/nu_mom_st); % Method of Moments calculation of sigma
    params_mle_st = fitdist(r_bts, 'tLocationScale'); % MLE fit of bootstrap sample
    mu_mle_st(i) = params_mle_st.mu;
    sigma_mle_st(i) = params_mle_st.sigma;
    nu_mle_st(i) = params_mle_st.nu;
end

% sort estimates
mu_mom_st = sort(mu_mom_st); 
sigma_mom_st = sort(sigma_mom_st); 
nu_mom_st = sort(nu_mom_st); 
mu_mle_st = sort(mu_mle_st); 
sigma_mle_st = sort(sigma_mle_st); 
nu_mle_st = sort(nu_mle_st);

% Compute CI for parameter estimators
fprintf('MOM mu CI at %3.2f CL: [%4.3f; %4.3f] \n',alpha, mu_mom_st(round(0.5*(1-alpha)*Nbts)), mu_mom_st(round(0.5*(1+alpha)*Nbts)));
fprintf('MOM sigma CI at %3.2f CL: [%4.3f; %4.3f] \n',alpha, sigma_mom_st(round(0.5*(1-alpha)*Nbts)), sigma_mom_st(round(0.5*(1+alpha)*Nbts)));
fprintf('MOM nu CI at %3.2f CL: [%4.3f; %4.3f] \n',alpha, nu_mom_st(round(0.5*(1-alpha)*Nbts)), nu_mom_st(round(0.5*(1+alpha)*Nbts)));
fprintf('MLE mu CI at %3.2f CL: [%4.3f; %4.3f] \n',alpha, mu_mle_st(round(0.5*(1-alpha)*Nbts)), mu_mle_st(round(0.5*(1+alpha)*Nbts)));
fprintf('MLE sigma CI at %3.2f CL: [%4.3f; %4.3f] \n',alpha, sigma_mle_st(round(0.5*(1-alpha)*Nbts)), sigma_mle_st(round(0.5*(1+alpha)*Nbts)));
fprintf('MLE nu CI at %3.2f CL: [%4.3f; %4.3f] \n',alpha, nu_mle_st(round(0.5*(1-alpha)*Nbts)), nu_mle_st(round(0.5*(1+alpha)*Nbts)));


%%% 4) Test fit on validation set using QQ-Plot

figure(5); clf; hold on;
subplot(3, 1, 1)
qqplot(r_val, pd_normal)
title('QQ Plot of Sample Data versus Normal Distribution')
xlabel('Quantiles of Normal Distribution')
ylabel('Quantiles of Sample')
subplot(3, 1, 2)
qqplot(r_val, pd_t_mom)
title('QQ Plot of Sample Data versus Student-t Distribution (M.O.M)')
xlabel('Quantiles of Student-t Distribution (M.O.M.)')
ylabel('Quantiles of Sample')
subplot(3, 1, 3)
qqplot(r_val, pd_t_mle)
title('QQ Plot of Sample Data versus Student-t Distribution (M.L.E.)')
xlabel('Quantiles of Student-t Distribution (M.L.E.)')
ylabel('Quantiles of Sample')

%%% 5) Test fit using Kolmogorov-Smirnov test

%[ks_test_normal, normal_p] = kstest(r_val, [r_val cdf(pd_normal, r_val)], 0.1, 'Tail', 'unequal')
%[ks_test_t_mom, t_mom_p] = kstest(r_val, [r_val cdf(pd_t_mom, r_val)], 0.1, 'Tail', 'unequal')
%[ks_test_t_mle, t_mle_p] = kstest(r_val, [r_val cdf(pd_t_mle, r_val)], 0.00001, 'Tail', 'unequal')

[ks_test_normal, normal_p] = kstest(r_val, fitdist(r_train, 'tLocationScale'))
[ks_test_t_mom, t_mom_p] = kstest(r_val, fitdist(r_train, 'Stable'))

%kstable = [ks_test_normal normal_p; ks_test_t_mom t_mom_p; ks_test_t_mle t_mle_p];
%kstable = array2table(kstable);
%kstable.Properties.VariableNames = {'Hypothesis_Rejected', 'p_Value'};

% 6) Test fit plotting ECDF vs CDF

x = linspace(min(r_val), max(r_val), 100);
figure(6); clf; hold on;
subplot(3, 1, 1)
cdfplot(r_val)
hold on;
plot(x, cdf(pd_normal, x));
legend('ECDF', 'Normal (MOM/MLE)')
xlim([-0.1 0.1])
xlabel('log-returns')
subplot(3, 1, 2)
cdfplot(r_val)
hold on;
plot(x, cdf(pd_t_mom, x));
legend('ECDF', 'Student-t (MOM)')
xlim([-0.1 0.1])
xlabel('log-returns')
subplot(3, 1, 3)
cdfplot(r_val)
hold on;
plot(x, cdf(pd_t_mle, x));
legend('ECDF', 'Student-t (MLE)')
xlim([-0.1 0.1])
xlabel('log-returns')

%%% 7) Test fit plotting histogram vs data

xx = linspace(min(r_val), max(r_val), 100);

figure(7); clf; hold on;
subplot(3, 1, 1)
plot(xx, pdf(pd_normal, xx), 'LineWidth', 1)
hold on;
histogram(r_val, 100, 'Normalization', 'pdf')
subplot(3, 1, 2)
plot(xx, pdf(pd_t_mom, xx), 'LineWidth', 1)
hold on;
histogram(r_val, 100, 'Normalization', 'pdf')
subplot(3, 1, 3)
plot(xx, pdf(pd_t_mle, xx), 'LineWidth', 1)
hold on;
histogram(r_val, 100, 'Normalization', 'pdf')

% 8) Test forecast accuracy on test_set

r_test_var_5 = quantile(r_test, 0.05, 1);
r_test_var_10 = quantile(r_test, 0.1, 1);
r_test_cvar_5 = mean(r_test(r_test<r_test_var_5));
r_test_cvar_10 = mean(r_test(r_test<r_test_var_10));
test_var = [r_test_var_10 r_test_var_5 r_test_cvar_10 r_test_cvar_5];

% Forecasts

%%% Student T distribution

%%% MOM
nu_mom_s = 4 + 1/(kurtosis(r_fit) - 3);
sigma_mom_s = sqrt((var(r_fit)*nu_mom_s - var(r_fit)*2)/nu_mom_s);
pd_t_mom = makedist('tLocationScale', 'mu', mu, 'sigma', sigma_mom_s, 'nu', nu_mom_s);

% VaR
t_var_5 = icdf(pd_t_mom, 0.05);
t_var_10 = icdf(pd_t_mom, 0.1);
% CVaR
x = linspace(min(r_fit), t_var_5, 1000);
t_cvar_5 = -trapz(pdf(pd_t_mom, x))*(t_var_5 - min(r_fit))/1000;
x = linspace(min(r_fit), t_var_10, 1000);
t_cvar_10 =  -trapz(pdf(pd_t_mom, x))*(t_var_10 - min(r_fit))/1000;
mom_forecast = [t_var_10 t_var_5 t_cvar_10 t_cvar_5];

%%% MLE
params_mle_s = fitdist(r_fit, 'tLocationScale');
mu_mle_s = params_mle_s.mu;
sigma_mle_s = params_mle_s.sigma;
nu_mle_s = params_mle_s.nu;
pd_t_mle = makedist('tLocationScale', 'mu', mu_mle_s, 'sigma', sigma_mle_s, 'nu', nu_mle_s);

% VaR
mle_var_5 = icdf(pd_t_mle, 0.05);
mle_var_10 = icdf(pd_t_mle, 0.1);
% CVaR
x = linspace(min(r_fit), mle_var_5, 1000);
mle_cvar_5 = -trapz(pdf(pd_t_mle, x))*(mle_var_5 - min(r_fit))/1000;
x = linspace(min(r_fit), mle_var_10, 1000);
mle_cvar_10 =  -trapz(pdf(pd_t_mle, x))*(mle_var_10 - min(r_fit))/1000;
mle_forecast = [mle_var_10 mle_var_5 mle_cvar_10 mle_cvar_5];

% Compare
results = array2table( 100*[test_var; mom_forecast; mle_forecast] )

figure(8); clf; hold on;
x = linspace(min(r_fit), max(r_fit), 1000);
y = pdf(pd_t_mle, x);
ymom = pdf(pd_t_mom, x);
semilogy(x, y, 'LineWidth', 1)
hold on
semilogy(x, ymom, 'r', 'LineWidth', 1)
set(gca,'YScale','log')
hold on
[b,a] = histnorm(r_test, 50);
semilogy(a, b, 'o', 'MarkerFaceColor', 'black')
hold on;
legend('Student-t (MLE)', 'Student-t (MoM)', 'Test')
ylim([0.25 40])
xlabel('log-return')

figure(9); clf; hold on;
subplot(2, 1, 1)
qqplot(r_test, pd_t_mom)
title('QQ Plot of Sample Data vs Student-t (MOM)')
subplot(2, 1, 2)
qqplot(r_test, pd_t_mle)
title('QQ Plot of Sample Data vs Student-t (MLE)')

%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Fitting right & left tail via Maximum Likelihood & Bootstrap %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

p = 0.1; % Defining tails as top p% of returns (both positive and negative)
bts = 0.8; % Fraction of data to be retained in each bootstrap sample
Nbts = 5000; % Number of bootstrap samples
alpha = 0.9; % Significance level

figure(10)

%%% Right tail

r_train = sort(r_train); % Sorting returns
r_right = r_train(round((1-p)*length(r_train)):end); % Selecting top p% of returns

N = length(r_right); % Number of returns selected as right tail
alpha_right = N/sum(log(r_right/min(r_right))); % Maximum-likelihood estimate for right tail exponent

fprintf('Right tail exponent: %4.3f\n',alpha_right)

x_right = linspace(min(r_right),max(r_right),100);
y_right = alpha_right*(x_right/min(r_right)).^(-alpha_right-1)/min(r_right); % Power law distribution

[b_right,a_right] = histnorm(r_right,20);

subplot(1,2,1)
loglog(a_right,b_right,'ob','MarkerSize',8,'MarkerFaceColor','b')
hold on
loglog(x_right,y_right,'r','LineWidth',2)
set(gca,'FontSize',20)
title('Right tail')

%%% Right tail with bootstrap

alpha_right_bts = []; % Vector to collect bootstrap estimates for right tail exponent

for i = 1:Nbts
   
    r_bts = r_train(randperm(length(r_train))); % Random permutation of returns
    r_bts = r_bts(1:round(bts*length(r_bts))); % Bootstrapping bts% of returns 
    r_bts = sort(r_bts); % Sorting bootstrapped returns
    r_right_bts = r_bts(round((1-p)*length(r_bts)):end); % Selecting top p% of returns
    
    N_bts = length(r_right_bts); % Number of bootstrapped returns
    
    alpha_right_bts = [alpha_right_bts; N_bts/sum(log(r_right_bts/min(r_right_bts)))];

end

alpha_right_bts = sort(alpha_right_bts); % Sorting bootstrap estimates for right tail exponent

fprintf('Right tail interval at %3.2f CL: [%4.3f; %4.3f] \n',alpha,alpha_right_bts(round(0.5*(1-alpha)*Nbts)),alpha_right_bts(round(0.5*(1+alpha)*Nbts)))
fprintf('\n')

%%% Left tail

r_left = r_train(1:round(p*length(r_train))); % Selecting bottom p% of returns
r_left = abs(r_left); % Converting negative returns to positive numbers

N = length(r_left); % Number of returns selected as left tail
alpha_left = N/sum(log(r_left/min(r_left))); % Maximum-likelihood estimate for left tail exponent

fprintf('Left tail exponent: %4.3f\n',alpha_left)

x_left = linspace(min(r_left),max(r_left),100);
y_left = alpha_left*(x_left/min(r_left)).^(-alpha_left-1)/min(r_left); % Power law distribution

[b_left,a_left] = histnorm(r_left,20);

subplot(1,2,2)
loglog(a_left,b_left,'ob','MarkerSize',8,'MarkerFaceColor','b')
hold on
loglog(x_left,y_left,'r','LineWidth',2)
set(gca,'FontSize',20)
title('Left tail')

%%% Left tail with bootstrap

alpha_left_bts = []; % Vector to collect bootstrap estimates for left tail exponent

for i = 1:Nbts
   
    r_bts = r_train(randperm(length(r_train))); % Random permutation of returns
    r_bts = r_bts(1:round(bts*length(r_bts))); % Bootstrapping bts% of returns 
    r_bts = sort(r_bts); % Sorting bootstrapped returns
    r_left_bts = r_bts(1:round(p*length(r_bts))); % Selecting bottom p% of returns
    r_left_bts = abs(r_left_bts); % Converting returns to positive
    
    N_bts = length(r_left_bts); % Number of bootstrapped returns
    
    alpha_left_bts = [alpha_left_bts; N_bts/sum(log(r_left_bts/min(r_left_bts)))];

end

alpha_left_bts = sort(alpha_left_bts); % Sorting bootstrap estimates for right tail exponent

fprintf('Left tail interval at %3.2f CL: [%4.3f; %4.3f] \n',alpha,alpha_left_bts(round(0.5*(1-alpha)*Nbts)),alpha_left_bts(round(0.5*(1+alpha)*Nbts)))
fprintf('\n')

figure(11); clf; hold on;
subplot(2, 1, 1)
histogram(alpha_right_bts, 100, 'Normalization', 'pdf')
title('right tail index')
subplot(2, 1, 2)
histogram(alpha_left_bts, 100, 'Normalization', 'pdf')
title('left tail index')


%% VaR/CVaR computation using forward validation %%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set ==> Distribution ==> Method ==> Threshold ==> Var, CVaR

% Split r into k sets %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

datasets = cell(1, sets);

for i = 0:(sets - 1)
    
    set_length = round( length(r)/sets, -1 );
    r_i = r(i*set_length+1:(i+1)*set_length);
    datasets{i+1} = r_i; 
    
end

distributions = {'normal', 'tLocationScale'};
meth = {'MLE'};
p_var = cell(1, length(sets));
var_mat = cell(1, length(sets));
var_emp = cell(1, length(sets));
kernel = cell(1, length(sets));

for i = 1:sets
    
    % 1) Compute empirical VAR/ CVAR in each set
    
    ev = [];
    ec = [];
    for j = 1:length(thresh)
        var_i = quantile(datasets{i}, thresh(j), 1);
        cvar_i = mean(datasets{i}(datasets{i}<var_i));
        ev = [ev var_i];
        ec = [ec cvar_i];
    end
    var_mat{i} = 100*[ev ec; ev ec];
    
    % 2) Expand training set
    
    r_i = [];
    for dt = 1:i
        r_i = [r_i; datasets{dt}];
    end
    
    % 3) Historical VaR/CVaR forecast (benchmark)
    
    evf = [];
    ecf = [];
    
    for j = 1:length(thresh)
        varf_i = quantile(r_i, thresh(j), 1);
        cvarf_i = mean(r_i(r_i<varf_i));
        evf = [evf varf_i];
        ecf = [ecf cvarf_i];
    end
    
    var_emp{i} = 100*[evf ecf; evf ecf];
    
    % 4) 70-30% split of r_i into train-validate sets
    r_train = r_i(1:round(train*length(r_i)));
    r_v = r_i(round(train*length(r_i))+1:end);
    
    % 5) KDE forecast
    h = logspace(-3,-1,100); 
    L = []; 
    
    for w = 1:length(h)
        % Vector of Gaussian kernel values calculated in each point of validation set
        p = gaussian_mix(r_v,r_train,h(w));  
        aux = sum(log(p));  
        L = [L; aux];
    end
    
    % Identifying optimal value (i.e., argmax of log-likelihood)
    h_opt = h(find(L == max(L)));
    
    % Building kernel density
    x = linspace(1.5*min(r), 1.5*max(r), 1000);
    y = gaussian_mix(x,r_train,h_opt);
    cdf_y = cumsum(y)/sum(y);
    
    kv = [];
    % Computing kernel VaR
    for j = 1:length(thresh)
        
        % sample from observed returns
        u = datasample(r_train, 2000);
        
        % sample from normal distribution with standard deviation =
        % optimal bandwith
        s = normrnd(u, h_opt);
        
        %compute relevant quantile of sampled returns
        kvar = quantile(s, thresh(j), 1);
        
        kv = [kv kvar];
    end
    
    % Computing kernel CVaR
    for j = 1:length(thresh)
        kcvar = x(find(round(cdf_y, 2) == round(thresh(j), 2), 1, 'first'));
        kv = [kv kcvar];
    end
    
    kernel{i} = 100*[kv; kv];
    
    % 5) Parametric forecast
    
    var_d = cell(1, length(distributions));
    
    for d = 1:length(distributions)
        var_m = [];
        for m = 1:length(meth)
            method = meth{m};
            switch method
                case 'MLE'
                    params = fitdist(r_train, distributions{d});
                case 'MOM'
                    % todo
                    params = fitdist(r_train, distributions{d});
            end
            % create pdf object
            switch distributions{d}
                case 'normal'
                    mu = params.mu;
                    sigma = params.sigma;
                    pd = makedist('normal', 'mu', mu, 'sigma', sigma);
                case 'tLocationScale'
                    mu = params.mu;
                    sigma = params.sigma;
                    nu = params.nu
                    pd = makedist('tLocationScale', 'mu', mu, 'sigma', sigma, 'nu', nu);
            end
            v = [];
            % compute VaR squared error
            for y = 1:length(thresh)
                var = icdf(pd, thresh(y));
                v = [v var];
            end
            % compute CVaR squared error
            for y = 1:length(thresh)
                x = linspace(1.5*min(r), icdf(pd, thresh(y)), 1000);
                cvar = -trapz(pdf(pd, x))*abs((icdf(pd, thresh(y)) - 1.5*min(r)))/1000;
                v = [v cvar];
            end
            var_m = [var_m v];
        end
        var_d{d} = var_m;
    end
    p_var{i} = 100*[var_d{1}; var_d{2}];
end

% Measure MSE

mse_emp  = zeros(size(p_var{1}));
mse_kernel = zeros(size(p_var{1}));
mse_p = zeros(size(p_var{1}));

for i = 1:sets-1

    mse_emp = mse_emp + (var_emp{i} - var_mat{i+1}).^2;
    mse_kernel = mse_kernel + (kernel{i} - var_mat{i+1}).^2;
    mse_p = mse_p + (p_var{i} - var_mat{i+1}).^2;
    
end
mse_emp = mse_emp/(sets-1);
mse_kernel = mse_kernel/(sets-1);
mse_p = mse_p/(sets-1);

mse = array2table( [mse_emp(1,:); mse_kernel(1, :); mse_p] );
mse = addvars(mse, {'HS', 'KDE', 'Normal', 'Student'}', 'Before', 'Var1');
mse.Properties.VariableNames = {'Method', 'VaR_15', 'VaR_10', 'VaR_5', 'CVaR_15', 'CVaR_10', 'CVaR_5'};
mse
