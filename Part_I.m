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
val = .5; % fraction of training dataset for validation
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

%  Fit distributions to log-returns

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
normal = [mu sigma]

%%% Student T distribution

%%% MLE
params_t = fitdist(r_train, 'tLocationScale');
mu_t = params_t.mu;
sigma_t = params_t.sigma;
nu_t = params_t.nu;
pd_t = makedist('tLocationScale', 'mu', mu_t, 'sigma', sigma_t, 'nu', nu_t);
student_t = [mu_t sigma_t nu_t]

%%% Stable distribution
params_s = fitdist(r_train, 'Stable')
alpha_s  = params_s.alpha;
beta = params_s.beta;
gam = params_s.gam;
delta = params_s.delta;
stable = [alpha beta gam delta]

% 3) fit distributions with bootstrap method to test parameter
% robustness

bts = 0.8; % Fraction of data to be retained in each bootstrap sample
Nbts = 1; % Number of bootstrap samples
alpha = 0.9; % Significance level

%%% Normal distribution (Note: MoM parameter estimators are the same as the
%%% for MLE).

mu_mle = zeros(1, Nbts); % Vector to collect bootstrap estimates for MLE mu
sigma_mle = zeros(1, Nbts); % Vector to collect bootstrap estimates for MLE s.d.

for i = 1:Nbts
    r_bts = r_train(randperm(length(r_train))); % Random permutation of returns
    r_bts = r_bts(1:round(bts*length(r_bts))); % Bootstrapping bts% of returns 
    params_mle = fitdist(r_bts, 'Normal'); % MLE fit of bootstrap sample
    mu_mle(i) = params_mle.mu;
    sigma_mle(i) = params_mle.sigma;
end

% sort estimates
mu_mle = sort(mu_mle);
sigma_mle = sort(sigma_mle);

% compute confidence interval for each parameter estimator
fprintf('MLE mu CI at %3.2f CL: [%4.3f; %4.3f] \n',alpha, mu_mle(round(0.5*(1-alpha)*Nbts)), mu_mle(round(0.5*(1+alpha)*Nbts)));
fprintf('MLE sigma CI at %3.2f CL: [%4.3f; %4.3f] \n',alpha, sigma_mle(round(0.5*(1-alpha)*Nbts)), sigma_mle(round(0.5*(1+alpha)*Nbts)));

%%% Student-t distribution 

mu_mle_t = zeros(1, Nbts); % Vector to collect bootstrap estimates for MLE mu
sigma_mle_t = zeros(1, Nbts); % Vector to collect bootstrap estimates for MLE s.d.
nu_mle_t = zeros(1, Nbts); % Vector to collect bootstrap estimates for MLE nu

for i = 1:Nbts
    r_bts = r_train(randperm(length(r_train))); % Random permutation of returns
    r_bts = r_bts(1:round(bts*length(r_bts))); % Bootstrapping bts% of returns 
    params_mle_t = fitdist(r_bts, 'tLocationScale'); % MLE fit of bootstrap sample
    mu_mle_t(i) = params_mle_t.mu;
    sigma_mle_t(i) = params_mle_t.sigma;
    nu_mle_t(i) = params_mle_t.nu;
end

% sort estimates
mu_mle_t = sort(mu_mle_t); 
sigma_mle_t = sort(sigma_mle_t); 
nu_mle_t = sort(nu_mle_t);

% Compute CI for parameter estimators
fprintf('MLE mu CI at %3.2f CL: [%4.3f; %4.3f] \n',alpha, mu_mle_t(round(0.5*(1-alpha)*Nbts)), mu_mle_t(round(0.5*(1+alpha)*Nbts)));
fprintf('MLE sigma CI at %3.2f CL: [%4.3f; %4.3f] \n',alpha, sigma_mle_t(round(0.5*(1-alpha)*Nbts)), sigma_mle_t(round(0.5*(1+alpha)*Nbts)));
fprintf('MLE nu CI at %3.2f CL: [%4.3f; %4.3f] \n',alpha, nu_mle_t(round(0.5*(1-alpha)*Nbts)), nu_mle_t(round(0.5*(1+alpha)*Nbts)));

%%% Stable distribution
% alpha_mle = zeros(1, Nbts);
% beta_mle = zeros(1, Nbts);
% gam_mle = zeros(1, Nbts);
% delta_mle = zeros(1, Nbts);
% 
% for i = 1:Nbts
%     r_bts = r_train(randperm(length(r_train))); % Random permutation of returns
%     r_bts = r_bts(1:round(bts*length(r_bts))); % Bootstrapping bts% of returns 
%     params_mle_s = fitdist(r_bts, 'Stable'); % MLE fit of bootstrap sample
%     alpha_mle(i) = params_mle_s.alpha;
%     beta_mle(i) = params_mle_s.beta;
%     gam_mle(i) = params_mle_s.gam;
%     delta_mle(i) = params_mle_s.delta;
% end
% 
% % sort estimates
% % Compute CI for parameter estimators
% fprintf('MLE alpha CI at %3.2f CL: [%4.3f; %4.3f] \n',alpha, alpha_mle(round(0.5*(1-alpha)*Nbts)), alpha_mle(round(0.5*(1+alpha)*Nbts)));
% fprintf('MLE beta CI at %3.2f CL: [%4.3f; %4.3f] \n',alpha, beta_mle(round(0.5*(1-alpha)*Nbts)), beta_mle(round(0.5*(1+alpha)*Nbts)));
% fprintf('MLE gamma CI at %3.2f CL: [%4.3f; %4.3f] \n',alpha, gam_mle(round(0.5*(1-alpha)*Nbts)), gam_mle(round(0.5*(1+alpha)*Nbts)));
% fprintf('MLE delta CI at %3.2f CL: [%4.3f; %4.3f] \n',alpha, delta_mle(round(0.5*(1-alpha)*Nbts)), delta_mle(round(0.5*(1+alpha)*Nbts)));

%% 4) Test fit on validation set using QQ-Plot

figure(5); clf; hold on;
subplot(3, 1, 1)
qqplot(r_val, pd_normal)
title('QQ Plot of Sample Data versus Normal Distribution')
xlim([-.25 .25])
xlabel('Quantiles of Normal Distribution')
ylabel('Quantiles of Sample')
subplot(3, 1, 2)
qqplot(r_val, pd_t_mom)
title('QQ Plot of Sample Data versus Student-t Distribution (M.O.M)')
xlabel('Quantiles of Student-t Distribution (M.O.M.)')
xlim([-.25 .25])
ylabel('Quantiles of Sample')
subplot(3, 1, 3)
qqplot(r_val, pd_t_mle)
title('QQ Plot of Sample Data versus Student-t Distribution (M.L.E.)')
xlabel('Quantiles of Student-t Distribution (M.L.E.)')
xlim([-.25 .25])
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
plot(a_right,b_right,'ob','MarkerSize',8,'MarkerFaceColor','b')
hold on
plot(x_right,y_right,'r','LineWidth',2)
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
plot(a_left,b_left,'ob','MarkerSize',8,'MarkerFaceColor','b')
hold on
plot(x_left,y_left,'r','LineWidth',2)
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
