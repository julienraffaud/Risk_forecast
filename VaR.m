%% Statistical characterisation of high-frequency time series
% Parametric & non-parametric VaR/CVaR forecasting

clear all;
close all;

% Inputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

crypto = 'TRX';
tau = 1;
start_time = '02-Jan-2018 11:00:00';
end_time = '14-Jun-2018 17:00:00';
var = .05;

% Import & format data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

TR = timerange(start_time,end_time);
data = importdata(strcat(crypto,'_merged.txt'));
posix = data.data(:,2);
time = datetime(posix, 'ConvertFrom', 'Posixtime');
price = data.data(:, 12);
table = timetable(time, price);
table = table(TR,:);
table.Properties.VariableNames = {crypto};

% Compute log-returns %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

r = log(table{2:end,1:end}./table{1:end-1,1:end});

% Plot price evolution %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(1); clf; hold on
plot(table.time, table{:,1})
title([crypto ' price evolution, January - June 2018'])
ylabel(['Price (' char(8364) ')'])
xlim(datetime(2018,[1 6],[2 14]))
legend(crypto)
set(gca,'FontSize', 15)

% Accumulate log-returns for different time horizon %%%%%%%%%%%%%%%%%%%%%%%

aux = [];
for t = 0:tau:length(r)-tau
    aux = [aux; sum(r(t+1:t+tau))];  
end
r = aux;

% Compute the first four moments %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = length(r(:,1));
m = mean(r); % Compute mean and store value in variable
std = std(r, 0, 1); % Compute standard deviation
sk = skewness(r); % Compute skew
kurt = kurtosis(r); % Compute kurtosis
ex_kurt = kurt - 3; % Compute excess kurtosis
stats = [m', std', sk', kurt' ex_kurt'];
stats = array2table(stats);
stats.Properties.VariableNames = {'Mean', 'Std', 'Skew', 'Kurtosis', 'Excess_K'};
stats = addvars(stats, {crypto}, 'Before', 'Mean');
stats.Properties.VariableNames = {'Cryptocurrency','Mean', 'Std', 'Skew', 'Kurtosis', 'Excess_K'};

% Plot of empirical PDF vs Gaussian PDF with same mean/std %%%%%%%%%%%%%%%%

minr = min(r);
maxr = max(r);
x = linspace(minr, maxr, 100);
g = exp(-(x-m).^2/(2*std^2))/sqrt(2*pi*std^2);
NB = 50;
[b,a] = histnorm(r, NB); % Normalized histogram of returns with NB bins
figure(2);
semilogy(a, b,'ob','MarkerSize',6,'MarkerFaceColor','b')
hold on;
semilogy(x,g,'r','LineWidth',2)
set(gca,'FontSize', 15)
xlim(1.2*[minr maxr])
ylim([0.03 100])
title(crypto)
legend('Empirical PDF', 'Normal PDF')
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
c = (1 - erf((xp-m)/(std*sqrt(2))));

% plot
figure(3)
loglog(xp,yp,'o','MarkerSize', 2, 'MarkerEdgeColor','b')
hold on
loglog(xn,yn,'o', 'MarkerSize', 2, 'MarkerEdgeColor', 'r')
loglog(xp, c, 'green', 'LineWidth', 2)
set(gca,'FontSize',15)
ylim([1e-4 1])
xlim([min(abs(r)) 1])
xlabel('log-return')
ylabel('complementary cumulative distribution')
title([crypto, ' complementary cumulative log-return distribution'])
legend({'positive' 'negative' 'normal'})

% Split r into train-validate-test sets %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%r = r(randperm(length(r)));

% fraction of dataset of each set
train = 0.5;
validate = 0.25;
test = 0.25;

% splitting set
r_train = r(1:round(train*length(r)));
r_v = r(length(r_train)+1:length(r_train)+1+round(validate*length(r)));
r_test = r(length(r_train) + length(r_v) + 1:end);

% Fitting PDFs on training set using bootstrap method %%%%%%%%%%%%%%%%%%%%%

dists = {'normal', 'tLocationScale', 'Stable'};

bts = 0.8; % Fraction of data to be retained in each bootstrap sample
Nbts = 50; % Number of bootstrap samples
alpha = 0.9; % Significance level
bins = 30;

% normal estimates
nmus = cell(1, Nbts);
nsigmas = cell(1, Nbts);

% student-t estimates
smus = cell(1, Nbts);
ssigmas = cell(1, Nbts);
nus = cell(1, Nbts);

% stable estimates
salphas = cell(1, Nbts);
sbetas = cell(1, Nbts);
sgams = cell(1, Nbts);
sdeltas = cell(1, Nbts);

for i = 1:length(dists)
    
    dist = dists{i};
    
    for i = 1:Nbts

        r_bts = r_train(randperm(length(r_train))); % Random permutation of returns
        r_bts = r_bts(1:round(bts*length(r_bts))); % Bootstrapping bts% of returns 
        
        params = fitdist(r_bts, dist);

        switch dist

            case 'tLocationScale'
                smus{i} = params.mu;
                nus{i} = params.nu;
                ssigmas{i} = params.sigma;

            case 'normal'
                nmus{i} = params.mu;
                nsigmas{i} = params.sigma;
                
            case 'Stable'
                salphas{i} = params.alpha;
                sbetas{i} = params.beta;
                sgams{i} = params.gam;
                sdeltas{i} = params.delta;
        end

    end
end

% sorting normal
nmus = sort(cell2mat(nmus));
nsigmas = sort(cell2mat(nsigmas));

% sorting student
smus = sort(cell2mat(smus));
ssigmas = sort(cell2mat(ssigmas));
nus = sort(cell2mat(nus));

% sorting stable
salphas = sort(cell2mat(salphas));
sbetas = sort(cell2mat(sbetas));
sgams = sort(cell2mat(sgams));
sdeltas = sort(cell2mat(sdeltas));

% Plotting estimates of parameters of normal distribution
figure(4); clf; hold on;
subplot(2, 1, 1)
histogram(nmus, bins, 'Normalization', 'pdf')
xlabel('mu')
hold on;
subplot(2, 1, 2)
histogram(nsigmas, bins, 'Normalization', 'pdf')
xlabel('sigma')

% Plotting estimates of parameters of student-t distribution
figure(5); clf; hold on;
subplot(3, 1, 1)
histogram(smus, bins, 'Normalization', 'pdf')
xlabel('mu')
subplot(3, 1, 2)
histogram(ssigmas, bins, 'Normalization', 'pdf')
xlabel('sigma')
subplot(3, 1, 3)
histogram(nus, bins, 'Normalization', 'pdf')
xlabel('nu')

% Plotting estimates of parameters of stable distribution
figure(6); clf; hold on;
subplot(4, 1, 1)
histogram(salphas, bins, 'Normalization', 'pdf')
xlabel('alpha')
subplot(4, 1, 2)
histogram(sbetas, bins, 'Normalization', 'pdf')
xlabel('beta')
subplot(4, 1, 3)
histogram(sgams, bins, 'Normalization', 'pdf')
xlabel('gamma')
subplot(4, 1, 4)
histogram(sdeltas, bins, 'Normalization', 'pdf')

% Assessing performance on validation set %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% normal dist parameters
mu = mean(nmus);
sigma = mean(nsigmas);
pdn = makedist('Normal','mu',mu,'sigma',sigma);

% student t dist parameters
smu = mean(smus);
snu = mean(nus);
ssigma = mean(ssigmas);
pds = makedist('tLocationScale','mu',smu,'sigma', ssigma, 'nu', snu);

% stable dist parameters
salpha = mean(salphas);
sbeta = mean(sbetas);
sgam = mean(sgams);
sdelta = mean(sdeltas);
pdstable = makedist('Stable', 'alpha', salpha, 'beta', sbeta, 'gam', sgam, 'delta', sdelta);

% Plot pdf
figure(7); hold on;
x = linspace(min(r_train), max(r_train), 100);
yn = pdf(pdn, x);
ys = pdf(pds, x);
yst = pdf(pdstable, x);
hold on;
semilogy(x,yn, 'r')
hold on;
semilogy(x,ys, 'b')
hold on;
semilogy(x, yst, 'black')
hold on;
histogram(r_v, 100, 'Normalization', 'pdf')

% Kolmogorov-Smirnov test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[hn, pn] = kstest(r_v, pdn);
[hs, ps] = kstest(r_v, pds);
[hst, pst] = kstest(r_v, pdstable);
result_ks = array2table([[hn, hs, hst]', [pn, ps, pst]']);
result_ks.Properties.VariableNames = {'Hypothesis', 'p_value'};
result_ks = addvars(result_ks, dists', 'Before', 'Hypothesis');
result_ks.Properties.VariableNames = {'Model', 'Hypothesis', 'p_value'};
result_ks

% Quantile-Quantile plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(8); clf; hold on;
subplot(3, 1, 1)
qqplot(r_v, pdn)
subplot(3, 1, 2)
qqplot(r_v, pds)
subplot(3, 1, 3)
qqplot(r_v, pdstable)

% Test chosen model prediction of VaR & CVaR of test data %%%%%%%%%%%%%%%%%

var_test = quantile(r_test, var, 1)
forecast = icdf(pdstable, var)
