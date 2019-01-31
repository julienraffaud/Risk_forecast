
clear all;
close all;

% Inputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 'LTC' 'XLM' 'TRX' 'BNB' 'ZEC' 'LSK' 'ADA' 'NEO'
cryptos = {'EURUSD', 'LTC', 'TRX'};
tau = 24;
start_time = '02-Jan-2018 11:00:00';
end_time = '14-Jun-2018 17:00:00';
var = .05;
pdf = 'Normal';

% Import & format data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

TT = cell(1, length(cryptos));
TR = timerange(start_time,end_time);
for i = 1:length(cryptos)
    
    if (strcmp(cryptos{i}, 'EURUSD'))
        data = readtable(strcat(cryptos{i},'_merged.txt'));
        table = table2timetable(data);
        
    else
        data = importdata(strcat(cryptos{i},'_merged.txt'));
        posix = data.data(:,2);
        time = datetime(posix, 'ConvertFrom', 'Posixtime');
        price = data.data(:, 12);
        table = timetable(time, price);
        table = table(TR,:);
    end

    table.Properties.VariableNames = {cryptos{i}};
    if (i>1)
        table = [TT{i-1} table];
    end
    TT{i} = table;
end
data = TT{end};

% Compute log-returns for each crypto %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

r = log(data{2:end,1:end}./data{1:end-1,1:end});

% preliminary plot of price %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot price evolution
figure(1); clf;
semilogy(data.time, data{:, :})
title('Cryptocurrencies price evolution, January 2018 - June 2018')
ylabel(char(8364))
xlim(datetime(2018,[1 6],[2 14]))
legend(cryptos)

% Accumulate log-returns for different time horizons %%%%%%%%%%%%%%%%%%%%%%

aux = cell(1, length(cryptos));
for i = 1:length(cryptos)
    r_i = r(:, i);
    for t = 0:tau:length(r_i)-tau
        aux{i} = [aux{i}; sum(r_i(t+1:t+tau))];
    end
end
r = cell2mat(aux);

%% Plot log-returns
figure(2); clf;
subplot(2, 1, 1)
%plot(data.time(1:end-tau), r)
plot(r)
title('Cryptocurrencies log-returns, January 2018 - June 2018')
ylabel('log-return')
%xlim(datetime(2018,[1 6],[2 14]))
subplot(2, 1, 2)
%plot(data.time(1:end-tau), abs(r))
plot(r)
title('Cryptocurriencies absolute log-returns, January 2018 - June 2018')
ylabel('absolute log-return')
%xlim(datetime(2018,[1 6],[2 14]))

%% Autocorrelation plot
figure(3); clf; hold on;
for i = 1:length(cryptos)
    [c, lags] = xcorr(abs(r(:,i)), abs(r(:,i)));
    c = c(lags>0);
    lags = lags(lags>0);
    c = c/c(1);
    plot(lags(1:100), c(1:100))
    title(" Autocorrelation of cryptocurrency absolute log-returns")
    xlabel('Lags')
    ylabel('Correlation')
end
legend(cryptos)

%% Compute first four moments of each cryptocurrency's log returns %%%%%%%%%

N = length(r(:,1));
m = mean(r); % Compute mean and store value in variable
std = std(r, 0, 1);
sk = skewness(r);
kurt = kurtosis(r);
ex_kurt = kurt - 3;

stats = [m', std', sk', kurt' ex_kurt'];
stats = array2table(stats);
stats.Properties.VariableNames = {'Mean', 'Std', 'Skew', 'Kurtosis', 'Excess_K'};
stats = addvars(stats, cryptos', 'Before', 'Mean');
stats.Properties.VariableNames = {'Cryptocurrency','Mean', 'Std', 'Skew', 'Kurtosis', 'Excess_K'};
stats

%% Plot of empirical PDF vs. Gaussian %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:length(cryptos)
    mn = min(r(:,i));
    mx = max(r(:,i));
    x = linspace(mn, mx, 100);
    g = exp(-(x-m(i)).^2/(2*std(i)^2))/sqrt(2*pi*std(i)^2);
    NB = 100;
    figure(3+i)
    [b,a] = histnorm(r(:,i),NB); % Normalized histogram of returns with NB bins
    semilogy(a,b,'ob','MarkerSize',8,'MarkerFaceColor','b')
    hold on
    semilogy(x,g,'r','LineWidth',2)
    %set(gca,'FontSize',20)
    xlabel('log-return')
    ylim([0.03 10^2])
    title(cryptos(i))
end

%% Plot of Empirical CCDF vs Gaussian %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:length(cryptos)
    
    figure(3+length(cryptos)+i)
    
    r_i = r(:,i);
    
    pos_r = r_i(r_i>0);
    neg_r = abs(r_i(r_i<0));
    
    % positive
    xp = sort(pos_r); % Returns sorted in ascending order
    yp = 1:1:length(pos_r); 
    yp = 1 - yp/(length(pos_r)+1); % Calculating CCDF
    
    % negative
    xn = sort(neg_r); % Returns sorted in ascending order
    yn = 1:1:length(neg_r); 
    yn = 1 - yn/(length(neg_r)+1); % Calculating CCDF
    
    % normal
    c = (1 - erf((xp-m(i))/(std(i)*sqrt(2))));

    loglog(xp,yp,'o','MarkerSize', 1, 'MarkerEdgeColor','b')
    hold on

    loglog(xn,yn,'o', 'MarkerSize', 1, 'MarkerEdgeColor', 'r')
    loglog(xp, c, 'green', 'LineWidth', 1)
    ylim([1e-4 1])
    xlim([0 1])
    xlabel('log-return')
    ylabel('complementary cumulative distribution')
    title([cryptos(i), ' complementary cumulative log-return distribution'])
    legend({'positive' 'negative' 'normal'})
    
end

%% Empirical VaR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

vars = cell(1, length(cryptos));
for i = 1:length(cryptos)
    v = quantile(r(:,i), var, 1);
    vars{i} = v;
end
vars = cell2mat(vars');

cvars = cell(1, length(cryptos));
for i = 1:length(cryptos)
    r_i = r(:,i);
    v = quantile(r_i, var, 1);
    cv = mean(r_i(r_i<v));
    cvars{i} = cv;
end
cvars = cell2mat(cvars');
vartable = array2table([vars cvars]);
vartable = addvars(vartable, cryptos', 'Before', 'Var1');
vartable.Properties.VariableNames = {'Cryptocurrency', 'VaR', 'CVaR'};
vartable

% Parametric fit of PDF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:length(cryptos)
    test_cdf = fitdist(r(:,i),pdf);
    test_cdf
    [h, p] = kstest(r(:,i),test_cdf);
    figure(20+i)
    qqplot(r(:,i), fitdist(r(:,i),pdf))
    title(cryptos{i})
end
