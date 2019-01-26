clear all;
close all; 

% Inputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 'LTC' 'XLM' 'TRX' 'BNB' 'ZEC' 'LSK' 'ADA' 'NEO'
cryptos = {'LTC' 'XLM' 'TRX' 'BNB' 'ZEC' 'LSK' 'ADA' 'NEO'};
tau = 1;
start_time = '02-Jan-2018 11:00:00';
end_time = '14-Jun-2018 17:00:00';

% Import & format data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

TT = cell(1, length(cryptos));
TR = timerange(start_time,end_time);
for i = 1:length(cryptos)
    data = importdata(strcat(cryptos{i},'_merged.txt'));
    posix = data.data(:,2);
    time = datetime(posix, 'ConvertFrom', 'Posixtime');
    price = data.data(:, 12);
    table = timetable(time, price);
    table = table(TR,:);
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

 ret = cell(1, length(cryptos));
 for i = 1:length(cryptos)
     aux = []; 
     for t = 0:tau:length(r(:,i))-tau
         aux = [aux; sum(r(t+1:t+tau,i))];
     end
     if (i>1)
         aux = [aux ret{i-1}];
     end
    ret{i} = aux;
 end
 ret(end)

% Plot log-returns
figure(2); clf;
subplot(2, 1, 1)
plot(data.time(1:end-1), r)
title('Cryptocurrencies log-returns, January 2018 - June 2018')
ylabel('log-return')
xlim(datetime(2018,[1 6],[2 14]))
subplot(2, 1, 2)
plot(data.time(1:end-1), abs(r))
title('Cryptocurriencies absolute log-returns, January 2018 - June 2018')
ylabel('absolute log-return')
xlim(datetime(2018,[1 6],[2 14]))

% Autocorrelation plot
figure(3); clf; hold on;
for i = 1:length(cryptos)
    [c, lags] = xcorr(abs(r(:,i)), abs(r(:,i)));
    c = c(lags>0);
    lags = lags(lags>0);
    c = c/c(1);
    plot(lags(1:300), c(1:300))
    title(" Autocorrelation of cryptocurrency absolute log-returns")
    xlabel('Lags')
    ylabel('Correlation')
end
legend(cryptos)

% Compute first four moments of each cryptocurrency's log returns %%%%%%%%%

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
