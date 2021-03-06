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
