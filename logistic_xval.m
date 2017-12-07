load('train.mat');
load('validation.mat');
load('vocabulary.mat');

cost_matrix = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];

cs = .5:.05:3;
inx = 1;
inx2 = 1;
cutoffs = 0;
scores = zeros(length(cutoffs), length(cs));

for cutoff = cutoffs
    [X, Y] = prep_data(X_train_bag, Y_train, cutoff);
    for c = cs
        predict_func = @(X, Y, Xt)(predict_logistic(X, Y, Xt, c));
        score = cross_val(X, Y, 10, predict_func);
        scores(inx, inx2) = score;
        inx2 = inx2+1;
    end
    inx = inx+1;
    inx2 = 1;
end
% joy (1), sadness (2), surprise (3), anger (4) and fear (5)
function Y_hat = predict_logistic(X_train, Y_train, X_test, c)
    cost_matrix = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
    params = sprintf('-s 7 -c %i', c);
    M = train(Y_train, X_train, params);
    n = size(X_test, 1);
    [~, ~, prob_estimates] = predict(ones(n,1), X_test, M, ['-b 1']);
    prob_estimates2 = prob_estimates;
    for i = 1:5
        prob_estimates2(:, M.Label(i)) = prob_estimates(:, i);
    end
    costs = cost_matrix * prob_estimates2';
    [~, Y_hat] = min(costs);
    Y_hat = Y_hat';
    
end

