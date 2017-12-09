function Y_hat = predict_logistic(X_train, Y_train, X_test, c)
    cost_matrix = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
    params = sprintf('-s 7 -c %i', c); % use L2 loss 
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