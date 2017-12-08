function Y_hat = predict_naivebayes(X_train, Y_train, X_test)
    cost_matrix = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
    M = fitcnb(X_train, Y_train);
    [~, prob_estimates, ~] = predict(M, X_test);
    prob_estimates2 = prob_estimates;
    for i = 1:5
        prob_estimates2(:, M.ClassNames(i)) = prob_estimates(:, i);
    end
    costs = cost_matrix * prob_estimates2';
    [~, Y_hat] = min(costs);
    Y_hat = Y_hat';
end

