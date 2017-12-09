function mean_score = cross_val(X_input, Y_input, K, predict_func)
    N = size(X_input, 1);
    indices = crossvalind('Kfold', N, K); 
    
    cost = 0;
    for i = 1:K
        i
        X_train = X_input(indices ~= i,:);
        Y_train = Y_input(indices ~= i);
        X_test = X_input(indices == i, :);
        Y_test = Y_input(indices == i);
        
        Y_hat = predict_func(X_train, Y_train, X_test);
       
        err = performance_measure(Y_hat, Y_test);
        cost = cost + err;
    end
    
    mean_score = cost / K;
end

