function [Y_hat] = predict_labels(X_test_bag, test_raw)
    load('train.mat');
    load('validation.mat');
    
    [X, Y, included_features] = prep_data(X_train_bag, Y_train);
    X_train = X;
    Y_train = Y;
    X_test = X_test_bag(:, included_features);
        
    M = train(Y_train, X_train, '-s 7 -c .31'); 
    [n, ~] = size(X_test_bag);
    [Y_hat, ~, ~] = predict(ones(n,1), X_test, M, ['-b 1']);
end