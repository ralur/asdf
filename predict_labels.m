function [Y_hat] = predict_labels(X_test_bag, test_raw)
    load('train.mat');
    load('validation.mat');
   
    X_train = X_train_bag;
    X_test = X_test_bag;

    M = train(Y_train, X_train, '-s 7 -c .207'); 
    [n, ~] = size(X_test_bag);
    [Y_hat, ~, ~] = predict(ones(n,1), X_test, M, ['-b 1']);
end