load('train.mat');
load('validation.mat');
load('vocabulary.mat');

[N, ~] = size(X_train_bag);

indices = crossvalind('Kfold', N, 10);

cost_matrix = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];

cost = 0;
[X, Y] = prep_data(X_train_bag, Y_train);
% X = X_train_bag;
% Y = Y_train;
for i = 1:10
    i
    X_train = X(indices ~= i,:);
    train_labels = Y(indices ~= i);
    X_test = X(indices == i, :);
    test_labels = Y(indices == i);
    [n, ~] = size(X_test);
    M = fitcknn(X_train, train_labels);
    [Y_hat, ~, ~] = predict(M, X_test);
    
    err = performance_measure(Y_hat, test_labels);
    cost = cost + err;
end

cost = cost / 10;

