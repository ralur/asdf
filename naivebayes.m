load('train.mat');
load('validation.mat');
load('vocabulary.mat');

[N, ~] = size(X_train_bag);

indices = crossvalind('Kfold', N, 10);
cutoff = .5;

cost_matrix = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];

[X, Y] = prep_data(X_train_bag, Y_train, cutoff);
X = awgn(X, 100);
X = max(X, 0);
i = 1;
X_train = X(indices ~= i,:);
train_labels = Y(indices ~= i);
X_test = X(indices == i, :);
test_labels = Y(indices == i);
[n, ~] = size(X_test);
M = fitcnb(X_train, train_labels);
[Y_hat, ~, ~] = predict(M, X_test);

cost = performance_measure(Y_hat, test_labels);