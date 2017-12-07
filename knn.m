load('train.mat');
load('validation.mat');
load('vocabulary.mat');

[N, ~] = size(X_train_bag);

indices = crossvalind('Kfold', N, 10);
K = 100;
num_p_components = 2000;
cost_matrix = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];

cost = 0;

[X, Y] = prep_data(X_train_bag, Y_train, .6);


for i = 1:10
    X_train = X(indices ~= i,:);
    train_labels = Y(indices ~= i);
    X_test = X(indices == i, :);
    test_labels = Y(indices == i);
    [n, ~] = size(X_test);
    
    [X_train, X_test] = pca_getpc(full(X_train), full(X_test), num_p_components);
    
%     M = fitcknn(X_train, train_labels, 'NumNeighbors',K);
%     Y_hat = predict(M, X_test);
    
    Y_hat = zeros(n,1);
    k_nearest_sets = knnsearch(X_train, X_test(), 'Distance', 'cityblock', 'K', K);
    for j = 1:n
        k_nearest = k_nearest_sets(j,:);
%         plabel = predict(M, X_test(j,:));
        
        k_nearest_lables = train_labels(k_nearest);
        number_labels = arrayfun(@(x)sum(k_nearest_lables == x), 1:5);
        
        [~, most_labeled] = max(cost_matrix * number_labels');
        Y_hat(j) = most_labeled;
    end
    err = performance_measure(Y_hat, test_labels);
    cost = cost + err;
end

cost = cost / 10;

function [score_train, score_test] = pca_getpc(X_train, X_test, numpc)
 
% input: original X for training and testing
% output: PCAed X for training and testing, number of PCs that you selected
 
cov_train = cov(X_train);
[coeff_train, latent] = pcacov(cov_train);
score_train = X_train * coeff_train;
score_train = score_train(:,1:numpc);
score_test = X_test * coeff_train; 
score_test = score_test(:,1:numpc);
end

