load('train.mat');
load('validation.mat');
load('vocabulary.mat');

cost_matrix = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];

cs = .5:.2:2;
cs = [.03];
inx = 1;
scores = zeros(1, length(cs));

% [X, Y] = prep_data(X_train_bag, Y_train, 0);
%X = double(X_train_bag > 0);
X = remove_uncommon_words(X_train_bag);
Y = Y_train;
for c = cs
    inx / length(cs)
    predict_func = @(X, Y, Xt)(predict_logistic_final(X, Y, Xt, c)); % partially apply function with chosen value of 'c'
    score = cross_val(X, Y, 10, predict_func);
    scores(inx) = score
    inx = inx+1;
end

function Y_hat = predict_logistic_final(X_train, Y_train, X_test, c)
%     cost_matrix = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
    params = sprintf('-s 6 -c %i', c); % use L2 loss
    M = train(Y_train, X_train, params);
    n = size(X_test, 1);
    [Y_hat, ~, ~] = predict(ones(n,1), X_test, M, ['-b 1']);
end

function cleanX = remove_uncommon_words(X)
[n, ~] = size(X);
word_count = sum((X > 1), 1);
occurs_infequently = word_count < 2;
%occurs_infequently = word_count > (n / 2); 
cleanX = X(:, ~occurs_infequently);
end
