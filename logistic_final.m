load('train.mat');
load('validation.mat');
load('vocabulary.mat');

cost_matrix = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];

cs = .2:.02:.3;
inx = 1;
scores = zeros(1, length(cs));

% [X, Y] = prep_data(X_train_bag, Y_train, 0);
X = double(X_train_bag > 0);
Y = Y_train;
for c = cs
    predict_func = @(X, Y, Xt)(predict_logistic_final(X, Y, Xt, c)); % partially apply function with chosen value of 'c'
    score = cross_val(X, Y, 10, predict_func);
    scores(inx) = score;
    inx = inx+1;
end

function Y_hat = predict_logistic_final(X_train, Y_train, X_test, c)
%     cost_matrix = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
    params = sprintf('-s 7 -c %i', c); % use L2 loss
    M = train(Y_train, X_train, params);
    n = size(X_test, 1);
    [Y_hat, ~, ~] = predict(ones(n,1), X_test, M, ['-b 1']);
end
