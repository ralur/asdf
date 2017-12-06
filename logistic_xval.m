load('train.mat');
load('validation.mat');
load('vocabulary.mat');

cs = .1:.05:1.01;
[X, Y] = prep_data(X_train_bag, Y_train, .1);
scores = zeros(1, length(cs));
inx = 1;
for c = cs
    predict_func = @(X, Y, Xt)(predict_logistic(X, Y, Xt, c));
    score = cross_val(X, Y, 10, predict_func);
    scores(inx) = score;
    inx = inx+1;
end

function Y_hat = predict_logistic(X_train, Y_train, X_test, c)
    params = sprintf('-s 7 -c %i', c);
    M = train(Y_train, X_train, params);
    n = size(X_test, 1);
    [Y_hat, ~, ~] = predict(ones(n,1), X_test, M, ['-b 1']);
end

