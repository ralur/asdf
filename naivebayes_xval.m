load('train.mat');
load('validation.mat');
load('vocabulary.mat');

cost_matrix = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
cutoff = 0;
X = [X_train_bag, sum(X_train_bag, 2)];
Y = Y_train;
X = awgn(X, 100); % perturb features to ensure that no class has zero variance (annoying bug otherwise)
X = max(X, 0); % ensure no negative counts
predict_func = @(X, Y, Xt)(predict_naivebayes(X, Y, Xt)); % no params for naive bayes (default distributions, priors etc are fine)
score = cross_val(X, Y, 10, predict_func);

