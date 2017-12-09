load('train.mat');
load('validation.mat');
load('vocabulary.mat');

cost_matrix = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];

X = awgn(X_train_bag, 100); % perturb features to ensure that no class has zero variance (annoying bug otherwise)
X = max(X, 0); % ensure no negative counts

cutoffs = 1:5:50;
scores = zeros(1, length(cutoffs));
inx = 1;
for cutoff = cutoffs
    [X, Y] = prep_data(X, Y_train, 0, cutoff);

    predict_func = @(X, Y, Xt)(predict_naivebayes(X, Y, Xt)); % no params for naive bayes (default distributions, priors etc are fine)
    scores(inx) = cross_val(X, Y, 4, predict_func);
    inx
    inx = inx+1;
end