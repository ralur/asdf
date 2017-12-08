load('train.mat');
load('validation.mat');
load('vocabulary.mat');

cost_matrix = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];


num_trees = 1000;
inx = 1;
inx2 = 1;
num_predictors = 100;
cutoff = 0;

scores = zeros(length(num_predictors), length(num_trees));

for np = num_predictors
    [X, Y] = prep_data(X_train_bag, Y_train, cutoff);
    X = full(X);
    inx
    for nt = num_trees
        predict_func = @(X, Y, Xt)(predict_rf(X, Y, Xt, nt, np));
        score = cross_val(X, Y, 2, predict_func);
        scores(inx, inx2) = score;
        inx2 = inx2+1;
    end
    inx = inx+1;
    inx2 = 1;
end

