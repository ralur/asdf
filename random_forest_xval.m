load('train.mat');
load('validation.mat');
load('vocabulary.mat');

cost_matrix = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];


num_trees = 1000;
inx = 1;
inx2 = 1;
cutoffs = 0;
scores = zeros(length(cutoffs), length(num_trees));

for cutoff = cutoffs
    [X, Y] = prep_data(X_train_bag, Y_train, cutoff);
    
    X = full(X);
    for num_tree = num_trees
        predict_func = @(X, Y, Xt)(predict_rf(X, Y, Xt, num_tree));
        score = cross_val(X, Y, 10, predict_func);
        scores(inx, inx2) = score;
        inx2 = inx2+1;
    end
    inx = inx+1;
    inx2 = 1;
end

function Y_hat = predict_rf(X_train, Y_train, X_test, num_trees)
    [X_train, X_test] = pca_getpc(X_train, X_test, 1000);
    M = TreeBagger(num_trees, X_train, Y_train);
    Y_hat = predict(M, X_test);
    Y_hat = str2double(Y_hat);
end

