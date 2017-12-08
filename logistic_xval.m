load('train.mat');
load('validation.mat');
load('vocabulary.mat');

cost_matrix = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];

cs = 1:.05:3;
inx = 1;
inx2 = 1;
cutoffs = .1;
scores = zeros(length(cutoffs), length(cs));

for cutoff = cutoffs
    [X, Y] = prep_data(X_train_bag, Y_train, cutoff);
    for c = cs
        predict_func = @(X, Y, Xt)(predict_logistic(X, Y, Xt, c)); % partially apply function with chosen value of 'c'
        score = cross_val(X, Y, 10, predict_func);
        scores(inx, inx2) = score;
        inx2 = inx2+1;
    end
    inx = inx+1;
    inx2 = 1;
end