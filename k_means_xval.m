load('train.mat');
load('validation.mat');
load('vocabulary.mat');

Ks = [1 100 200 500 1000];
scores = zeros(1,5);
%runs the knn function with a series of parameters to find the best
for K = Ks
    K = Ks(i);
    knn_f = @(X, Y, P)(predict_k_means(X, Y, P, K));
    scores(i) = cross_val(X_train_bag, Y_train, 10, knn_f);
end