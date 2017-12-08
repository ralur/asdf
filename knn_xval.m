load('train.mat');
load('validation.mat');
load('vocabulary.mat');


Ks = [1 100 200 500];
num_p_components = [20 400 2000 4000];
scores = zeros(4,5);
%runs the knn function with a series of parameters to find the best
for i = 1:4
    K = Ks(i);
    for j = 1:5
        num_p_component = num_p_components(j);
        knn_f = @(X, Y, P)(predict_knn(X, Y, P, K, num_p_component));
        scores(i, j) = cross_val(X_train_bag, Y_train, 10, knn_f);
    end
end