load('train.mat');
load('validation.mat');
load('vocabulary.mat');

cutoff = .2;

[X_train, Y_train, included_features] = prep_data(X_train_bag, Y_train, cutoff);
X_test = X_validation_bag(:, included_features);

c_lr = .31;
num_trees_rf = 1000;
K_km = 100;
K_knn = 100;
num_p_components_knn = 1000;


Y_lr = predict_logistic(X_train, Y_train, X_test, c_lr); 
Y_rf = predict_rf(X_train, Y_train, X_test, num_trees_rf);
Y_nb = predict_naivebayes(X_train, Y_train, X_test);

cutoff = .9;
[X_train, Y_train, included_features] = prep_data(X_train_bag, Y_train, cutoff);
X_test = X_validation_bag(:, included_features);

Y_km = predict_k_means(X_train, Y_train, X_test, K_km);
Y_knn = predict_knn(X_train, Y_train, X_test, K_knn, num_p_components_knn);

