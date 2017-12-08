function Y_hat = predict_rf(X_train, Y_train, X_test, num_trees, num_p)
    cost_matrix = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
    [X_train, X_test] = pca_getpc(full(X_train), full(X_test), 1000);
    M = TreeBagger(num_trees, X_train, Y_train, 'Cost', cost_matrix, 'NumPredictorsToSample', num_p);
    Y_hat = predict(M, X_test);
    Y_hat = str2double(Y_hat);
end