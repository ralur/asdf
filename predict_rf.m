function Y_hat = predict_rf(X_train, Y_train, X_test, num_trees)
    [X_train, X_test] = pca_getpc(full(X_train), full(X_test), 1000);
    M = TreeBagger(num_trees, X_train, Y_train);
    Y_hat = predict(M, X_test);
    Y_hat = str2double(Y_hat);
end