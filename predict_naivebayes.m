function Y_hat = predict_naivebayes(X_train, Y_train, X_test)
    M = fitcnb(X_train, Y_train);
    [Y_hat, ~, ~] = predict(M, X_test);
end

