function [score_train, score_test] = pca_getpc(X_train, X_test, numpc)
    cov_train = cov(X_train);
    [coeff_train, ~] = pcacov(cov_train);
    score_train = X_train * coeff_train;
    score_train = score_train(:,1:numpc);
    score_test = X_test * coeff_train; 
    score_test = score_test(:,1:numpc);
end