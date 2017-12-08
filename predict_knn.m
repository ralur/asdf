%Runs K-nearest neighborhs on the supplied test data and returns the
%expected labels. Treates the k nearest neighbors as fractiona likelihoods
%and uses cost matrix to predict least expected cost label
%X_train - Data to train on
%Labels - of the training data
%X_test - Data to predict for
%K - Number of nearest neighbors to use
%num_p_components - number of principal components to use
function Y_hat = predict_knn(X_train, train_labels, X_test, K, num_p_components)
    if nargin == 3
        K = 100;
        num_p_components = 2000;
    end

    cost_matrix = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];

    [X_train, X_test] = pca_getpc(full(X_train), full(X_test), num_p_components);

    [n, ~] = size(X_test);
    Y_hat = zeros(n,1);
    k_nearest_sets = knnsearch(X_train, X_test(), 'Distance', 'cityblock', 'K', K);
    for j = 1:n
        k_nearest = k_nearest_sets(j,:);
        k_nearest_lables = train_labels(k_nearest);
        number_labels = arrayfun(@(x)sum(k_nearest_lables == x), 1:5);

        [~, most_labeled] = max(cost_matrix * number_labels');
        Y_hat(j) = most_labeled;
    end
end


