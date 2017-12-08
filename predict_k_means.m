function Y_hat = predict_k_means(X_train, train_labels, X_test, K)
if nargin == 3
    K = 300;
end
cost_matrix = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
 
[X_train, train_labels, keep] = prep_data(X_train, train_labels, .1);
X_test = X_test(:,keep);

[n, dim] = size(X_test);

[ids, centroids] = kmeans(full(X_train), K);
centroids_to_ids = zeros(K,1);
for j = 1:K
    in_centroids = train_labels(ids == j);
    most_labeled = 0;
    num_labels = 0;
    for l = 1:K
        num_l = sum(in_centroids == l);
        if (num_l > num_labels)
            most_labeled = l;
            num_labels = num_l;
        end
    end
    centroids_to_ids(j) = most_labeled;
end

Y_hat = zeros(n,1);
for j = 1:n
    distances = centroids * X_test(j,:)';
    [~, closest_centroid] = min(distances);
    Y_hat(j) = centroids_to_ids(closest_centroid);
end
    
end


