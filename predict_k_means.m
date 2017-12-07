load('train.mat');
load('validation.mat');
load('vocabulary.mat');
 
[N, ~] = size(X_train_bag);
K = 300;
indices = crossvalind('Kfold', N, 10);
 
cost_matrix = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
 
cost = 0;
 
[X, Y] = prep_data(X_train_bag, Y_train, .1);
% X = X_train_bag;
% Y = Y_train;
for i = 1:10
    X_train = X(indices ~= i,:);
    train_labels = Y(indices ~= i);
    X_test = X(indices == i, :);
    test_labels = Y(indices == i);
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
 
    
    err = performance_measure(Y_hat, test_labels);
    cost = cost + err;
end
 
cost = cost / 10
 
% M = train(train_labels, X_train, '-s 7 -c .207');
 


