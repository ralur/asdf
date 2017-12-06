X_input = X_train_bag;
Y_input = Y_train;

X = double(X_input > 0);
Y = Y_input;
num_features = size(X, 2);
[baseline, ~] = hist(Y_input, 5);
baseline = baseline / sum(baseline);
included_features = true(1, 1e4); 
label_dists = zeros(1e4, 5);
for i = 1:num_features
    labels = Y_input(X(:, i) > 0);
    if (~isempty(labels))
        [label_dist, ~] = hist(labels, 5);
        label_dist = label_dist / sum(label_dist);
        label_dists(i, :) = label_dist;
        diff = abs(label_dist - baseline);
        if (max(diff) < .1)
            included_features(i) = false;
        end
    else
        included_features(i) = false;
    end
end

hist(max(label_dists' - baseline'))

% load('train.mat');
% load('validation.mat');
% load('vocabulary.mat');
% 
% N = size(X_train_bag, 1);
% 
% K = 10;
% indices = crossvalind('Kfold', N, K);
% 
% cost_matrix = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
% cutoffs = 0:.01:.2;
% cs = .3:.01:.5;
% 
% scores = zeros(length(cutoffs), length(cs));
% inx1 = 1;
% inx2 = 1;
% [X, Y] = prep_data(X_train_bag, Y_train, .12);
% 
% cost = 0;
% for i = 1:K
%     X_train = X(indices ~= i,:);
%     train_labels = Y(indices ~= i);
%     X_test = X(indices == i, :);
%     test_labels = Y(indices == i);
%     [n, ~] = size(X_test);
%     params = sprintf('-s 7 -c %i', .41);
%     M = train(train_labels, X_train, params); 
%     [Ys, ~, prob_estimates] = predict(ones(n,1), X_test, M, ['-b 1']);
% %     costs = cost_matrix * prob_estimates';
% %     [~, Y_hat] = min(costs);
% %     Y_hat = Y_hat';
%     Y_hat = Ys;
% 
%     err = performance_measure(Y_hat, test_labels);
%     cost = cost + err;
% end
% cost = cost / K;
%       
% 
% % M = train(train_labels, X_train, '-s 7 -c .207'); 

hist(sum(X_train_bag, 1));

