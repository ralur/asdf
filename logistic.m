load('train.mat');
load('validation.mat');
load('vocabulary.mat');

N = size(X_train_bag, 1);

K = 10;
indices = crossvalind('Kfold', N, K);

cost_matrix = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
cutoffs = .1;
cs = .2:.01:5;

scores = zeros(length(cutoffs), length(cs));
inx1 = 1;
inx2 = 1;
for cutoff = cutoffs
    [X, Y] = prep_data(X_train_bag, Y_train, cutoff);
    for c = cs
        cost = 0;
        indices = crossvalind('Kfold', N, K);
        for i = 1:K
            X_train = X(indices ~= i,:);
            train_labels = Y(indices ~= i);
            X_test = X(indices == i, :);
            test_labels = Y(indices == i);
            [n, ~] = size(X_test);
            params = sprintf('-s 7 -c %i', c);
            M = train(train_labels, X_train, params); 
            [Ys, ~, prob_estimates] = predict(ones(n,1), X_test, M, ['-b 1']);
            costs = cost_matrix * prob_estimates';
            [~, Y_hat] = min(costs);
            Y_hat = Y_hat';
            %Y_hat = Ys;

            err = performance_measure(Y_hat, test_labels);
            cost = cost + err;
        end
        scores(inx1, inx2) = cost / K;
        inx2 = inx2+1;
    end
    inx1 = inx1+1;
    inx2 = 1;
end
% M = train(train_labels, X_train, '-s 7 -c .207'); 

