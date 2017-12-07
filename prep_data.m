function [X, Y, included_features] = prep_data(X_input, Y_input, cutoff)
    X = [X_input sum(X_input, 2)];
%     X = [double(X(:, 1:10000) > 1) X(:, 10001)];
   
    Y = Y_input;
    num_features = size(X, 2);
    [baseline, ~] = hist(Y_input, 5);
    baseline = baseline / sum(baseline);
    included_features = true(1, 1e4); 
    
    for i = 1:num_features
        labels = Y_input(X(:, i) > 0);
        if (~isempty(labels))
            [label_dist, ~] = hist(labels, 5);
            label_dist = label_dist / sum(label_dist);
            diff = abs(label_dist - baseline);
            if (max(diff) < cutoff)
                included_features(i) = false;
            end
        else
            included_features(i) = false;
        end
    end
    X = X(:, included_features);
end

