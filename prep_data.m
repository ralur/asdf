function [X, Y, included_features] = prep_data(X_input, Y_input, cutoff)
    num_features = size(X_input, 2);
    X = [X_input sum(X_input, 2)]; % add feature containing length of tweet
    X = [double(X(:, 1:num_features) > 0) X(:, num_features + 1)];
    
    Y = Y_input;
    [baseline, ~] = hist(Y_input, 5);
    baseline = baseline / sum(baseline); % calculate label distribution
    included_features = true(1, num_features); 
    
    for i = 1:num_features
        labels = Y_input(X(:, i) > 0);
        if (~isempty(labels))
            [label_dist, ~] = hist(labels, 5);
            label_dist = label_dist / sum(label_dist); % calculate label distribution conditioned on ith feature being present
            diff = abs(label_dist - baseline);
            if (max(diff) < cutoff)
                included_features(i) = false; % remove feature if the distribution doesn't differ materially from baseline
            end
        else
            included_features(i) = false;
        end
    end
    X = X(:, included_features);
end

