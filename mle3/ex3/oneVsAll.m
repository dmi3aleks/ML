function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Size
m = size(X, 1);
n = size(X, 2);

all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

initial_theta = zeros(n + 1, 1);

options = optimset('GradObj', 'on', 'MaxIter', 50);

for k = 1:num_labels

    logical_y = (y == k);
    disp('logical_y')
    disp(size(logical_y,1))
    disp(size(logical_y,2))
    disp('X')
    disp(size(X,1))
    disp(size(X,2))
    disp('initial_theta')
    disp(size(initial_theta,1))
    disp(size(initial_theta,2))
    [theta] = fmincg (@(t)(lrCostFunction(t, X, (y == k), lambda)), initial_theta, options);
    all_theta(k,:) = theta';
end

end
