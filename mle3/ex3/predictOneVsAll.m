function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier.

m = size(X, 1);
X = [ones(m, 1) X];

A = X * all_theta';
% A has dimensions: (amount_of_examples, amount_of_labels)
% A(i,j): probability proxy for input i belonging to type j.
[v,p] = max(A, [], 2);

% less elegant and more intuitive implementation:
%{
for i = 1:2
    A = X(i,:) * all_theta';
    % A is (1,10)
    disp(size(A));
    [max_val, p(i)] = max(A);
end
%}

end
