function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier.

m = size(X, 1);
num_labels = size(all_theta, 1)

p = zeros(size(X, 1), 1);
X = [ones(m, 1) X];

e = size(X,1)

for i = 1:e
  max_val = -32;
  max_ind = 1;
  for j = 1:num_labels
    A = X(i,:) * (all_theta(j,:))';
    if A > max_val
      max_val = A;
      max_ind = j;
    end
  end
  p(i) = max_ind;
end

% =========================================================================


end
