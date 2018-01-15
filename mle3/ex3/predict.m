function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);
X = [ones(m, 1) X];

L2 = sigmoid_generic(X * Theta1');
L2 = [ones(size(L2,1), 1) L2];
L3 = sigmoid_generic(L2 * Theta2');
[v,p] = max(L3, [], 2);

% a less elegant and more intuitive alternative implementation:
%{
p = zeros(size(X, 1), 1);

for i = 1:size(X,1)
  L2 = sigmoid_generic(X(i,:) * Theta1');
  L2 = [ones(size(L2,1), 1) L2];
  %L2 is (1, 26)
  L3 = sigmoid_generic(L2 * Theta2');
  L3 = L3';
  %L2 is (10, 1)
  [v,p(i)] = max(L3);
end
%}

end
