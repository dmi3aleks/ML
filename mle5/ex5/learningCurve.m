function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular,
%       it returns two vectors of the same length - error_train and
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).

m = size(X, 1);

for i = 1:m
  Xtrain = X(1:i,:);
  ytrain = y(1:i,:);
  theta = trainLinearReg(Xtrain, ytrain, lambda);
  [error_train(i), grad] = linearRegCostFunction(Xtrain, ytrain, theta, 0);
  [error_val(i), grad] = linearRegCostFunction(Xval, yval, theta, 0);
end


end
