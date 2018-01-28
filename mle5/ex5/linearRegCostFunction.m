function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

m = length(y);

% cost funciton
J = (1/(2*m))*(sum(((X * theta) .- y).^2)) + (lambda/(2*m))*(sum(theta .^ 2) - theta(1)^2);

% gradient vector
grad = (sum(((X * theta) .- y).*X))/m .+ theta' .* (lambda/m);
% note: not normalizing the bias unit
grad(1) -= (lambda * theta(1))/m;

end
