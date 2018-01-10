function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

thetaTail = theta
thetaTail(1) = 0
J = 1/m * (sum((-y)'*log(sigmoid(X*theta)) - (1-y)'*log(1 - sigmoid(X*theta)))) + (lambda/(2*m))*sum(thetaTail .^ 2);

grad = sum((sigmoid(X*theta) - y).*X)*1/m;
grad = grad + thetaTail' .* (lambda/m);

end
