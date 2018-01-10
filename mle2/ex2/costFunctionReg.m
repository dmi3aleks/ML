function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

thetaTail = theta
thetaTail(1) = 0
J = 1/m * (sum((-y)'*log(sigmoid(X*theta)) - (1-y)'*log(1 - sigmoid(X*theta)))) + (lambda/(2*m))*sum(thetaTail .^ 2);

grad(1) = sum((sigmoid(X*theta) - y).*X(:,1))*1/m;

for j = 2:size(X)(2)
  grad(j) = sum((sigmoid(X*theta) - y).*X(:,j))*1/m + theta(j)*lambda/m;
end

end
