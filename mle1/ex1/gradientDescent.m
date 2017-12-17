function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % Instructions: Perform a single gradient step on the parameter vector theta. 
    newTheta = theta;
    newTheta(1) = newTheta(1) - (alpha/m)*sum((X*theta - y)' * X(:,1));
    newTheta(2) = newTheta(2) - (alpha/m)*sum((X*theta - y)' * X(:,2));
    theta = newTheta
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
