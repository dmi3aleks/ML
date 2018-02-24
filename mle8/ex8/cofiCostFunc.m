function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

J = 0;
for i = 1:num_movies
  for j = 1:num_users
    if R(i,j) == 1
      J += (X(i,:) * Theta(j,:)' - Y(i,j)).^2;
    end
  end
end
J = J/2;

regularization  = (lambda/2) * (sum(sum(Theta .^2)) + sum(sum(X .^2)));
J += regularization;

%        X_grad - num_movies x num_features matrix, containing the
%                 partial derivatives w.r.t. to each element of X

X_grad = zeros(size(X));
for i = 1:num_movies
  for j = 1:num_users
    if R(i,j) == 1
      X_grad(i,:) += (X(i,:) * Theta(j,:)' - Y(i,j)) * Theta(j,:);
    end
  end
  X_grad(i,:) += lambda * X(i,:);
end

%        Theta_grad - num_users x num_features matrix, containing the
%                     partial derivatives w.r.t. to each element of Theta

Theta_grad = zeros(size(Theta));
for j = 1:num_users
  for i = 1:num_movies
    if R(i,j) == 1
      Theta_grad(j,:) += (X(i,:) * Theta(j,:)' - Y(i,j)) * X(i,:);
    end
  end
  Theta_grad(j,:) += lambda * Theta(j,:);
end

grad = [X_grad(:); Theta_grad(:)];

end
