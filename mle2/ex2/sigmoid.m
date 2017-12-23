function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

if size(z) == 1 
    g = 1/(1 + exp(-z));
elseif
    g = arrayfun(@sigmoid, z);
end

end
