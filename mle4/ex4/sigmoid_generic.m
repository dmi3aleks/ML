function g = sigmoid_generic(z)
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

if size(z) == 1
    g = 1/(1 + exp(-z));
elseif
    g = arrayfun(@sigmoid_generic, z);
end

end
