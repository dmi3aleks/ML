function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
if size(z) == 1
    g = sigmoid(z)*(1 - sigmoid(z));
elseif
    g = arrayfun(@sigmoidGradient, z);
end

end
