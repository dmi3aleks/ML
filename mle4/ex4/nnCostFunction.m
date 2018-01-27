function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% amount of examples
m = size(X, 1);

%--------------------------------------------
% Calculate cost function
J = 0;
S = 0;
%for each example
for i = 1:m
  %for each label
  for k = 1:num_labels

    logical_y = (y == k);
    hyp = hypothesysCalc(X(i,:), Theta1, Theta2);

    S = S + ((logical_y(i))*log(hyp(k,1)) + (1-logical_y(i))*log(1 - ((hyp(k,1)))));

  end
end

regularization = (lambda/(2*m))*(sum((Theta1.^2)(:)) - sum((Theta1.^2)(:,1)) + sum((Theta2.^2)(:)) - sum((Theta2.^2)(:,1)));

J = -S/m + regularization;
         
%--------------------------------------------
% Calculate unregularized gradients of the cost function (dJ/dTheta)
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%for each example
for i = 1:m

  a1 = X(i,:)';
  a1 = [1; a1];
  z2 = Theta1 * a1;
  a2 = sigmoid_generic(z2);;
  a2 = [1; a2];
  z3 = Theta2 * a2;
  a3 = (sigmoid_generic(z3));
  y_l = zeros(size(a3));
  y_l(y(i)) = 1;
  d3 = a3 - y_l;
  d2 = Theta2(:,2:end)' * d3 .* sigmoidGradient(z2);
;
  Theta1_grad = Theta1_grad + d2*(a1');
  Theta2_grad = Theta2_grad + d3*(a2');
end

Theta1_grad = Theta1_grad ./ m;
Theta2_grad = Theta2_grad ./ m;

% Regularization
Theta1_grad += (lambda/m)*[zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_grad += (lambda/m)*[zeros(size(Theta2,1),1) Theta2(:,2:end)];

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
%--------------------------------------------

end
