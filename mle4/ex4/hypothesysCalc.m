function H = hypothesysCalc(X, Theta1, Theta2)
%------------------------------------
% NOTE: hypothesys calculation
%------------------------------------
m = size(X, 1);
X = [ones(m, 1) X];

L2 = sigmoid_generic(X * Theta1');
L2 = [ones(size(L2,1), 1) L2];
L3 = sigmoid_generic(L2 * Theta2');
% NOTE: only evaluated hypothesys matters here
H = L3';
%disp('Size H:');
%disp(size(H));
%------------------------------------
end

%{
    S = S + ((y(1,k))'*log(hypothesysCalc(X(i,:), Theta1, Theta2)(1,k)) + ((1-y)(1,k))'*log(1 - hypothesysCalc(X(i,:), Theta1, Theta2)(1,k)));
%}
