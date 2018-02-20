function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%
[m, n] = size(X);

% compute a covariance matrix
Sigma = 1/m * (X' * X);

% compute eigenvecors and eigenvalues
[U, S, V] = svd(Sigma);
end
