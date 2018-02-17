function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

K = size(centroids, 1);
n = size(X,1);
idx = zeros(n, 1);

for i = 1:n
  minDistance = 10e8;
  minCentroid = 0;
  for j = 1:K
      dist = distance(X(i,:), centroids(j,:));
      if dist <= minDistance
        minDistance = dist;
        minCentroid = j;
      end
  end
  idx(i) = minCentroid;
end

end

function dist = distance(A, B)
  dist = sum((A - B) .^ 2);
end
