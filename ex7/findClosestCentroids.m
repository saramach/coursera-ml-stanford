function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
% <SATISH>
disp(centroids);
bestCentroid = ones(size(X,1), 1);
for centroidIdx = 2:K,
  % disp(sprintf("Centroid idx: %d", centroidIdx));
  for inputIdx = 1:size(X,1),
    best = X(inputIdx, :) - centroids(bestCentroid(inputIdx), :);
    current = X(inputIdx, :) - centroids(centroidIdx, :);
    % disp(sprintf("best: %f, current: %f", best * best', current * current'));
    if ((current * current ') < (best * best')),
      % Found a better centroid
      bestCentroid(inputIdx) = centroidIdx;
    end;
  end;
end;

% disp(bestCentroid);

idx = bestCentroid;

% </SATISH>






% =============================================================

end

