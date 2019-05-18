function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% <SATISH>
C_try = [0.01;0.03;0.1;0.3;1;3;10;30];
sigma_try = [0.01;0.03;0.1;0.3;1;3;10;30];
bestError = 1;
best_C = 1;
best_sigma = 0.3;
for C_itr = 1:size(C_try,1),
  for sigma_itr = 1:size(sigma_try,1),
    C = C_try(C_itr);
    sigma = sigma_try(sigma_itr);
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    if (error < bestError),
      bestError = error;
      best_C = C;
      best_sigma = sigma;
    end
  end
end

C = best_C;
sigma = best_sigma;
    
% </SATISH>





% =========================================================================

end
