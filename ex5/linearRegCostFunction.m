function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% <SATISH>
% Add a 1 column at the front
% Looks like X already has first column set as 1. dough.
% X = [ones(m,1) X];
hx = X * theta;
% Cost without regularization
cost_notreg = sum((hx - y) .^ 2) / (2 * m);
% Regularization param
reg = sum(theta([2:end]) .^ 2) * (lambda / (2 * m));

J = cost_notreg + reg;

grad = (X' * (hx - y)) * (1 / m) + [0;theta(2:end)] * (lambda / m);


% =========================================================================

grad = grad(:);

end
