function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma = [0.01 0.03 0.1 0.3 1 3 10 30];


% Loop over the different C's and sigmas
scores = zeros(size(C,2),size(sigma,2));
for i = 1:size(C,2)
    for j = 1:size(sigma,2)
        model= svmTrain(X, y, C(i), @(x1, x2) gaussianKernel(x1, x2, sigma(j)));
        predictions = svmPredict(model,Xval);
        scores(i,j) = mean(double(predictions~=yval));
    end
end

%Find the best combination
best = min(min(scores));
[k,l]= find(scores==best);
C = C(k);
sigma= sigma(l);


end
