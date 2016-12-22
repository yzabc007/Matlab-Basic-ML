function [acc_rate_train, acc_rate_test] = LeastSquares_Classifier(X_train, Y_train, X_test, Y_test)
% Use a least squares linear classifier
% training parameters using normal equations
w = pinv(X_train'*X_train)*(X_train'*Y_train);

% testing on training data
P_train = X_train*w;
P_train(find(P_train>0))=1;
P_train(find(P_train<0))=-1;
acc_rate_train = 1 - sum(Y_train~= P_train)/size(P_train,1);

% testing on testing data
P_test = X_test*w;
P_test(find(P_test>0))=1;
P_test(find(P_test<0))=-1;
acc_rate_test = 1 - sum(Y_test~= P_test)/size(P_test,1);
end