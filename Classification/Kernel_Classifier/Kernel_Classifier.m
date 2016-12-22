function [acc_rate_train, acc_rate_test] = Kernel_Classifier(X_train, Y_train, X_test, Y_test, lambda, sigma)
% Implementing a Least Squares Kernel classifier with a Gaussian Kernel
% argument test_on_testing can be assigned any value for testing on testing set
% training
K = Get_Gaussian_Kernal_Matrix(X_train, X_train, sigma);
Alpha = (K + lambda*eye(size(K)))\Y_train;

% testing on training set
P_train = Alpha' * K;
P_train(find(P_train>0))=1;
P_train(find(P_train<0))=-1;
acc_rate_train = 1 - sum(Y_train'~= P_train)/size(P_train,2);

% testing on testing set
K = Get_Gaussian_Kernal_Matrix(X_train, X_test, sigma);
P_test = Alpha' * K;
P_test(find(P_test>0))=1;
P_test(find(P_test<0))=-1;
acc_rate_test = 1 - sum(Y_test'~= P_test)/size(P_test,2);

end