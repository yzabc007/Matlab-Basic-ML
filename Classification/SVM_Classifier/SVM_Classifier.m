function [acc_rate_train, acc_rate_test] = SVM_Classifier(X_train, Y_train, X_test, Y_test)
% Use a linear support vector machine classifier
% traning
svmStruct = svmtrain(X_train,Y_train);

% testing on traing data
P_train = svmclassify(svmStruct,X_train);
acc_rate_train = 1 - sum(Y_train~= P_train)/size(P_train,1);

% testing on testing data
P_test = svmclassify(svmStruct,X_test);
acc_rate_test = 1 - sum(Y_test~= P_test)/size(P_test,1);
end