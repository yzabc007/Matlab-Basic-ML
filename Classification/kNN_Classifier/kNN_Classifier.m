function [acc_rate_train, acc_rate_test] = kNN_Classifier(X_train, Y_train, X_test, Y_test, k)

% testing on training data
P_train = knnclassify(X_train, X_train, Y_train, k);
P_train(find(P_train>0))=1;
P_train(find(P_train<0))=-1;
acc_rate_train = 1 - sum(Y_train~= P_train)/size(P_train,1);

% testing on testing data
P_test = knnclassify(X_test, X_train, Y_train, k);
P_test(find(P_test>0))=1;
P_test(find(P_test<0))=-1;
acc_rate_test = 1 - sum(Y_test~= P_test)/size(P_test,1);

end