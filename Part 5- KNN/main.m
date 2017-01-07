clear; close all; clc;
%% ======================== KNN =================================
fprintf('Import data ... \n');
% load training set
train_data = load('train79.mat');
X_train = train_data.d79;
Y_train = [7*ones(1000,1); 9*ones(1000,1)];
% load testing set
test_data = load('test79.mat');
X_test = test_data.d79;
Y_test = Y_train;

[X_train, Y_train] = PreProcess(X_train, Y_train);
[X_test, Y_test] = PreProcess(X_test, Y_test);

k = [1 3 5 10 20 200];
for i = 1:length(k)
	fprintf(['K = ' num2str(k(i)) '.\n']);
	tic;
	[acc_correct(i)] = KNN(k(i), X_train, Y_train, X_test, Y_test)
	time_knn(i) = toc
end
