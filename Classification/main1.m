%% Main function for testing different kind of classifiers

% addpath for all subfolders
addpath(genpath('LeastSquares_Classifier'))
addpath(genpath('Bayes_Classifier'))
addpath(genpath('kNN_Classifier'))
addpath(genpath('SVM_Classifier'))
addpath(genpath('Kernel_Classifier'))

%% ==================== Part 1: Initialization ====================
clear ; close all ; clc

% Import training and testing data
fprintf('Import data ... \n');
train_data = load('train79.mat');
X_train = train_data.d79;
Y_train = [ones(1,1000) -ones(1,1000)]';
test_data = load('test79.mat');
X_test = test_data.d79;
Y_test = [ones(1,1000) -ones(1,1000)]';

% all classifiers have similar format -
%% ==================== Part 1: LeastSquares_Classifier ====================
%fprintf('Running LeastSquares_Classifier ... \n');
%[acc_rate_train, acc_rate_test] = LeastSquares_Classifier(X_train, Y_train, X_test, Y_test)

%% ==================== Part 2: Bayes_Classifier ====================
fprintf('Running Bayes_Classifier ... \n');
[acc_rate_train, acc_rate_test] = Bayes_Classifier(X_train, Y_train, X_test, Y_test)

%% ==================== Part 3: kNN_Classifier ====================
%fprintf('Running kNN_Classifier ... \n');
%k = 3;
%[acc_rate_train, acc_rate_test] = kNN_Classifier(X_train, Y_train, X_test, Y_test, k)

%% ==================== Part 4: SVM_Classifier ====================
%fprintf('Running SVM_Classifier ... \n');
%[acc_rate_train, acc_rate_test] = SVM_Classifier(X_train, Y_train, X_test, Y_test)

%% ==================== Part 5: Kernel_Classifier ====================
%fprintf('Running Kernel_Classifier ... \n');
% k-fold cross validation for parameter selection
%{
k=10;
n = size(X_train,1);
shuffle_samp = (randperm(n));
sigma = [100, 1000, 10000, 100000];
lambda = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000];
for i = 1:size(sigma,2)
	for j = 1:size(lambda,2)
		average_acc_rate(i,j) = kFold_Cross_Validation(k, X_train, Y_train, lambda(j), sigma(i),shuffle_samp);
	end
end
%}
% select appropriate parameter
%lambda = 0.001;
%sigma = 1000;
%[acc_rate_train, acc_rate_test] = Kernel_Classifier(X_train, Y_train, X_test, Y_test, lambda, sigma)