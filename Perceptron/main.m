clear; close all; clc;

%% Classification
fprintf('Import data ... \n');
train_data = load('train79.mat');
X_train = train_data.d79;
Y_train = [ones(1,1000) -ones(1,1000)]';
test_data = load('test79.mat');
X_test = test_data.d79;
Y_test = [ones(1,1000) -ones(1,1000)]';

% Iteration threshold
Iter = 3000;
% Learning rate
eta = 1;
%% =====================================================================================
%% 'GradDes' 'IsNormEq' 'OutType' 'linear' 'class' 'logistic'
%% ==================== Perceptron for Multi Classification ============================
weights_logistic = Perceptron_Train(X_train, Y_train, 'GradDes', Iter, eta, 'OutType', 'logistic');
acc_rate_train_logistic = Perceptron_Predict(weights_logistic, X_test, Y_test, 'OutType', 'logistic')
%% =====================================================================================
%weights_GradDes = Perceptron_Train(X_train, Y_train, 'GradDes', Iter, eta, 'OutType', 'class');
%weights_NormEq = Perceptron_Train(X_train, Y_train, 'IsNormEq', 'OutType', 'class');
%acc_rate_train_GradDes = Perceptron_Predict(weights_GradDes, X_train, Y_train, 'OutType', 'class')
%acc_rate_test_GradDes = Perceptron_Predict(weights_GradDes, X_test, Y_test, 'OutType', 'class')
%acc_rate_train_NormEq = Perceptron_Predict(weights_NormEq, X_train, Y_train, 'OutType', 'class')
%acc_rate_test_NormEq = Perceptron_Predict(weights_NormEq, X_test, Y_test, 'OutType', 'class')

%% ==================== Linear Regression ==============================================
fprintf('Import data ... \n');
data = load('ex1data1.txt');
X = data(:, 1:size(data,2)-1);
y = data(:, size(data,2));

Iter = 10000;
eta = 0.01;

%[weights1] = Perceptron_Train(X, y, 'GradDes', Iter, eta, 'OutType', 'linear')
%[err_sum_square1] = Perceptron_Predict(weights1, X, y, 'OutType', 'linear')

%[weights2] = Perceptron_Train(X, y, 'IsNormEq', 'OutType', 'linear')
%[err_sum_square2] = Perceptron_Predict(weights2, X, y, 'OutType', 'linear')