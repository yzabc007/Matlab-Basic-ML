clear; close all; clc;

%% =====================================================================================
%% Patameters for Perceptron_Train and Perceptron_Predict
%% 'GradDes' Iter eta
%% 'IsNormEq' 
%% 'OutType' 'linear' or 'binary' or 'logistic'
%% =====================================================================================

%% ==================== Perceptron =====================================================
fprintf('Import data for Perceptron ... \n');
train_data = load('train79.mat');
X_train = train_data.d79;
Y_train = [ones(1,1000) -ones(1,1000)]';
test_data = load('test79.mat');
X_test = test_data.d79;
Y_test = [ones(1,1000) -ones(1,1000)]';
X_train = FeatureNorm(X_train);
X_test = FeatureNorm(X_test);

Iter = 1000;
eta = 1;
weights_GradDes = Perceptron_Train(X_train, Y_train, 'GradDes', Iter, eta, 'OutType', 'binary');
acc_rate_train_GradDes = Perceptron_Predict(weights_GradDes, X_train, Y_train, 'OutType', 'binary')
acc_rate_test_GradDes = Perceptron_Predict(weights_GradDes, X_test, Y_test, 'OutType', 'binary')

weights_NormEq = Perceptron_Train(X_train, Y_train, 'IsNormEq', 'OutType', 'binary');
acc_rate_train_NormEq = Perceptron_Predict(weights_NormEq, X_train, Y_train, 'OutType', 'binary')
acc_rate_test_NormEq = Perceptron_Predict(weights_NormEq, X_test, Y_test, 'OutType', 'binary')

%% ==================== Perceptron for Logistic Regression =============================
fprintf('Import data for Logistic Regression ... \n');
data = load('ex2data1.txt');
X_train = data(:, 1:size(data,2)-1);
Y_train = data(:, size(data, 2)); % lable should be 1 and 0

% Iteration threshold
Iter = 1000000;
% Learning rate (Choose carefully)
eta = 0.0001;
weights_logistic = Perceptron_Train(X_train, Y_train, 'GradDes', Iter, eta, 'OutType', 'logistic');
acc_rate_train_logistic = Perceptron_Predict(weights_logistic, X_train, Y_train, 'OutType', 'logistic')

%% ==================== Perceptron for Linear Regression ================================
fprintf('Import data for Linear Regression... \n');
data = load('ex1data1.txt');
X = data(:, 1:size(data,2)-1);
y = data(:, size(data,2));

Iter = 10000;
eta = 0.01;
[weights1] = Perceptron_Train(X, y, 'GradDes', Iter, eta, 'OutType', 'linear');
[err_sum_square1] = Perceptron_Predict(weights1, X, y, 'OutType', 'linear')

[weights2] = Perceptron_Train(X, y, 'IsNormEq', 'OutType', 'linear');
[err_sum_square2] = Perceptron_Predict(weights2, X, y, 'OutType', 'linear')
