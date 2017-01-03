%clear; close all; clc;

%X_train = [0 0; 0 1; 1 0; 1 1];
%Y_train = [0; 1; 1; 0];

fprintf('Import data ... \n');
train_data = load('train79.mat');
X_train = train_data.d79;
Y_train = [ones(1,1000) -ones(1,1000)]';
test_data = load('test79.mat');
X_test = test_data.d79;
Y_test = [ones(1,1000) -ones(1,1000)]';

[X_train, Y_train] = PreProcess(X_train, Y_train);
[X_test, Y_test] = PreProcess(X_test, Y_test);

% Parameters
eta = 1;
Iter = 5000;
hidden_layer_size = 1024;
error_Threshold = 100;

fprintf(['Begin training ... \n']);
%[Weight1_train, Weight2_train] = MLP_Train(hidden_layer_size, X_train, Y_train, eta, Iter, error_Threshold);

fprintf(['Begin predicting ... \n']);
[acc_rate_train, confusion_matrix_train] = MLP_Predict(Weight1_train, Weight2_train, X_train, Y_train)
[acc_rate_test, confusion_matrix_test] = MLP_Predict(Weight1_train, Weight2_train, X_test, Y_test)