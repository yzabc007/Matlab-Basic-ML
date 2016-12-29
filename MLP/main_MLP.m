clear; close all; clc;

%X_train = [0 0; 0 1; 1 0; 1 1];
%Y_train = [0; 1; 1; 0];

fprintf('Import data ... \n');
train_data = load('train79.mat');
X_train = train_data.d79;
Y_train = [ones(1,1000) -ones(1,1000)]';
test_data = load('test79.mat');
X_test = test_data.d79;
Y_test = [ones(1,1000) -ones(1,1000)]';
% shuffle
samples_shuffle = randperm(size(X_train,1));
X_train = X_train(samples_shuffle,:);
Y_train = Y_train(samples_shuffle,:);
% normalize
X_train = (X_train - repmat(mean(X_train,1), size(X_train,1), 1));
Y_train = Y_train - mean(Y_train);
X_test = X_test - repmat(mean(X_test,1), size(X_test,1), 1);
Y_test = Y_test - mean(Y_test);

%X_train_var = var(X_train, 1);
%X_train_var(find(X_train_var==0)) = 1;
%X_train = (X_train - repmat(mean(X_train,1), size(X_train,1), 1)) ./ repmat(X_train_var, size(X_train,1), 1);
%Y_train = (Y_train - mean(Y_train)) ./ var(Y_train);

% Parameters
eta = 1;
Iter = 5000;
hidden_layer_size = 1024;
error_Threshold = 100;

fprintf(['Begin training ... \n']);
[Weight1_train, Weight2_train] = MLP_Train(hidden_layer_size, X_train, Y_train, eta, Iter, error_Threshold);
fprintf(['Begin predicting ... \n']);
[acc_rate_train, confusion_matrix_train] = MLP_Predict(Weight1_train, Weight2_train, X_train, Y_train)
[acc_rate_test, confusion_matrix_test] = MLP_Predict(Weight1_train, Weight2_train, X_test, Y_test)