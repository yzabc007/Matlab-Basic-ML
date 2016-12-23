clear ; close all ; clc

fprintf('Import data ... \n');
train_data = load('train79.mat');
X_train = train_data.d79;
Y_train = [ones(1,1000) -ones(1,1000)]';
test_data = load('test79.mat');
X_test = test_data.d79;
Y_test = [ones(1,1000) -ones(1,1000)]';

Iter = 3000;
eta = 1;
[acc_rate_train_GradDes, acc_rate_test_GradDes] = Perceptron_Class(X_train, Y_train, X_test, Y_test, Iter, eta, false)
[acc_rate_train_NormEq, acc_rate_test_NormEq] = Perceptron_Class(X_train, Y_train, X_test, Y_test, Iter, eta, true)
%% ==================== test the influence of #Iteration to results ====================
Arr_Iter = 1:200:3001;
eta = 1;
for i = 1:length(Arr_Iter)
	fprintf(['#Iteration = ' num2str(Arr_Iter(i)) '\n'])
	[acc_rate_train1(i), acc_rate_test1(i)] = Perceptron_Class(X_train, Y_train, X_test, Y_test, Arr_Iter(i), eta, false);
end
plot(Arr_Iter, acc_rate_train1, 'g-o', Arr_Iter, acc_rate_test1, 'r--s');
legend('Accuracy on Training Set','Accuracy on Testing Set');
xlabel('Number of Iteration');
ylabel('Accuracy');
saveas(gcf, 'Perceptron_Iteration.jpg')

%% ==================== test the influenc of learning rate eta to results ====================
Arr_eta = [0.000001 0.00001 0.0001 0.001 0.01 1 10 100 1000 10000 100000];
Iter = 100000; % large enough to ensure the convergence of perceptron
for i = 1:length(Arr_eta)
	fprintf(['Learning rate = ' num2str(Arr_eta(i)) '\n'])
	tic;
	[acc_rate_train2(i), acc_rate_test2(i)] = Perceptron_Class(X_train, Y_train, X_test, Y_test, Iter, Arr_eta(i), false);
	time(i) = toc;
end
log_Arr_eta = log(Arr_eta)/log(10);
plot(log_Arr_eta, time, 'g-o');
legend('Learning Rate');
ylabel('Costing Time');
saveas(gcf, 'Perceptron_Learning_rate.jpg')