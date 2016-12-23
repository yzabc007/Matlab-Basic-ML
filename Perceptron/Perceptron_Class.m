function [acc_rate_train, acc_rate_test] = Perceptron_Class(X_train, Y_train, X_test, Y_test, Iter, eta, boolNormEq)

if boolNormEq ~= true
	inputs = [-ones(size(X_train,1),1) X_train];
	targets = Y_train;
	weights = zeros(size(inputs,2),1);
	%weights = rand(size(inputs,2),1)*0.1-0.05;

	for i = 1:Iter
		activations = inputs*weights;
		activations(find(activations > 0)) = 1;
		activations(find(activations <= 0)) = -1;
		if sum(abs(targets-activations)) ~= 0
			weights = weights + eta * inputs' * (targets-activations);
		else
			fprintf(['Finish convergence  ...' num2str(i) ' \n']);
			break;
		end
	end
	if i == Iter
		fprintf('Haven''t finished convergence\n');
	end
	activations = inputs * weights;
	inputs_test = [-ones(size(X_test,1),1) X_test];
	activations_test = inputs_test * weights;
else
	weights = pinv(X_train' * X_train) * (X_train' * Y_train);
	activations = X_train * weights;
	activations_test = X_test * weights;
end

activations(find(activations > 0)) = 1;
activations(find(activations <= 0)) = -1;
acc_rate_train = 1 - sum(Y_train ~= activations)/length(activations);

activations_test(find(activations_test > 0)) = 1;
activations_test(find(activations_test <= 0)) = -1;
acc_rate_test = 1 - sum(Y_test ~= activations_test)/length(activations_test);
end