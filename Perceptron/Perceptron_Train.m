function [weights] = Perceptron_Train(X_train, Y_train, Iter, eta, boolNormEq)

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
else
	X_train = [-ones(size(X_train, 1), 1) X_train];
	weights = pinv(X_train' * X_train) * (X_train' * Y_train);
end

end