function [cost, grad] = CostFunc(OutType, weights, inputs, targets)

	activations = ActivationFunc(OutType, inputs, weights);
	m = length(targets);
	grad = zeros(size(weights));
	if strcmp(OutType, 'linear')	
		cost = sum((activations - targets).^2)/2/m;
		grad = inputs' * (activations - targets)/m;
	elseif strcmp(OutType, 'binary')
		activations(find(activations > 0)) = 1;
		activations(find(activations <= 0)) = -1;
		cost = sum((activations - targets).^2)/2/m;
		grad = inputs' * (activations - targets)/m;
	elseif strcmp(OutType, 'logistic')
		cost = sum(-targets' * log(activations) - (1-targets)' * log(1-activations)) /m;
		grad = inputs' * (activations - targets)/m;
	end
	
end