function activations =  ActivationFunc(activat_type, inputs, weights)
%% Different Activation function for outpur layer
	if strcmp(activat_type, 'linear')
		activations = inputs * weights;
	elseif strcmp(activat_type, 'binary')
		activations = inputs * weights;
		activations(find(activations > 0)) = 1;
		activations(find(activations <= 0)) = -1;
	elseif strcmp(activat_type, 'logistic')
		activations = inputs * weights;
		activations = 1 ./ (1 +  exp(-1 * activations));
	end
end
