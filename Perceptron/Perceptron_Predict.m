function [Test_Metrics] = Perceptron_Predict(weights, inputs, targets, varargin)

	for i = 1:length(varargin)
		if ischar(varargin{i})
			if (strcmp(cellstr(varargin{i}), 'OutType'))
				OutType = cellstr(varargin{i+1});
			end
		end
	end
	inputs = [ones(size(inputs, 1), 1) inputs];
	if strcmp(OutType, 'linear')
		predicts = ActivationFunc(OutType, inputs, weights);
		Test_Metrics = sum((predicts - targets).^2);
		PlotData(OutType, weights, inputs, targets, predicts);
	elseif strcmp(OutType, 'binary')
		predicts = ActivationFunc(OutType, inputs, weights);
		Test_Metrics = 1 - sum(targets ~= predicts)/length(targets);
	elseif strcmp(OutType, 'logistic')
		predicts = ActivationFunc(OutType, inputs, weights);
		predicts(find(predicts > 0.5)) = 1;
		predicts(find(predicts <= 0.5)) = 0;
		Test_Metrics = 1 - sum(targets ~= predicts)/length(targets);
		PlotData(OutType, weights, inputs, targets, predicts);
	end
	
end