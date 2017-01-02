function [weights, varargout] = Perceptron_Train(inputs, targets, varargin)
	inputs = [-ones(size(inputs, 1), 1) inputs];
	IsNormEq = false;
	GradDes = false;
	%% Control input arguments
	for i = 1:length(varargin)
		if ischar(varargin{i})
			if (strcmp(cellstr(varargin{i}), 'IsNormEq'))
				IsNormEq = true;
				GradDes = false;
			elseif (strcmp(cellstr(varargin{i}), 'GradDes')) 
				IsNormEq = false;
				GradDes = true;
				Iter = varargin{i+1};
				eta = varargin{i+2};
			elseif (strcmp(cellstr(varargin{i}), 'OutType'))
				OutType = cellstr(varargin{i+1});
			end
		end
	end
	%% Handle Exception
	if (IsNormEq == false) && (GradDes == false)
		fprintf('Wrong Input Arguments. \n');
		return
	end
	%% Gradient Descent Algorithm 
	if GradDes == true
		fprintf([char(OutType) ': Gradient Descent ... \n']);
		m = length(targets);
		weights = zeros(size(inputs, 2), 1);
		for i = 1:Iter
			activations = ActivationFunc(OutType, inputs, weights);
			Cost_Func(i) = sum((activations - targets).^2)/2/m;
			weights = weights + eta * inputs' * (targets - activations)/m;
			%fprintf(['Iteration: ' num2str(i) ' Error: ' num2str(Cost_Func(i)) '\n']);
			if Cost_Func(i) == 0 % or set a stopping threshold
				fprintf(['Finish convergence  ...' num2str(i) ' \n']);
				break;
			end
		end
		fprintf('Plotting ... \n');
		figure;
		plot([1:length(Cost_Func)], Cost_Func, 'r-x');
		xlabel('Iteration');
		ylabel('Cost');
		title(['Convergence: ' char(OutType)]);
		varargout{1} = Cost_Func;
	end
	%% Normal Equation Method
	if IsNormEq == true
		fprintf([char(OutType) ': Normal Equation.']);
		weights = pinv(inputs' * inputs) * inputs' * targets;
	end
end
