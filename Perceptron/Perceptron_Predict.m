function [Test_Metrics] = Perceptron_Predict(weights, inputs, targets, varargin)

for i = 1:length(varargin)
    if ischar(varargin{i})
        if (strcmp(cellstr(varargin{i}), 'OutType'))
            OutType = cellstr(varargin{i+1});
        end
    end
end

inputs = [-ones(size(inputs, 1), 1) inputs];

if strcmp(OutType, 'linear')
    predicts = ActivationFunc(OutType, inputs, weights);
    Test_Metrics = sum((predicts - targets).^2);
    if size(inputs, 2) == 2
        fprintf('Plotting ... \n');
        figure;
        plot(inputs(:,2), targets, 'rx');
        xlabel('Inputs');
        ylabel('Targets');
        title('Visualization of Linear Regression');
        hold on;
        plot(inputs(:,2), predicts, '-');
        legend('Data', 'Linear regression');
        %legend('Train data fit', 'Linear Regression');
        hold off
    end
elseif strcmp(OutType, 'class')
    predicts = ActivationFunc(OutType, inputs, weights);
    Test_Metrics = 1 - sum(targets ~= predicts)/length(targets);
elseif strcmp(OutType, 'logistic')
	predicts = ActivationFunc(OutType, inputs, weights);
	Test_Metrics = 1 - sum(targets ~= predicts)/length(targets);
end
end