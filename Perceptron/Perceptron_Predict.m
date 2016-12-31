function [acc_rate, predicts] = Perceptron_Predict(weights, inputs, targets)

inputs = [-ones(size(inputs,1),1) inputs];
predicts = inputs * weights;
predicts(find(predicts > 0)) = 1;
predicts(find(predicts <= 0)) = -1;
acc_rate = 1 - sum(targets ~= predicts)/length(targets);

end