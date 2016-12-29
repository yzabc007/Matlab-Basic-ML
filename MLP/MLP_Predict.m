function [acc_rate, confusion_matrix] = MLP_Predict(Weight1, Weight2, X, target)

name_class = unique(target);
num_classes = length(name_class);
category_classes = zeros(size(target,1),1);
for i = 1:num_classes
	category_classes(find(target==name_class(i))) = i;
end

% Test Multi Perceptron
inputs = [ones(size(X,1),1) X];
hiddens = inputs * Weight1;
activation_hiddens = [ones(size(hiddens,1),1) Sigmoid(hiddens)];
hidden_outputs = activation_hiddens*Weight2;
outputs = Sigmoid(hidden_outputs);

[dummy, predicts] = max(outputs,[], 2);

acc_rate = 1 - sum(category_classes ~= predicts)/length(category_classes);
% Compute Confusion Matrix
confusion_matrix = zeros(num_classes);
for i = 1:size(predicts)
	confusion_matrix(category_classes(i), predicts(i)) =...
		confusion_matrix(category_classes(i), predicts(i)) + 1;
end
% Data for plotting ROC Curve (only for binary classifiers)
if num_classes == 2
	% assume the first class is positive
	True_Positive = confusion_matrix(1,1);
	True_Negative = confusion_matrix(2,2);
	False_Positive = confusion_matrix(2,1);
	False_Negative = confusion_matrix(1,2);
	Sensitivity = True_Positive / (True_Positive + False_Negative);
	Specificity = True_Negative / (True_Negative + False_Positive);
	Y_axis = [0; Sensitivity; 1];
	X_axis = [0; 1 - Specificity; 1];
	%f = fit(X_axis, Y_axis, 'poly2');
	%plot(f, X_axis, Y_axis, 'r-o');
	plot(X_axis, Y_axis, 'r-o')
	xlabel('False Positives rate');
	ylabel('True Positives rate');
	title('ROC curve');
end
end