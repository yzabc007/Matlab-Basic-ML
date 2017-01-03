function [Weight1, Weight2] = MLP_Train(hidden_layer_size, X, target, eta, Iter, error_Threshold)
% Train Multi Perceptron
% initialize weights
num_Output_layer_node = length(unique(target));
Weight1 = rand(size(X, 2) + 1, hidden_layer_size);
%Weight1(find(Weight1 > 1/sqrt(size(X,2)+1))) = 1/sqrt(size(X,2)+1);
%Weight1(find(Weight1 < -1/sqrt(size(X,2)+1))) = -1/sqrt(size(X,2)+1);
Weight2 = rand(hidden_layer_size + 1, num_Output_layer_node);
%Weight2(find(Weight2 > 1/sqrt(hidden_layer_size + 1))) = 1/sqrt(hidden_layer_size + 1);
%Weight2(find(Weight2 < -1/sqrt(hidden_layer_size + 1))) = -1/sqrt(hidden_layer_size + 1);

name_classes = unique(target);
category_classes = zeros(size(target,1),1);
target_maxtrix = zeros(size(target,1), length(name_classes));
for i = 1:length(name_classes)
	category_classes(find(target==name_classes(i))) = i;
	target_maxtrix(find(category_classes==i),i) = 1;
end

for i = 1:Iter
	% forward
	input = [ones(size(X, 1), 1) X];
	hidden = input*Weight1;
	%activation_hidden = Sigmoid(hidden);
	activation_hidden = [ones(size(hidden, 1), 1) Sigmoid(hidden)];
	hidden_output = activation_hidden*Weight2;
	output = Sigmoid(hidden_output);
	Error = sum(sum((target_maxtrix - output).^2));
	fprintf(['Iteration: ' num2str(i) ' Error: ' num2str(Error) '\n']);
	if Error <= error_Threshold
		fprintf(['Converge at iter = ' num2str(i) '\n']);
		break;
	end
	% backward
	delta_o = (target_maxtrix-output).*output.*(1.0-output); 
	delta_h = activation_hidden.*(1.0-activation_hidden).*(delta_o*Weight2'); 
	update_W1 = eta*input'*delta_h(:,2:end);
	update_W2 = eta*activation_hidden'*delta_o;
	Weight1 = Weight1 + update_W1;
	Weight2 = Weight2 + update_W2;
end
end