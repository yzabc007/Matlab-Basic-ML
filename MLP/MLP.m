clear; close all; clc;

X = [0 0; 0 1; 1 0; 1 1];
target = [0; 1; 1; 0];

% hidden layer size = 2
num_hidden_nodes = 2;
Weight1 = rand(size(X, 2) + 1, num_hidden_nodes); % 3*2
Weight2 = rand(num_hidden_nodes + 1, 1); % 3*1

% Training
Iter = 100000;
eta = 0.25;
for i = 1:Iter
	% forward
	input = [ones(size(X, 1), 1) X]; % 4*3
	hidden = input*Weight1; % 4*2 = 4*3 * 3*2
	%activation_hidden = Sigmoid(hidden); % 4*2
	activation_hidden = [ones(size(hidden, 1), 1) Sigmoid(hidden)]; % 4*3
	hidden_output = activation_hidden*Weight2; % 4*3 * 3*1
	output = Sigmoid(hidden_output); % 4*1
	Error = sum((target - output).^2);
	fprintf(['Iteration: ' num2str(i) ' Error: ' num2str(Error) '\n']);
	if Error == 0
		fprintf(['Converge at iter = ' num2str(i) '\n']);
		break;
	end
	% backward
	delta_o = (target-output).*output.*(1.0-output); % 4*1 
	delta_h = activation_hidden.*(1.0-activation_hidden).*(delta_o*Weight2'); % 4*3 
	update_W1 = eta*input'*delta_h(:,2:end); % 3*2 = 3*4 * 4*2
	update_W2 = eta*activation_hidden'*delta_o; % 3*1 = 3*4 * 4*1
	Weight1 = Weight1 + update_W1;
	Weight2 = Weight2 + update_W2;
end
% Testing
input = [ones(size(X, 1), 1) X];
hidden = input*Weight1;
%activation_hidden = Sigmoid(hidden);
activation_hidden = [ones(size(hidden, 1), 1) Sigmoid(hidden)];
hidden_output = activation_hidden*Weight2;
output2 = Sigmoid(hidden_output);
