function [acc_correct, Pred_labels_test] = KNN(k, data_train, labels_train, data_test, labels_test)

	num_pred_data = size(data_test, 1);
	name_classes = unique(labels_train);
	num_classes = length(name_classes);
	Count_K_neighbour = zeros(num_pred_data, num_classes);
	%% Compute the distance between each testing datapoint with all training datapoints
	distances = dist(data_test, data_train');
	[dump, Index_all] = sort(distances, 2, 'ascend');
	Index_k = Index_all(:, 1:k);
	%% Count the number of each class in first k neighbours
	for i = 1:num_classes
		Count_K_neighbour(:,i) = sum(labels_train(Index_k) == name_classes(i),2);
	end
	%% Pick the class with majority voting
	[dump, Index_most_vote] = max(Count_K_neighbour, [], 2);
	Pred_labels_test = name_classes(Index_most_vote);
	
	acc_correct = 1 - sum(Pred_labels_test ~= labels_test)/length(labels_test);
end