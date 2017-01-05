function gain = Compute_Info_Gain(data, labels, feature)
	gain = 0;
	num_Data = size(data, 1);
	%% extract all values in the feature of the data
	values = [];
	j = 1;
	not_exist = zeros(1,0);
	for i = 1:num_Data
		detect_value = find(values == data(i, feature));
		if isequal(detect_value, not_exist) || isequal(detect_value, [])
			values(j) = data(i, feature);
			j = j + 1;
		end
	end
	%% count numbers of different values for each class
	num_values = length(values);
	name_classes = unique(labels);
	num_classes = length(name_classes);
	freq_values = zeros(num_values, num_classes);
	Entropy_val_class = 0;
	Entropy_val = 0;
	for i = 1:num_values
		curr_labels = labels(find(data(:, feature) == values(i)));
		for j = 1:num_classes
			freq_values(i, j) = sum(curr_labels == name_classes(j));
			Entropy_val_class = Entropy_val_class + Compute_Entropy(freq_values(i,j)/length(curr_labels));
		end
		Entropy_val = Entropy_val + (length(curr_labels)/num_Data) * Entropy_val_class;
		Entropy_val_class = 0;
	end
	%% count numbers of different classes
	Entropy_all = 0;
	freq_classes = zeros(1, num_classes);
	for i = 1:num_classes
		freq_classes(i) = sum(labels == name_classes(i));
		Entropy_all = Entropy_all + Compute_Entropy(freq_classes(i)/num_Data);
	end
	%% compute the final infromation gain 
	gain = Entropy_all - Entropy_val;
	
end