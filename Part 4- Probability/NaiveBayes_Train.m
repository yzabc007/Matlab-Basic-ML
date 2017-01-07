function model = NaiveBayes_Train(train_data, train_labels)

	[train_data, data_map_table] = FeatureMap(train_data);
	[train_labels, labels_map_table] = FeatureMap(train_labels);

	name_classes = unique(train_labels);
	num_classes = length(name_classes);
	Prior_classes = zeros(1, num_classes);
	for i = 1:num_classes
		Prior_classes(i) = sum(train_labels == name_classes(i))/length(train_labels);
	end
	
	num_features = size(train_data, 2);
	fea_like_lookup = cell(1, num_features);
	for i = 1:num_features
		feature = train_data(:,i);
		name_values = unique(feature);
		num_values = length(name_values);
		loop_up = zeros(num_values, num_classes);
		for m = 1:num_classes
			Index_cur_class = find(train_labels == name_classes(m));
			num_cur_class = sum(train_labels == name_classes(m));
			for n = 1:num_values
				fea_cur_class = feature(Index_cur_class,:);
				%loop_up(n, m) = sum(fea_cur_class == name_values(n)) / num_cur_class;
				loop_up(n, m) = (sum(fea_cur_class == name_values(n))+1) / (num_cur_class+2);
			end
		end
		fea_like_lookup{i} = loop_up;
	end
	model = struct();
	model.prior = Prior_classes;
	model.likelihood = fea_like_lookup;
	model.datamap = data_map_table;
	model.labelmap = labels_map_table;
end