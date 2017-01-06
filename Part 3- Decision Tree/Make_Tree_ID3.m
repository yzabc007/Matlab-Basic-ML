function tree = Make_Tree_ID3(data, labels, featureNames)
	%% featureNames = {'Deadline', 'Party', 'Lazy'}

	num_data = size(data, 1);
	num_features = size(data, 2);
	name_classes = unique(labels);
	num_classes = length(name_classes);
	
	freq_classes = zeros(1, num_classes);
	for i = 1:num_classes
		freq_classes(i) = sum(labels == name_classes(i));
	end
	default = name_classes(find(freq_classes == max(freq_classes)));
	if num_data == 0 || num_features == 0
		fprintf(['Have reached an empty branch. Return class name = ' num2str(default) '\n']);
		tree = default;
		return
	else if num_classes == 1
		fprintf(['Only one class left. Return class name = ' num2str(labels(1)) ' \n']);
		tree = labels(1);
		return
	else
		gains = zeros(num_features, 1);
		for i = 1:num_features
			gains(i) = Compute_Info_Gain(data, labels, i);
		end
		Index_BestFeature = find(gains == max(gains));
		tree = struct();
		tree.name = featureNames(Index_BestFeature);
		fprintf(['Current feature with max info gain is ' char(featureNames(Index_BestFeature)) '\n']);
		
		%% extract all values in the feature of the data
		values = [];
		j = 1;
		not_exist = zeros(1,0);
		for i = 1:num_data
			detect_value = find(values == data(i, Index_BestFeature));
			if isequal(detect_value, not_exist) || isequal(detect_value, [])
				values(j) = data(i, Index_BestFeature);
				j = j + 1;
			end
		end
		
		for i = 1:length(values)
			Index_Curr_Value = find(data(:,Index_BestFeature) == values(i));
			NewData = data(Index_Curr_Value,:);
			NewData(:, Index_BestFeature) = [];
			NewFeatureName = featureNames;
			NewFeatureName(Index_BestFeature) = [];
			NewLabels = labels(Index_Curr_Value);
			
			fprintf(['Recursive calling at value = ' num2str(values(i)) ' \n']);
			subtree = Make_Tree_ID3(NewData, NewLabels, NewFeatureName);
			s = ['tree.child' num2str(values(i)) '= subtree;'];
			eval(s);
			tree;
		end
	end
end
