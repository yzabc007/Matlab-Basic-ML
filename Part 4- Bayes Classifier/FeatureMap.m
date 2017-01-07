function [data_mapped, map_table] = FeatureMap(data_cells)
%% convert string into number
	num_features = size(data_cells, 2);
	data_mapped = zeros(size(data_cells));
	map_table = cell(1, num_features);
	
	for i = 1:num_features
		datapoint = data_cells(:,i);
		name_values = unique(datapoint, 'stable');
		map_table{i} = name_values;
		num_values = length(name_values);
		for j = 1:num_values
			Pos_equal = cellfun(@isequal, datapoint, repmat({name_values{j}}, length(datapoint), 1));
			Index_value = find(Pos_equal == 1);
			data_mapped(Index_value,i) = j; 
		end
	end
end