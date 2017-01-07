function [predict_label, posterior] = NaiveBayes_Predict(model, predict_data)

	%model.prior = Prior_classes;
	%model.likelihood = fea_like_lookup; %%cell
	%model.datamap = data_map_table;     %%cell
	%model.labelmap = labels_map_table;  %%1*1 cell
	prior = model.prior;
	fea_like_lookup = model.likelihood;
	data_map_table = model.datamap;
	labels_map_table = model.labelmap;
	
	num_classes = length(prior);
	num_features = size(predict_data, 2);
	num_data = size(predict_data, 1);
	posterior = zeros(num_data, num_classes);
	mapped_predict_data = zeros(size(predict_data));
	
	for i = 1:num_features
		data_map_table_fea = data_map_table{i};
		for j = 1:num_data
			Pos_equal = cellfun(@isequal, data_map_table_fea, repmat({predict_data{j,i}}, length(data_map_table_fea), 1));
			mapped_predict_data(j,i) = find(Pos_equal);
		end
	end
	
	for p = 1:num_data
		for i = 1:num_classes
			posterior_temp = prior(i);
			for j = 1:num_features
				fea_cur_loopup =fea_like_lookup{j};
				likelihood_cur = fea_cur_loopup(mapped_predict_data(p,j), i);
				posterior_temp = posterior_temp * likelihood_cur;
			end
			posterior(p,i) = posterior_temp;
		end
	end
	[dump, predict_label] = max(posterior, [], 2);
	predict_label = labels_map_table{1}(predict_label);

end