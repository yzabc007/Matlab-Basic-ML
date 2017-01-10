function [Index_centroids, k_centroids, error_dist, num_Iter] = kMeans(k, data)

	num_data = size(data, 1);
	shuffle_data = randperm(num_data);
	%% initialize k centroids
	k_centroids = data(shuffle_data(1:k),:);
	error_dist(1) = sum(sum(dist(data, k_centroids')));
	%fprintf(['Initial distance error = ' num2str(error_dist(1)) '.\n']);
	
	Index_centroids_old = zeros(num_data, 1);
	num_Iter = 1;
	while true
		fprintf(['Iteration = ' num2str(num_Iter) '.\n']);
		num_Iter = num_Iter + 1;
		distances = dist(data, k_centroids');
		[dump, Index_centroids] = min(distances, [], 2);
		if isequal(Index_centroids, Index_centroids_old)
			break;
		end
		for i = 1:k
			this_centroid = (Index_centroids == i);
			if sum(this_centroid) > 0
				%% mean
				%k_centroids(i,:) = sum(data.*repmat(this_centroid, 1, size(data,2)), 1) ./ sum(this_centroid);
				%% median
				data_this_centroid = data(this_centroid, :);
				k_centroids(i,:) = FindMedianMultiDim(data_this_centroid);
			end
		end
		Index_centroids_old = Index_centroids;
		error_dist(num_Iter) = sum(sum(dist(data, k_centroids')));
		%fprintf(['Distance error = ' num2str(error_dist(num_Iter)) '.\n']);
	end
end

function median = FindMedianMultiDim(data_matrix)
	%% samples * dims 
	num_data = size(data_matrix, 1);
	dist_withOtherPts = zeros(1, num_data);
	distance_matrix = dist(data_matrix');
	distance_sum = sum(distance_matrix, 2);
	median = data_matrix(find(distance_sum == min(distance_sum)),:);
	median = median(1,:);
end