function [acc_train, label_cluster] = Cluster_Acc_Train(X_train, Y_train, IDX, k)
	label_map = unique(Y_train);
	vote_cluster = zeros(k, size(label_map,2));
	label_cluster = zeros(k,1);
	% counting number of different classes in each cluster
	for m = 1:size(X_train, 1)
		for n = 1:size(label_map, 2)
			if Y_train(m) == label_map(n)
				vote_cluster(IDX(m), n) = vote_cluster(IDX(m), n) + 1;
			end
		end
	end
	% labeling each cluster
	for n = 1:k
        a = label_map(find(vote_cluster(n,:)==max(vote_cluster(n,:))));
		label_cluster(n,1) = a(1);
	end
	% compute accuracy on training set
	acc_train = 0;
	for n = 1:k
		acc_train = acc_train + max(vote_cluster(n,:));
	end
	acc_train = acc_train/size(X_train, 1);
end
