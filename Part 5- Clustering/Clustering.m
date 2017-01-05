function [acc_train_kmeans, acc_test_kmeans, acc_train_linkage] = Clustering()
clear
clc
% load training set
train_data = load('train79.mat');
X_train = train_data.d79;
Y_train = [7*ones(1,1000) 9*ones(1,1000)];
% load testing set
test_data = load('test79.mat');
X_test = test_data.d79;
Y_test = Y_train;
num_centroid = [2, 5, 10, 50];
for i = 1: size(num_centroid, 2)
	%%%%%%%%%%%%%%%% kmeans %%%%%%%%%%%%%%%%%%%%%%%
	[IDX, C] = kmeans(X_train, num_centroid(i));
	[acc_train_kmeans(i), label_cluster] = cluster_acc_train(X_train, Y_train, IDX, num_centroid(i));
	acc_test_kmeans(i) = cluster_acc_test(X_test, Y_test, C, label_cluster);
	%%%%%%%%%%%%%%%% linkage %%%%%%%%%%%%%%%%%%%%%%%
	Z = linkage(X_train, 'single', 'euclidean');
	T = cluster(Z, 'MaxClust', num_centroid(i));
	[acc_train_linkage(i), label_cluster] = cluster_acc_train(X_train, Y_train, T, num_centroid(i));
end
end
%%%%%%%%%% compute accuracy on training set %%%%%%%%%%%%
function [acc_train, label_cluster] = cluster_acc_train(X_train, Y_train, IDX, k)
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
%%%%%%%%%% compute accuracy on testing set %%%%%%%%%%%%
function acc_test = cluster_acc_test(X_test, Y_test, C, label_cluster)
	% compute accuracy on testing set
	label_testing = zeros(1, size(X_test, 1));
	dis_test = dist(X_test, C');
	for j = 1:size(X_test, 1)
		label_testing(j) = label_cluster(find(dis_test(j,:)==min(dis_test(j,:))));
	end
	acc_test = 1-sum(Y_test~= label_testing)/size(label_testing,2);
end
