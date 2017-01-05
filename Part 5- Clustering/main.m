clear;close all; clc;

fprintf('Import data ... \n');
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
	fprintf(['Number of centroids is ' num2str(num_centroid(i)) '.\n']);
	[Index_centroids, k_centroids, error] = kMeans(num_centroid(i), X_train);
	err{i,:} = error;
	[acc_train_kmeans(i), label_cluster] = Cluster_Acc_Train(X_train, Y_train, Index_centroids, num_centroid(i));
	acc_test_kmeans(i) = Cluster_Acc_Test(X_test, Y_test, k_centroids, label_cluster);
	
	%%%%%%%%%%%%%%%% linkage %%%%%%%%%%%%%%%%%%%%%%%
	%Z = linkage(X_train, 'single', 'euclidean');
	%T = cluster(Z, 'MaxClust', num_centroid(i));
	%[acc_train_linkage(i), label_cluster] = Cluster_Acc_Train(X_train, Y_train, T, num_centroid(i));
end
