function acc_test = Cluster_Acc_Test(X_test, Y_test, C, label_cluster)
	% compute accuracy on testing set
	label_testing = zeros(1, size(X_test, 1));
	dis_test = dist(X_test, C');
	for j = 1:size(X_test, 1)
		label_testing(j) = label_cluster(find(dis_test(j,:)==min(dis_test(j,:))));
	end
	acc_test = 1-sum(Y_test~= label_testing)/size(label_testing,2);
end