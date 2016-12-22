function average_acc_rate = kFold_Cross_Validation(k, X_train, Y_train, lambda, sigma, shuffle_samp)

n = size(X_train,1);
num_fold = floor(n/k);

acc_rate = zeros(1,k);
for i = 1:k
	if i == k
		test_X_fold = X_train(shuffle_samp((k-1)*num_fold+1:n),:);
		test_Y_fold = Y_train(shuffle_samp((k-1)*num_fold+1:n));
		train_X_fold = X_train(shuffle_samp(1:(k-1)*num_fold),:);
		train_Y_fold = Y_train(shuffle_samp(1:(k-1)*num_fold));
	else
		test_X_fold = X_train(shuffle_samp((i-1)*num_fold+1:i*num_fold),:);
		test_Y_fold = Y_train(shuffle_samp((i-1)*num_fold+1:i*num_fold));
		train_X_fold = X_train([shuffle_samp(1:(i-1)*num_fold),shuffle_samp(i*num_fold+1:n)],:);
		train_Y_fold = Y_train([shuffle_samp(1:(i-1)*num_fold),shuffle_samp(i*num_fold+1:n)]);
	end
	disp(['test fold=' num2str(i) ' sigma=' num2str(sigma) ' lambda=' num2str(lambda)]);
	tic;
	[acc_rate_train, acc_rate_test(i)] = Kernel_Classifier(train_X_fold, train_Y_fold, test_X_fold, test_Y_fold, lambda, sigma)
	toc;
end	
average_acc_rate = mean(acc_rate_test);
end