function [XX, yy] = PreProcess(X, y)

	% shuffle
	samples_shuffle = randperm(size(X,1));
	X = X(samples_shuffle,:);
	y = y(samples_shuffle,:);
	
	% normalize
	X_std = std(X, 1);
	X_std(find(X_std==0)) = 1;
	X = (X - repmat(mean(X,1), size(X,1), 1)) ./ repmat(X_std, size(X,1), 1);
	%y = (y - mean(y)) ./ std(y);
	XX = X;
	yy = y;
end