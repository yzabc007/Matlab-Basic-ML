function NormFeature = FeatureNorm(Features)
	
	m = size(Features, 1);
	Features_std = std(Features, 1);
	Features_std(find(Features_std==0)) = 1;
	NormFeature = (Features - repmat(mean(Features,1), m, 1)) ./ repmat(Features_std, m, 1);

end