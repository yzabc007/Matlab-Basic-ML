function Gaussian_Kernal = Get_Gaussian_Kernal_Matrix(X, Y, sigma)

m = size(X, 1);
n = size(Y, 1);
Gaussian_Kernal = zeros(m,n);

%for i = 1:m
%    for j = 1:n
%        Gaussian_Kernal(i,j) = exp(-(sum(X(i,:)-Y(j,:)).^2)/(2*sigma^2));
%    end
%end

for i = 1:n
	Gaussian_Kernal(:,i) = exp(-(sum(((X - repmat(Y(i,:), m, 1)).^2),2))/(2*sigma^2));
end

end