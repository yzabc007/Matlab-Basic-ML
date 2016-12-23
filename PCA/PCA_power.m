function [P,P_pca,V,V_pca,a] = PCA_power(X,n)
% Subtract mean
B = X-ones(size(X,1),1)*mean(X,1);
% Compute convariance matrix
C = 1/(size(B,1)-1)*(B'*B);
% Compute top n eigenvector by power method
[a, V] = PowerMethod(C,1000,n);
% Compute eigenvector by standard PCA
COEFF = pca(X);
V_pca = COEFF(:,1:n);
% Keep same direction of eignevector from
% power method and standard PCA
for i = 1:size(V,2)
    temp = find(V(:,i)~=0);
    if V_pca(temp(1))<0
        V_pca(:,i) = -V_pca(:,i);
    end
    if V(temp(1))<0
        V(:,i) = -V(:,i);
    end;
end
% Embed raw data
P = X*V;
P_pca = X*V_pca;
