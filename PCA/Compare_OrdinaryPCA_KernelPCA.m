function Compare_OrdinaryPCA_KernelPCA()
clear
clc
% load training set
train_data = load('train79.mat');
X_train = train_data.d79;
X = X_train;
% try different bandwidth sigma
sigma = 550;
disp('Implementing Kernel PCA ...');
[P_KernelPCA, V_KernelPCA] = KernelPCA(X, 2, sigma);
disp('Implementing Ordinary PCA ...');
[P_OrdinaryPCA, V_OrdinaryPCA] = OrdinaryPCA(X, 2);

disp('Visualizing ...');
figure(1)
plot(P_KernelPCA(1:1000,1),P_KernelPCA(1:1000,2),'r x');hold on
plot(P_KernelPCA(1001:2000,1),P_KernelPCA(1001:2000,2),'g o')
legend('Digit 7','Digit 9');
title(['Visualization of Kernel PCA (sigma = ' num2str(sigma) ')']);
xlabel('The First Eigenvector Basis');
ylabel('The Second Eigenvacor Basis')
grid on;
saveas(gcf,['KernelPCA' num2str(sigma) '.jpg'])

figure(2)
plot(P_OrdinaryPCA(1:1000,1),P_OrdinaryPCA(1:1000,2),'r x');hold on;
plot(P_OrdinaryPCA(1001:2000,1),P_OrdinaryPCA(1001:2000,2),'g o');
legend('Digit 7','Digit 9');
title('Visualization of Ordinary PCA');
xlabel('The First Eigenvector Basis');
ylabel('The Second Eigenvacor Basis');
grid on;
saveas(gcf,'OrdinaryPCA.jpg')
end 

function [P,V] = KernelPCA(X, dim, sigma)
K = Get_Gaussian_Kernal_Matrix(X, X, sigma);
n = size(X, 1);
%formulation from https://en.wikipedia.org/wiki/Kernel_principal_component_analysis
KK = K-(1/n)*ones(n)*K-K*ones(n)*(1/n)+(1/n)*ones(n)*K*ones(n)*(1/n);
[V D] = eig((1/n)*KK);
[~, des_order] = sort(diag(D), 'descend');
V = V(:, des_order(1:dim));
P = K*V;
end

function [P, V] = OrdinaryPCA(X, dim)
n = size(X,1);
B = X - repmat(mean(X,1), n, 1);
C = 1/(size(B,1)-1)*(B'*B);
[V D] = eig(C);
[~, des_order] = sort(diag(D), 'descend');
V = V(:, des_order(1:dim));
P = X*V;
end

function Gaussian_Kernal = Get_Gaussian_Kernal_Matrix(X, Y, sigma)
m = size(X, 1);
n = size(Y, 1);
Gaussian_Kernal = zeros(m,n);
for i = 1:n
	Gaussian_Kernal(:,i) = exp(-(sum(((X - repmat(Y(i,:), m, 1)).^2),2))/(2*sigma^2));
end
end