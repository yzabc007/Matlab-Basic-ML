function [lambda,E] = PowerMethod(A,n,m)
% Initialization by ones matrix
x0 = ones(size(A,1),1);
% Compute top m eigenvectors
for j = 1:m
    % Iteration of Power Method
    for i = 1:n
        x0 = A*x0/norm(A*x0);
    end
    V = x0;
    % Compute eigenvalue through Rayleigh quotient
    a = norm(A*V/norm(V));
    % Update A to compute next eigenvector
    A = A -a*(V*V'/(V'*V));
    lambda(:,j) = a;
    E(:,j) = V;
end
end