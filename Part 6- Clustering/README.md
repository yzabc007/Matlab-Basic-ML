# Clustering

Implementation of K-means clustering and linkage clustering

The code for simple k-means clustering is presented and the errors of distance in each iteration are stored to check the convergence. However, after first update, the errors of each iteration are not strictly converged.

In the main function, there is one method, majority voting, to evalute the performance of clustering algorithms on training set as well as on testing set.

To remove noise effectively, we can replace mean average with MEDIAN, which is also computationally more expensive.
