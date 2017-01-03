# Multi-Perceptron
Implemenation of Multi-Perceptron using back-propagation

For now, the MLP in this file is for multi-classification.

The structure of the MLP in this file is a one hidden layer with "you-decide" nodes and a hard-max output layer which means to take the class with largest value as the predicted class.

In the file of main_MLP.m, the main job is to preprocess the data which is crucial for the performance of the MLP. Basically, I shuffle and normalize the data. There are more improvements that can be tried. Then I train the MLP and predict it in training set and testing set.

* MLP_Train.m 
 
I first categorize the lable and then use standard back-propagation algorithm (there are some variants of standard back-propagation that will be added) to update the weights.

* MLP_Predict.m

I use a hard-max output criteria to predict the class. Then confusion matrix and ROC curve are generated for better judgement of the classifier.

According to the experiments, the initialization of weights, preprocessing of the data and the number of nodes of hidden layer are crucial for the performance of MLP including the convergence rate and the accuracy.

Accoding to my understanding of MLP, two hidden layers are enough to approximate any decision boundary, so a version with two hidden layers will be added.
