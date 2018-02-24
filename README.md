# Machine Learning notes

## Linear and polynomial regression

Numeric prediction and forecasting.
Belongs to a supervised learning category as training data contains labeled (numerical) data.

Trained by using optimization algorithm on regression parameters.
Optimization objective is a reduction of the cost function, which represents an error in predictions for data points from a training set.

## Logistic regression

Applied to classification problems. Applies sigmoid to regression function to convert prediction to a class.

Training set is labeled.

## Neural Networks

Supervised learning (labeled training dataset).

Usually applied to classification problems.
Input is a vector of features (E.g. pixels of an image in a vectorized form).

A few layers of network (2~3):
1. input layer: usually of the same size as the input feature vector;
2. [optional] middle layer (amount of nodes is comparable to that of the input layer;
3. output layer: the same size as the amount of classes.

Training: fitting the weights and biases of the network (bias is a constant node in a network).

Training algorithm for node weights:
1. assign weights randomly;
2. make a forward propagation, pushing input data through the network to obtain initial predictions;
3. calculate vector of errors (expected results minis predictions made by the network);
4. make a backpropagation pass, pushing error obtained in the previous step through the network, calculating partial derivatives of the cost function with respect to weights and biases;
5. pass cost funciton and its gradient to the optimization routing (E.g. fminunc) or to a Gradient Descent routine to get an optimal weights and biases for the network.

Applying neural network: for any new example, convert it to feature vector and apply a transformation dictated by the network weights (layer by layer) to get the prediction (E.g. picking a class with the highest value in the output layer).



