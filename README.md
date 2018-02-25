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
1. [optional] middle layer (amount of nodes is comparable to that of the input layer;
1. output layer: the same size as the amount of classes.

Training: fitting the weights and biases of the network (bias is a constant node in a network).

Training algorithm for node weights:
1. assign weights randomly;
1. make a forward propagation, pushing input data through the network to obtain initial predictions;
1. calculate vector of errors (expected results minis predictions made by the network);
1. make a backpropagation pass, pushing error obtained in the previous step through the network, calculating partial derivatives of the cost function with respect to weights and biases;
1. pass cost funciton and its gradient to the optimization routing (E.g. fminunc) or to a Gradient Descent routine to get an optimal weights and biases for the network.

Applying neural network: for any new example, convert it to feature vector and apply a transformation dictated by the network weights (layer by layer) to get the prediction (E.g. picking a class with the highest value in the output layer).


## Learning with large datasets

"It's not who has the best algorithm that wins. It's who has the most data".j

### Plotting learning curves (cost function with regard to the dataset size)

Shape of learning curves for training and cross-validation sets help understand whether increasing dataset size actually helps.

If learning curves meet go close to each other and flat out - there is no much point in pursuing a bigger dataset.

If it is not the case there are two options:
1. training a model on more data;
1. adding more features to the model to reduce a bias (i.e. achieve the better fit).

### Stochastic Gradient Descent

The regular (Batch) Gradient Descnt (GD hereon) calculates hypothesis value and associated error for each of the datapoints in the dataset in order to make a single step (it does so for every dimension of feature space to calculate partial derivative of the cost function in a given point in feature space).

Stochastic GD suggest an alternative approach:
1. randomly shuffle dataset;
1. calculate the next position in the feature space by calculating partial derivative for a single (next) example.

Obvious benefit is that it is less computationally expensive to start making progress.

On the other hand, there is no monotonous convergence. Stochastic GD will sometimes move the search point in a wrong direction, but eventually come to the global optimum and will continue oscillating around it.

A method for stabilizing a Stochastic GD is as follows:
1. decrease the learning rate Alpha as the amount of iteration grows.

Convergence check:
1. calculate cost function for a given example before adjusting transformation function Theta, take an average of this cost value over a 1000 or so points, and see if it goes down in general as the algorithm progresses.

### Mini-batch Gradient Descent

A modificaiton of Stochastic GD that takes N datapoints at a time to make a single descent step (E.g. N = 10).

### Online Learning

A learning algorithm which learns as new data points come in.

A modification of Stochastic GD can be used by repeating:
1. get new data point;
1. update regression function Theta with a single step of Stochastic GD: Theta = Theta - Alpha * (h(x) - y)*x.

Example applicatoin is an online resource that offers shipping services. For each new user it can take user location and shipment parameters as well as the proposed bid as the feature vector and calculate the probability of a given user accepting a bid, therefore picking an optimal bid price.

### Map-reduce and data parallelism

Many learning algorithms involve calculating summs of function application to large training sets.
Such calculation can be split in chunks and distributed between CPU cores or computers in a newtwork and then a sum can be assembled with a map-reduce job.
