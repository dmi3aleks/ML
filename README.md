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

## Support Vector Machines

This is a supervised learning technique that is somewhat similar to logistic regression with a Large Margin optimization.

Instead of a sigmoid used in the logistic regression, a special cost function is used:
1. "\_" for a positive example (i.e. is growing linearly as X values go farther from 1 in a direction of negative values;
1. "_/" for a negative example (i.e. is growing linearly as X values go farther from -1 in a direction of positive values.

Application example: large margin classifier, which will try to split data points into classes while maximizing the margin.

### Kernel

Given input data set with points X we pick a few of them randomly and call them landmarks.

We build a new set of features based on proximity of data points to landmarks.

For Gaussian kernels, we introduce a similarity function to calculate values of new features. F1 = exp( - (|x - l1|/(2*sigma^2))).

Changing sigma results in changing the shape of the similarity function.

We predict 1 if Theta0 + Theta1*F1 + ... ThetaN*FN >= 0.

SVM is represented in Octave/Matlab as liblinear, libsvm packages.

### Multi-class classification



## Clustering using K-means algorithm (unsupervised learning)

Problem statement: find clusters in an unlabeled dataset.

Optimization objective: find such K centroids for clusters that the total distortion (sum of distances between a point and its cluster centroid) are minimal.

Algorithm:
1. randomly initialize K cluster centroids (E.g. by picking K data points from the input set at random);
1. repeat:
- find the closest centroid for each data point in the input set;
- update cluster centroids with a mean of the data points it is linked to.

Since there is a chance to converge to a local minimum, it makes sense to:
1. for many different random initializations for centroid centers:
- run algorithm to convergence;
- calculate the cost function.
2. pick the result with the smallest value of the cost function.

Picking the right amount of clusters K:
1. Elbow method: plot minimum of the cost function with regard to the amount of clusters - pick K at the "elbow" of the graph (i.e. the last one that gives a substantial improvement in terms of cost function reduction);
1. An alternative is to look at how clusters are going to be used later by the downstream purpose (E.g. see what K enables downstream calculation to perform the best).

## Dimensionality Reduction and Principal Component Analysis

Can be used for data compression. For example, reducing data from 3D to 2D.

Objective: make a projection of points in 3D space on a plane in such a way that the total distance to the plane is minimal (i.e. maximum variability of the data is retained).

Another application: data visualization (as it is hard to visualize more than 3D space).

Another one: sppeding up learning algorithm.

### Principal Component Analysis

Reduce from n-dimension to k-dimension: find k vectors onto which to project the data, so as to minimize the projection error.

#### Algorithm

Pre-processing:

Do feature scaling and mean normalization: Xj = (Xj - MUj)/SIGMAj

Main steps:
1. calculate a covariance matrix:
Sigma = (1/m) * SUM(Xi * Xi');
1. compute eigenvectors of matrix Sigma:
[U, S, V] = svd(Sigma);
1. take only first k vectors:
Ureduce = U(:, 1:k)
1. calculate a projection:
z = Ureduce' * x.

To reconstruct data from a reduced form:
1. Xi = Ureduce * Zi.

Choosing k (number of principal components):
1. [U, S, V] = svd(Sigma);
1. pick smallest value of k for which:
SUM(Sigma(i,i) where i = 1, k) / SUM(Sigma(i,i) where i = 1 to last) >= 0.99 i.e. 99% of varience is retained.

### Applying unsupervised learning

Find dimension reducing mapping on a training set. Pick one that performs the best on the cross-validation set.
Estimate performance on the test set.

Note: applying PCA to reduce overfitting is not a good idea, use regularization instead ((lambda/2*m)*SUM(THETAj^2)).

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

## Anomaly Detection

Unsupervised learning algorithm.

Problem: identify unusual data points.

Algorithm:
1. choose features Xi that are believed to be indicative of anomalous examples and project the training set into a feature space;
1. fit parameters MUi and SIGMAi for a Gaussian distribution that would model a training set;
1. for a new example, compute its probabilty using a multi-dimensional probability density function;
1. mark example as anomalous if its probability is lower than a pre-picked threshold Epsilon.

Algorithm evaluation and choosing the right Epsilon:
1. split available data into training, cross validation and test sets;
1. fit a model to the training set;
1. for various Epsilon values:
- calculate precision/recall or an F1 Score on a cross-validation and pick an Epsilon that optimizes it.
Precision = (True Positives)/(True Positives + False Positives)
Recall = (True Positives)/(True Positives + False Negatives)
F1Score = 2*Recall*Precision/(Recall + Precision).

Notes:
1. anomaly detection is a better fit for a problem as opposed to the supervised learning algorithm when:
- there is a very small number of positive examples (anomalies);
- there is a large amount of negative examples (i.e. regular cases);
- many different types of anomalies are anitipated (i.e. it is hard to learn from previous examples).

## Recommender system

Problem statement: to recommend a new item to a user based on the ratings he gave before to other items and on ratings from other users.

### Content-based recommender system

Assumption:
1. items have a known set of features and have numerical values for them (i.e. can be projected to the feature space as vector Xi);
1. user has rater at least some items.

#### Algorithm

Cost function:
- Alt A: for each user minimize a sum of prediction error for all content items by picking an optimal Theta, where prediction Yi = Theta' * Xi for item i;
- Alt B: find Theta for multiple user by optimizing a sum of total error across a set of users.

Note: need to add regularization by penalizing high values of weights in order to avoid overfitting.

Gradient descent update (partial derivative of a cost function with regard to Theta): similar to that of a linear regression.

Pass cost function and partial derivatives to the optimization method (E.g. fminunc) to obtain an optimal set of weights Theta.

### Collaborative Filtering

Is useful when there is no pre-defined set of features for items being rated.

Basically we are trying to find optimal values for item features as well as optimal weights that, when applied to each other, would minimze an error with regard to the actual user ratings.

Algorithm:
1. initialize feature vectors and weights vectors to small random numbers;
1. minimize cost function combining features and weights using gradient descent or similar optimisation method;
1. to make a prediction calculate: Theta(user)' * x(item), where both Theta and x have been learned in the optimisation step above.

Note: if feature vectors for items being rated are known or once they are learned, items can be projected into the feature space and similarity between them can be established using an absolute distance between vectors.

Note: in case there are many missing ratings, Mean Normalization technique can be helpful to get better predictions. Namely, it is a transformation of the rating values that subtracts mean rating for a given item from all ratings for this item. Once optimization has found feature and weight values, rating for a new combination of user and item can be calculated as: Theta(user)' * x(movie) + MeanRating(movie).

## Photo OCR

Problem: detect and recognize text in images.

### Photo OCR pipeline:
1. text detection: given an image, find areas containing text (E.g. using a "sliding window" technique plus region expansion [similar to dilation]);
1. character segmentation: find splits between individual characters in a text region (E.g. sliding window in the text region + detection check based on a classifier: E.g a specially trained NN for splits);
1. character classification: classify individual characters (take areas between splits and pass them to a classifier one by one [E.g. a NN]).

### Notes and considerations:
1. artificial data synthesis can be used to get more data for a training set (E.g. rendering artificial pictures with text);
1. more data can be synthesized by adding meaningful distortions to the exisitng data (E.g. warps);
1. adding Gaussian noise is not helpful in terms of data synthesis;
1. it is important to make sure that classifier being used has low bias (can verify by plotting learning curves) so that getting more data would result in a better fit;
1. when working on an ML problem it is worth asking the following question: "How much work would it be to get 10x as much data as we currently have?". Sometimes it is surprisingly inexpensive to get more data and as a result get a better performing algorithm.

### Ceiling analysis

Ceiling analysis helps answer the question of what part of the pipline should be improved.

Algorithm:

1. estimate accuracy of the overal pipeline given a test set;
1. pick a single part of a pipeline and substitute its outcome with perfect manually created results - check how the overal pipeline accurace changes as the result;
1. if there is no much change, than probably it is not worth investing into this part of the pipeline and find one improvements to which can improve the overall accuracy substantially.
