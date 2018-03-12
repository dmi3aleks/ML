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

A supervised learning algorithm (training dataset is labeled) used for solving classification problems.

Usually applied to classification problems.
Input is a vector of features (E.g. pixels of an image in a vectorized form).

A few layers of network (2~3):
1. input layer: usually of the same size as the input feature vector;
1. hidden layers: one by default, or multiple but with the same amount of hidden units in each hidden layer (usually the more units there are the better);
1. output layer: the same size as the number of classes.

Cost function:
J(Theta) = -(1/m) * (sum_for_all_examples(sum_for_all_classes( Yk * log(H*Xi) + (1 - Yk)*log(1 - (H*Xi))) + lamda/2*m * sum(THETAij^2)

Training: fitting the weights and biases of the network (bias is a constant node in a network).

Training algorithm for node weights:
1. randomly initialize weights;
1. implement forward propagation to get a hypothesis vector for each;
1. implement code to compute cost funciton (J(Thetat));
1. implement backpropagation to compute partial derivatives of J with regard to THETA(j,k). For each example from a training set:
- perform forward propagation, pushing input data through the network to obtain the initial hypothesis for a given example;
- get activation (a) and delta terms for each layer in the network (pushing errors back through the network) and calculate partial detivatives of J with regard to THETA(j,k);
- use gradient checking to compare patial derivatives computed using backpropagation VS using numerical estimate; then disable gradient checking (as it is computationally expensive);
1. pass cost funciton and its gradient to the optimization routing (E.g. fminunc) or to a Gradient Descent routine to get an optimal weights and biases for the network (i.e. minimizing J(Theta) as a function of parameter Theta).

Applying neural network: for any new example, convert it to feature vector and apply a transformation dictated by the network weights (layer by layer) to get a hypothesis (E.g. picking a class with the highest value in the output layer).

### Gradient checking

Backpropagation algorithm is hard to debug and is giving a surprisingly reasonable results even if trivial errors (like off by one) are present in the implementation. One way to make veriy correctness of partial derivatives is to compare those calculated values with numerically calculated estimates.

dJ/dTheta = lim( (J(Theta + Epsilon) - J(Thetat - Epsilon)/2*Epsilon) )

Sufficiently small Epsilon (say around 10^-4) usually work well.

### Random Initialization

Initializing initial Theta to zeroes would result in getting the same values for different units (nodes) of a hidden layer are going to be identical, introducing a false symmetry. Initializing Theta randomly (values is a [-Epsilon, Epsilon] range) allows to break a symmetry.

## Advice for applying machine learning

It is important to estimate model performance by minimizing a test error on a cross-validation set.

### Algorithm

Split your example data into:
1. 60%: training set;
1. 20%: test set;
1. 20%: cross-validation set.

For each set of model parameters:
1. train model on a training set;
1. calculate test error on a test set;
1. pick a parameter set that minimizes training error on a test set;
1. calculate cross-validation error -> this becomes an estimate of model performance.

### Diagnosing Bias VS Variance

High Bias is a phenomenon that occurs when a model is build upon a strong sentiment that does not actually model the data in question well.
High Bias results in underfitting (E.g. picking a linear regression model for a data that follows a polynomial curve).

High Variance is a phenomenon that occurs when model overfits a training set (E.g. a high order polynomial is used).

It helps to plot a dependency of error values on model parameters to see whether algorithm is suffering from bias or from varience:
1. plot test error and cross-validation (CV) error against a variable model parameter (E.g. an order of polynomial used in a regression model);
1. if both test error and CV error are high for a given value of a parameter - it is idicative of a high bias;
1. if test error is low but CV error is high - it is indication of a high variance.

### Regression with regularization and Bias/Variance

Large values of a regularization coefficient Lambda penalize high values of regression weights Theta, leading to a high bias (underfitting).
Small values of a regularization coefficient Lambda encourage high variance (overfitting).

To pick an optimal value for a regularization coefficient Lambda one can try multiple values and pick the one that minimizes the cross-validation error.

### Learning Curves

Plot cross-validation error and training error against the amount of examples used in a training set:
1. In case of a high bias, both CV error and training error will be quite high (and will get close to each other pretty soon). If a learning algorithm is suffering from high bias, getting more training data will not (by itself) help much.
1. In case of high varience, training error will be relatively low, but CV error will be initially high, getting lower as the amount of examples in a training set increases. If a learning algorithm is suffering from high variance, getting more training data is likely to help.

### Typical fixes for high bias and high varience

High bias:
1. try getting additional features;
1. try adding polynomial features;
1. try decreasing a regularization parameter Lambda.

High variance:
1. get more training examples;
1. try a smaller set of features;
1. try increasin a regularization parameter lambda.

### Bias and variance with Neural Networks

Neural Networks with a small amount of layers and nodes in middle layers, though computationally cheaper to train, tend to be prone to high bias.

Naural Networks with a large amount of layers and nodes in middle layers, tend to be computationally more expensive and can suffer from high variance. Using regularization helps to address overfitting.

### Building a Spam Classifier

#### Error Analysis

1. start with a simple algorithm that can be implemented quickly;
1. plot learning curves to decide if more data, more features, etc. are likely to help;
1. manually examine the examples your algoritm made errors on. See if there is a systematic trend.


#### Numerial evaluation of algorithm performance

When evaluating algorithm it is important to have a quick numerical metric of how good it performs.

It is important to use correct metrics for skewed classes (i.e. when there is a big disbalance between the amount of positive and negative examples). Such metrics are:

Precision = True Positives/(True Positives + False Positives)

Recall = Ture Positives/(True Positives + False Negatives)

F1Score = 2 * (Precision * Recall)/(Precision + Recall)

#### Designing a high accuracy learning system

Features should have sufficient information to predict the result accurately.

Useful test: given the input features, can a human expert confidently predict the result?

## Support Vector Machines

This is a supervised learning technique that is somewhat similar to logistic regression with an addition a Large Margin optimization.

Instead of a sigmoid used in a Logistic Regression, a special cost function is used:
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

Many SVM packages already have built-in multi-class classification functionality.

Otherwise, one can use a one-vs-all method. Namely, to train K separate SVM models, one for each class of points.

For each new point, calculate Theta[1..K] and pick the class with the highest prediction value.

### A rule of thumb for applying SVM VS applying logistic regression

Legend:
n: number of features
m: number of examples

1. if n is large relative to m: use logistic regression or SVM without a kernel (i.e. using a linear kernel);
1. if n is small, while m is intermediate: use SVM with Gaussian kernel;
1. if n is small, while m is large: create or add more features, then use a logistic regression or SVM without kernel.

Note: a Neural Network is likely to work well for all of the cases above, but it is slower to train.

### Applying SVM to Spam Classification

Algorithm.
1. preprocessing an email:
- lower-casing all words;
- stripping HTML (remove all the HTML tags);
- normalizing URLs (all URLs get replaced with a fixed token, E.g. "httpaddr");
- nomralizing email addresses;
- normalizing numbers;
- normalizing currency signs;
- word stemming (using a dictionary form of a word);
- removeal of non-words;
1. extract features from the email:
- map resulting text to word indices using a Vocabulary List (VL). VL is a list of words most commonly used in spam emails.
- convert email to a feature vector of lenght N, where N is the amount of words in the Vocabulary List. Each vector element is a binary value indicating whether a given word from VL is present in a given email;
1. train an SVM of a resulting set of vectors and corresponding labels;
1. for any new email apply a transformation with weights from a trained SVM to obtain a spam/non-spam label.

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
