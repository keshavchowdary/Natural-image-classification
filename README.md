# Natural-image-classification
An end to end ML model to classify the natural images
This project is based on the classification of the natural images. Natural image classification is a useful technique in many areas. The aim of our project is to model three classifiers and compare the performance efficiency of all these classifiers.
The project we worked on is the natural image classification and the dataset is CIFAR-10. This CIFAR-10 dataset contains 60,000 32*32 colour images in 10 target classes and each image has 3 colour values i.e. (r,g,b) that means we have in total of 32*32*3 input features. Each target class contains 6000 images. The entire dataset is divided into 6 data batches, out of which five data batches are for training and one data batch for testing. Each data batch contains 10,000 samples.
Models used are,
RandomForest Classifier
RandomForest Classifier is a supervised algorithm that fits the data using decision trees as the base estimators. This classifier makes decision trees using some random subsets in the dataset and collects the data from all the decision trees by voting and makes the final prediction.
GaussianNB Classifier:
This classifier assumes that all the features are independent of each other and considers all of these properties to independently contribute to the probability that the image belongs to a particular class and the probability is calculated using bayes theorem.
  
P(y) is the prior probability, P(X/y) is the likelihood and P(X) is the evidence. It calculates the posterior probability for a sample being classified into different target classes and the class with highest probability will be the target class corresponding to that sample.
K-Nearest Neighbors:
This is a supervised classification algorithm that works by calculating the distance between the test data and each row in the training dataset (in general euclidean distance is considered) and sort all those distance in ascending order and chooses the k nearest points and assign the most frequent class to the test data.
Evaluating the parameters for each model
For each model I have evaluated the model with different parameters using GridSearchCV and the best_parameters obtained are:
For RandomForest model:
{'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 8}
For GaussianNB model:
{'var_smoothing': 0.02310129700083159}
For KNN model:
{'n_neighbors': 9, 'p': 2}

Then trained the model with corresponding best_params_ which are obtained from GridSearchCV.
