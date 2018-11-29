"""
Robert Best
CMSC 510
Assignment 2
Due 11/12/18
"""
import random
import numpy as np
from time import time
from sklearn import metrics


def getK(X1, X2):
    print("\tBuilding K ({}x{})...".format(X1.shape[0], X2.shape[0]), end='', flush=True); t = time()
    checkpoints = np.multiply(range(1, 11), X1.shape[0]/10)
    K = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x in enumerate(X1):
        if i in checkpoints: # Print progress
            print("{}%...".format((np.where(checkpoints==i)[0][0]+1)*10), end='', flush=True)
        for j, y in enumerate(X2):
            K[i][j] = np.exp(np.negative(np.sum(np.square(x-y))))
    print("finished {:.3f}s".format(time()-t))
    return K


def sample(X, Y, sample_size):
    """Takes in a dataset and its labels and returns a
    random sample of them according to the given sample_size

    Arguments:
        - X: ndarray (samplesxfeatures)
        - Y: ndarray (samplesx1)
        - sample_size: 0<float<=1
            Percentage of dataset to sample
    
    Returns:
        X and Y sampled according to sample_size
    """
    n_samples = X.shape[0]
    if n_samples != Y.shape[0]:
        raise ValueError("X and Y input sizes did not match")
    if sample_size < 1:
        sampleIndicies = random.sample(range(n_samples), int(n_samples*sample_size))
        X = np.array([x for i, x in enumerate(X) if i in sampleIndicies])
        Y = np.array([y for i, y in enumerate(Y) if i in sampleIndicies])
    return X, Y


def parseArgs(argv):
    """Processes command line arguments, currently the only one is the sample
    size to use for training, given as a percentage value in the range (0, 1]

    Arguments:
        argv: array-like
            The arguments obtained from sys.argv
    
    Returns:
        sampleSize: float
            The percentage of training samples to be used
    """
    sampleSize = 1
    if len(argv) > 1:
        try:
            temp = float(argv[1])

            if temp <= 0 or temp > 1:
                raise ValueError
            else:
                sampleSize = temp
        except ValueError:
            print("WARN: Invalid sample size, must be a decimal value in range (0, 1]. Using full sample set for this run.")

    return sampleSize


def preprocess(X, Y, C0, C1):
    """Takes in a dataset from keras.datasets.mnist in two arrays, 
    one with samples and the other with the labels at corresponding indices, 
    and applies preprocessing rules, including 

    Arguments:
        X: array-like (samplesx(2d features))
            Array of MNIST samples
        Y: array-like (1xsamples)
            Array of MNIST labels
        C0: array-like
            The labels belonging to class -1
        C1: array-like
            The labels belonging to class 1
    
    Returns:
        X: ndarray (samplesxfeatures)
            The preprocessed sample set as a NumPy array
        Y: ndarray (featuresx1)
            The preprocessed label set as a NumPy array
    """
    # Flatten the 2D representations of the samples into 1D arrays
    X = np.reshape(X, (len(X), len(X[0])**2))

    # Normalize sample values to be between 0 and 1
    X = np.divide(X, 256)

    # Replace the 0-9 class labels with -1 and 1 depending on which group the label is in
    Y = np.fromiter((-1 if y in C0 else 1 for y in Y), int)

    return np.array(X), Y.reshape(len(Y), 1)


def evaluate(labels, gold):
    """Takes in an array of predicted labels and the corresponding
    gold standard and calculates precision, recall, and accuracy.

    Arguments:
        labels: array-like (1D)
            The predicted labels, either -1 or 1
        gold: array-like (1D)
            The correct labels
    
    Returns:
        None
    """
    labels = list(labels)

    # Get confusion matrix, thanks scikit-learn!
    conf_matrix = metrics.confusion_matrix(gold, labels)
    correct = conf_matrix[0][0]+conf_matrix[1][1] # Total correct is true 0 + true 1
    conf_matrix = [[str(x)+"  " if x <= 9 else str(x)+" " if x <= 99 else str(x) for x in row] for row in conf_matrix]
    print("-----------------------------------------")
    print("|                         Actual        |")
    print("|          -----------------------------|")
    print("|          |      |    -1    |    +1    |")
    print("|          |------+----------+----------|")
    print("|          |  -1  |    {}   |    {}   |".format(conf_matrix[0][0], conf_matrix[0][1]))
    print("|Predicted |      |          |          |")
    print("|          |  +1  |    {}   |    {}   |".format(conf_matrix[1][0], conf_matrix[1][1]))
    print("-----------------------------------------")
    
    # Get precision/recall/f-measure for both classes with one method call, thanks scikit-learn!
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(gold, labels)
    print("Class -1 Precision: {:.3f}".format(precision[0]))
    print("Class -1 Recall: {:.3f}".format(recall[0]))
    print("Class -1 F-Measure: {:.3f}".format(fscore[0]))
    print("Class +1 Precision: {:.3f}".format(precision[1]))
    print("Class +1 Recall: {:.3f}".format(recall[1]))
    print("Class +1 F-Measure: {:.3f}".format(fscore[1]))
    print("Accuracy: {}/{} = {:.3f}%".format(correct, len(gold), correct/len(gold)*100))
