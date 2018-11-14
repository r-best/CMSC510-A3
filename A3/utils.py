"""
Robert Best
CMSC 510
Assignment 2
Due 11/12/18
"""
import numpy as np
from sklearn import metrics


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
    X = [[x/256 for x in sample] for sample in X]

    # Replace the 0-9 class labels with -1 and 1 depending on which group the label is in
    Y = np.fromiter((-1 if y in C0 else 1 for y in Y))

    return X, Y.reshape(len(Y), 1)


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
