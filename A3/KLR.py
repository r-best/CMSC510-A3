"""
Robert Best
CMSC 510
Assignment 3
Due 12/3/18
"""
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import random
import sys
from pprint import pprint

from . import utils


def getK(X, Y):
    print(X.shape[0], Y.shape[0])
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            K[i][j] = np.exp(np.negative(np.sum(np.square(x-y))))
    return np.array(K)


def train(K, y_train, epochs=100, delta=0.0001):
    """Training function, takes in a training set and its labels and uses gradient descent w/
    logistic loss to calculate feature weights and bias for a classifier

    Arguments:
        x_train: ndarray (samplesxfeatures)
            The training set
        y_train: ndarray (samplesx1)
            The labels of the training set
        epochs: int, default 100
            Number of training iterations
        batch_size: int, default 128
            Number of samples to process at a time in each epoch
        a: float, default 0.1
            Gradient descent change parameter
        prox_const: float, default 0.00001
            Threshold value for soft thresholding
    
    Returns
        w: ndarray (featuresx1)
            The calculated feature weights after training
        b: float
            The calculated bias after training
    """
    n_samples = K.shape[0]

    K = tf.constant(K, name="K") # TF CONSTANT

    a = tf.Variable(np.random.rand(n_samples, 1).astype(dtype='float64'), name="w") # Gaussian thing (samplesx1)
    b = tf.Variable(0.0, dtype=tf.float64, name="b") # Bias offset (scalar)

    x = tf.placeholder(dtype=tf.float64, name='x') # Training set (featuresxsamples)
    y = tf.placeholder(dtype=tf.float64, name='y') # Training set labels (samplesx1)

    l = lambda i: tf.log(1 + tf.exp(
        tf.reduce_sum(
            tf.multiply(tf.multiply(a, y), tf.reshape(K[i], [-1, 1]))
        ) + b
    ))
    _, hell = tf.while_loop(
        lambda i, s: tf.less(i, n_samples),
        lambda i, s: (
            i+1,
            s + a[i][0] * tf.reduce_sum(
                tf.multiply(a, tf.reshape(K[i], [-1, 1]))
            )
        ),
        [tf.constant(0, name="hell_i"), tf.constant(0, dtype=tf.float64, name="hell_s")],
        return_same_structure=True
    )

    _, loss = tf.while_loop(
        lambda i, s: tf.less(i, n_samples),
        lambda i, s: (
            i+1,
            s + l(i) + hell
        ),
        [tf.constant(0, name="loss_i"), tf.constant(0, dtype=tf.float64, name="loss_s")],
        return_same_structure=True
    )

    train = tf.train.GradientDescentOptimizer(delta).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            print("Epoch {}".format(i))
            sess.run([train], feed_dict={y: y_train})
            print(sess.run(loss, feed_dict={y: y_train}))
            # print(curr_a, curr_b)
        curr_a,curr_b = sess.run([a,b])

    return curr_a, curr_b


def predict(a, b, K, test):
    """
    Uses given feature weights and bias to classify
    samples in the given test set and yields their labels

    Arguments:
        w: ndarray (featuresx1)
            Column matrix of feature weights
        b: float
            Y-offset value of classifier line
        test: ndarray (samplesxfeatures)
            Test set
    
    Yields:
        Predicted labels for the samples of the test set
    """
    labels = list()
    ak = np.multiply(np.multiply(a, np.ones(K.shape[1])), K).T
    print(ak)
    for row in ak:
        x = np.sum(row)
        print(x)
        if x < 0:
            labels.append(1)
        else:
            labels.append(-1)
    return labels


def main(argv):
    C0 = [0, 1, 2, 3, 4]
    C1 = [5, 6, 7, 8, 9]

    # Read args from command line
    sampleSize_train = utils.parseArgs(argv)
    print(sampleSize_train)

    # Load the train and test sets from MNIST
    print("Loading datasets from MNIST...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Sample training set
    if sampleSize_train < 1:
        sampleIndicies_train = random.sample(range(len(x_train)), int(len(x_train)*sampleSize_train))
        x_train = np.array([_ for i, _ in enumerate(x_train) if i in sampleIndicies_train])
        y_train = np.array([_ for i, _ in enumerate(y_train) if i in sampleIndicies_train])

    sampleSize_test = 0.2
    sampleIndicies_test = random.sample(range(len(x_test)), int(len(x_test)*sampleSize_test))
    x_test = np.array([_ for i, _ in enumerate(x_test) if i in sampleIndicies_test])
    y_test = np.array([_ for i, _ in enumerate(y_test) if i in sampleIndicies_test])

    # Apply preprocessing to the training and test sets
    print("Preprocessing training set...")
    x_train, y_train = utils.preprocess(x_train, y_train, C0, C1)
    print("Preprocessing testing set...")
    x_test, y_test = utils.preprocess(x_test, y_test, C0, C1)

    K = getK(x_train, x_train)

    print("Training model...")
    a, b = train(K, y_train)

    print("Evaluating model...")
    labels = predict(a, b, getK(x_train, x_test), x_test)

    print("Calculating metrics...")
    utils.evaluate(labels, y_test)

if __name__ == '__main__':
    main(sys.argv)
