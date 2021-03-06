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
from time import time
from pprint import pprint

from . import utils


def train(x_train, y_train, epochs=100, delta=0.0001):
    """Training function, takes in a training set and its labels and uses gradient descent w/
    logistic loss to calculate feature weights and bias for a classifier

    Arguments:
        x_train: ndarray (samplesxfeatures)
            The training set
        y_train: ndarray (samplesx1)
            The labels of the training set
        epochs: int, default 100
            Number of training iterations
        delta: float, default 0.001
            Gradient descent change parameter
    
    Returns
        a: ndarray (samplesx1)
            The calculated alpha values after training
        b: float
            The calculated bias after training
    """
    n_samples = x_train.shape[0]

    K = tf.constant(utils.getK(x_train, x_train), name="K") # TF CONSTANT

    a = tf.Variable(np.random.rand(n_samples, 1).astype(dtype='float64'), name="w") # Gaussian thing (samplesx1)
    b = tf.Variable(0.0, dtype=tf.float64, name="b") # Bias offset (scalar)

    y = tf.placeholder(dtype=tf.float64, name='y', shape=[n_samples, 1]) # Training set labels (samplesx1)

    # The first part of the formula
    l = lambda i: tf.log(1 + tf.exp(
        tf.reduce_sum(
            tf.multiply(tf.multiply(a, y), tf.reshape(K[i], [-1, 1]))
        ) + b
    ))
    # The second part of the formula, managed to reduce the double summation
    # down to an equation with a bunch of element-wise multiplication
    term = tf.reduce_sum(tf.multiply(
        tf.multiply(a, y),
        tf.multiply(
            tf.multiply(a, np.ones((1, a.shape[0]))),
            tf.multiply(tf.multiply(y, np.ones((1, y.shape[0]))), K)
        )
    ))

    # Generate the summation of l+term over all i
    _, loss = tf.while_loop(
        lambda i, s: tf.less(i, n_samples),
        lambda i, s: (
            i+1,
            s + l(i) + term
        ),
        [tf.constant(0, name="loss_i"), tf.constant(0, dtype=tf.float64, name="loss_s")],
        return_same_structure=True
    )

    train = tf.train.GradientDescentOptimizer(delta).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            # print("Epoch {}".format(i))
            sess.run([train], feed_dict={y: y_train})
            # print(sess.run(loss, feed_dict={y: y_train}))
        curr_a,curr_b = sess.run([a,b])

    return curr_a, curr_b


def predict(a, b, train, test):
    """
    Uses given alpha weights and bias to classify
    samples in the given test set

    Arguments:
        a: ndarray (train_samplesx1)
            Column matrix of sample weights
        b: float
            Y-offset value of classifier line
        train: ndarray (train_samplesxfeatures)
            Training set 
        test: ndarray (test_samplesxfeatures)
            Test set
    
    Returns:
        Predicted labels for the samples of the test set
    """
    K = utils.getK(train, test)
    ak = np.multiply(np.multiply(a, np.ones(K.shape[1])), K).T
    labels = list()
    for row in ak:
        pred = np.average(row)
        if pred < 0:
            labels.append(1)
        else:
            labels.append(-1)
    return labels


def main(argv):
    t0 = time()

    C0 = [0, 1, 2, 3, 4]
    C1 = [5, 6, 7, 8, 9]

    # Read args from command line/establish constants
    SAMPLE_TRAIN = utils.parseArgs(argv)
    SAMPLE_TEST = 0.2

    # Load the train and test sets from MNIST
    print("Loading datasets from MNIST...", end='', flush=True); t = time()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("finished {:.3f}s".format(time()-t))

    # Sample datasets
    print("Sampling training set...", end='', flush=True); t = time()
    x_train, y_train = utils.sample(x_train, y_train, SAMPLE_TRAIN)
    print("finished {:.3f}s".format(time()-t))
    print("Sampling testing set...", end='', flush=True); t = time()
    x_test, y_test = utils.sample(x_test, y_test, SAMPLE_TEST)
    print("finished {:.3f}s".format(time()-t))

    # Apply preprocessing to the training and test sets
    print("Preprocessing training set...", end='', flush=True); t = time()
    x_train, y_train = utils.preprocess(x_train, y_train, C0, C1)
    print("finished {:.3f}s".format(time()-t))
    print("Preprocessing testing set...", end='', flush=True); t = time()
    x_test, y_test = utils.preprocess(x_test, y_test, C0, C1)
    print("finished {:.3f}s".format(time()-t))

    # Train model
    print("Training model..."); t = time()
    a, b = train(x_train, y_train)
    print("Finished model training in {:.3f}s".format(time()-t))

    # Evaluate model on test set
    print("Evaluating model..."); t = time()
    labels = predict(a, b, x_train, x_test)
    print("Finished evaluating model in {:.3f}s".format(time()-t))

    # Calculate metrics
    print("Pipeline finished in {:.3f}s, calculating results...".format(time()-t0))
    return utils.evaluate(labels, y_test)

if __name__ == '__main__':
    main(sys.argv)
