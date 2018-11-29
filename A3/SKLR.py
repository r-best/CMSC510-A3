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


def getL(X):
    print("\tBuilding L ({}x{})...".format(X.shape[0], X.shape[0]), end='', flush=True); t = time()
    g = lambda i, j, t=1: np.exp(np.negative(np.sum(np.square(X[i]-X[j]))/t))

    n = X.shape[0]
    checkpoints = np.multiply(range(1, 11), n/10)

    A = np.zeros((n, n))
    D = np.zeros((n, n))
    for i, _ in enumerate(X):
        if i in checkpoints: # Print progress
            print("{}%...".format((np.where(checkpoints==i)[0][0]+1)*10), end='', flush=True)
        D[i][i] = np.sum([g(i, k) for k in range(n)])
        for j, _ in enumerate(X):
            A[i][j] = g(i, j)
    print("finished {:.3f}s".format(time()-t))
    return D - A


def train(x_train, y_train, supervised=0.10, epochs=1000, delta=0.0001):
    """Training function, takes in a training set and its labels and uses gradient descent w/
    logistic loss to calculate feature weights and bias for a classifier

    Arguments:
        x_train: ndarray (samplesxfeatures)
            The training set
        y_train: ndarray (samplesx1)
            The labels of the training set
        epochs: int, default 100
            Number of training iterations
        delta: float, default 0.01
            Gradient descent change parameter
    
    Returns
        c: ndarray (featuresx1)
            The calculated sample weights after training
        b: float
            The calculated bias after training
    """
    m = x_train.shape[0]
    k = int(m*supervised)
    for i in range(k, m):
        y_train[i] = 0
    print("\t# of total samples: {}".format(m))
    print("\t# of supervised samples: {}".format(k))

    K = tf.constant(utils.getK(x_train, x_train), name="K")
    L = tf.constant(getL(x_train), name="L")

    c = tf.Variable(np.random.rand(m, 1).astype(dtype='float64'), name="c") # Gaussian thing (samplesx1)
    b = tf.Variable(0.0, dtype=tf.float64, name="b") # Bias offset (scalar)

    y = tf.placeholder(dtype=tf.float64, name='y', shape=[m, 1]) # Training set labels (samplesx1)

    l = lambda i: tf.log(1 + tf.exp(
        tf.negative(y[i][0]) * (
        tf.reduce_sum(
            tf.multiply(c, tf.reshape(K[i], [-1, 1]))
        ) + b)
    ))
    term1 = 0.5*tf.matmul(
        tf.transpose(c),
        tf.matmul(K, c)
    )[0][0]
    term2 = tf.matmul(tf.transpose(c),
        tf.matmul(K,
            tf.matmul(L,
                tf.matmul(K, c)
    )))[0][0]

    _, loss = tf.while_loop(
        lambda i, s: tf.less(i, k),
        lambda i, s: (
            i+1,
            s + l(i) + term1 + term2
        ),
        [tf.constant(0, name="loss_i"), tf.constant(0, dtype=tf.float64, name="loss_s")]
    )

    train = tf.train.GradientDescentOptimizer(delta).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            # print("Epoch {}".format(i))
            sess.run([train], feed_dict={y: y_train})
            # print(sess.run(loss, feed_dict={y: y_train}))
        curr_c,curr_b = sess.run([c,b])

    return curr_c, curr_b


def predict(c, b, train, test):
    """
    Uses given c and bias to classify samples
    in the given test set

    Arguments:
        c: ndarray (train_samplesx1)
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
    ck = np.multiply(np.multiply(c, np.ones(K.shape[1])), K).T
    labels = list()
    for row in ck:
        pred = np.average(row)
        if pred < 0:
            labels.append(-1)
        else:
            labels.append(1)
    return labels


def main(argv):
    t0 = time()

    C0 = [0, 1, 2, 3, 4]
    C1 = [5, 6, 7, 8, 9]

    # Read args from command line/establish constants
    SAMPLE_TRAIN = utils.parseArgs(argv)
    SAMPLE_TEST = 0.2
    SUPERV_RATE = 0.10

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
    c, b = train(x_train, y_train, SUPERV_RATE)
    print("Finished model training in {:.3f}s".format(time()-t))

    # Evaluate model on test set
    print("Evaluating model...")
    labels = predict(c, b, x_train, x_test); t = time()
    print("Finished evaluating model in {:.3f}s".format(time()-t))

    # Calculate metrics
    print("Pipeline finished in {:.3f}s, calculating results...".format(time()-t0))
    utils.evaluate(labels, y_test)


if __name__ == '__main__':
    main(sys.argv)
