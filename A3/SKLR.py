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


def getK(X, Y):
    print("\tBuilding K ({}x{})".format(X.shape[0], Y.shape[0]))
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        # print("\t{}/{}".format(i, X.shape[0]))
        for j, y in enumerate(Y):
            K[i][j] = np.exp(np.negative(np.sum(np.square(x-y))))
    return np.array(K)


def getL(X):
    g = lambda i, j, t=1: np.exp(np.negative(np.sum(np.square(X[i]-X[j]))/t))

    n = X.shape[0]
    A = np.zeros((n, n))
    D = np.zeros((n, n))
    for i, _ in enumerate(X):
        D[i][i] = np.sum([g(i, k) for k in range(n)])
        for j, _ in enumerate(X):
            A[i][j] = g(i, j)
    return D - A


def train(x_train, y_train, supervised=0.10, epochs=500, delta=0.01):
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
    m = x_train.shape[0]
    k = int(m*supervised)
    for i in range(k, m):
        y_train[i] = 0
    print("# of total samples: {}".format(m))
    print("# of supervised samples: {}".format(k))

    K = tf.constant(getK(x_train, x_train), name="K")
    L = tf.constant(getL(x_train), name="L")

    a = tf.Variable(np.random.rand(m, 1).astype(dtype='float64'), name="c") # Gaussian thing (samplesx1)
    b = tf.Variable(0.0, dtype=tf.float64, name="b") # Bias offset (scalar)

    y = tf.placeholder(dtype=tf.float64, name='y', shape=[m, 1]) # Training set labels (samplesx1)

    l = lambda i: tf.log(1 + tf.exp(
        tf.negative(y[i][0]) * (
        tf.reduce_sum(
            tf.multiply(a*y, tf.reshape(K[i], [-1, 1]))
        ) + b)
    ))
    hell1 = 0.5*tf.matmul(
        tf.transpose(a*y),
        tf.matmul(K, a*y)
    )[0][0]
    hell2 = tf.matmul(tf.transpose(a*y),
        tf.matmul(K,
            tf.matmul(L,
                tf.matmul(K, a*y)
    )))[0][0]

    _, loss = tf.while_loop(
        lambda i, s: tf.less(i, k),
        lambda i, s: (
            i+1,
            s + l(i) + hell1 + hell2
        ),
        [tf.constant(0, name="loss_i"), tf.constant(0, dtype=tf.float64, name="loss_s")]
    )

    train = tf.train.GradientDescentOptimizer(delta).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            print("Epoch {}".format(i))
            sess.run([train], feed_dict={y: y_train})
            print(sess.run(loss, feed_dict={y: y_train}))
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
    # print(ak)
    for row in ak:
        x = np.sum(row)
        # print(x)
        if x < 0:
            labels.append(1)
        else:
            labels.append(-1)
    return labels


def sample(X, Y, sample_size):
    n_samples = X.shape[0]
    if n_samples != Y.shape[0]:
        raise ValueError()
    if sample_size < 1:
        sampleIndicies = random.sample(range(n_samples), int(n_samples*sample_size))
        X = np.array([x for i, x in enumerate(X) if i in sampleIndicies])
        Y = np.array([y for i, y in enumerate(Y) if i in sampleIndicies])
    return X, Y


def main(argv):
    C0 = [0, 1, 2, 3, 4]
    C1 = [5, 6, 7, 8, 9]

    # Read args from command line
    SAMPLE_TRAIN = utils.parseArgs(argv)
    SAMPLE_TEST = 0.2
    SUPERV_RATE = 0.10

    # Load the train and test sets from MNIST
    print("Loading datasets from MNIST...", end='', flush=True); t = time()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("finished {:.3f}s".format(time()-t))

    # Sample datasets
    print("Sampling training set...", end='', flush=True); t = time()
    x_train, y_train = sample(x_train, y_train, SAMPLE_TRAIN)
    print("finished {:.3f}s".format(time()-t))
    print("Sampling testing set...", end='', flush=True); t = time()
    x_test, y_test = sample(x_test, y_test, SAMPLE_TEST)
    print("finished {:.3f}s".format(time()-t))

    # Apply preprocessing to the training and test sets
    print("Preprocessing training set...", end='', flush=True); t = time()
    x_train, y_train = utils.preprocess(x_train, y_train, C0, C1)
    print("finished {:.3f}s".format(time()-t))
    print("Preprocessing testing set...", end='', flush=True); t = time()
    x_test, y_test = utils.preprocess(x_test, y_test, C0, C1)
    print("finished {:.3f}s".format(time()-t))

    print("Training model...")
    a, b = train(x_train, y_train, SUPERV_RATE)

    print("Evaluating model...")
    labels = predict(a, b, getK(x_train, x_test), x_test)

    print("Calculating metrics...")
    utils.evaluate(labels, y_test)

if __name__ == '__main__':
    main(sys.argv)
