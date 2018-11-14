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

from . import utils


def train(x_train, y_train, epochs=100, batch_size=128, a=0.1, prox_const=0.00001):
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
    n_samples, n_features = x_train.shape
    
    w = tf.Variable(np.random.rand(n_features, 1).astype(dtype='float64'), name="w") # Feature weights (featuresx1)
    b = tf.Variable(0.0, dtype=tf.float64, name="b") # Bias offset (scalar)

    x = tf.placeholder(dtype=tf.float64, name='x') # Training set (featuresxsamples)
    y = tf.placeholder(dtype=tf.float64, name='y') # Training set labels (samplesx1)

    predictions = tf.matmul(x, w) + b
    loss = tf.reduce_mean(
        tf.log(1 + tf.exp(
            tf.multiply(-1.0*y, predictions)
        ))
    )

    train = tf.train.GradientDescentOptimizer(a).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for _ in range(epochs):
            # Train model on batches of training set
            for i in range(0,n_samples,batch_size):
                iE = min(n_samples, i+batch_size)
                x_batch = x_train[i:iE,:]
                y_batch = y_train[i:iE,:]
                sess.run([train],feed_dict={x: x_batch, y: y_batch})

            # training done in this epoch, get current values of w and b
            curr_w,curr_b = sess.run([w,b])

            # Soft thresholding
            for i in range(len(curr_w)):
                if curr_w[i][0] < prox_const*-1:
                    curr_w[i][0] += prox_const
                elif curr_w[i][0] > prox_const:
                    curr_w[i][0] -= prox_const
                else:
                    curr_w[i][0] = 0
            sess.run([tf.assign(w, curr_w)])

    return curr_w, curr_b


def predict(w, b, test):
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
    for item in test:
        item = np.atleast_2d(item)
        u = np.matmul(item,w) + b
        if u < 0:
            yield -1
        elif u > 0:
            yield 1


def main(argv):
    C0 = [0, 1, 2, 3, 4]
    C1 = [5, 6, 7, 8, 9]

    # Read args from command line
    sampleSize = utils.parseArgs(argv)

    # Load the train and test sets from MNIST
    print("Loading datasets from MNIST...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Apply preprocessing to the training and test sets
    print("Preprocessing training set...")
    x_train, y_train = utils.preprocess(x_train, y_train, C0, C1)
    print("Preprocessing testing set...")
    x_test, y_test = utils.preprocess(x_test, y_test, C0, C1)

    # Sample training set
    sampleIndicies = random.sample(range(len(x_train)), int(len(x_train)*sampleSize))
    x_train_sample = np.array([_ for i, _ in enumerate(x_train) if i in sampleIndicies])
    y_train_sample = np.array([_ for i, _ in enumerate(y_train) if i in sampleIndicies])

    print("Training model...")
    w, b = train(x_train_sample, y_train_sample)

    print("Evaluating model...")
    labels = predict(w, b, x_test)

    print("Calculating metrics...")
    utils.evaluate(labels, y_test)

if __name__ == '__main__':
    main(sys.argv)
