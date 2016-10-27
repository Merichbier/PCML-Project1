# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
from costs import *


def compute_gradient(y, tx, w):
    """ Linear regression using gradient descent
    """
    e = y - tx.dot(w)
    n = len(y)

    return -np.dot(tx.T, e) / n


def sigmoid(t):
    """apply sigmoid function on t."""
    return 1 / (1 + np.exp(-t))


def compute_gradient_inv_log_likelihood(y, tx, w):
    """compute the gradient of the loss by inverse log likelihood."""
    return tx.T @ (sigmoid(tx @ w) - y)


def compute_hessian(y, tx, w):
    """return the hessian of the loss function."""
    s = np.zeros([len(y), len(y)])

    for i in range(len(y)):
        s[i, i] = sigmoid(tx[i] @ w) * (1 - sigmoid(tx[i] @ w))

    return tx.T @ s @ tx


def param_logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""
    return compute_loss_neg_log_likelihood(y, tx, w), compute_gradient_inv_log_likelihood(y, tx, w), compute_hessian(y, tx, w)


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""

    new_loss = compute_loss_neg_log_likelihood(y, tx, w) + lambda_ * np.sum(w**2)
    new_gradient = compute_gradient_inv_log_likelihood(y, tx, w) + 2 * lambda_ * np.sum(w)
    new_hessian = compute_hessian(y, tx, w) + 2 * lambda_ * len(y)

    return new_loss, new_gradient, new_hessian


def learning_by_newton_method(y, tx, w, gamma):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    loss, gradient, hessian = param_logistic_regression(y, tx, w)
    w -= gamma * np.dot(np.linalg.inv(hessian), gradient)

    return loss, w


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient, hessian = penalized_logistic_regression(y, tx, w, lambda_)
    w -= gamma * np.dot(np.linalg.inv(hessian), gradient)

    return loss, w


def standardize(x, mean_x=None, std_x=None):
    """ Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x > 0] = x[:, std_x > 0] / std_x[std_x > 0]
    
    tx = np.hstack((np.ones((x.shape[0], 1)), x))
    return tx, mean_x, std_x


def batch_iter(y, tx, batch_size, num_batches=None, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    num_batches_max = int(np.ceil(data_size/batch_size))
    if num_batches is None:
        num_batches = num_batches_max
    else:
        num_batches = min(num_batches, num_batches_max)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_poly(x, degree):
    """ Apply a polynomial basis to all the X features. """
    # First, we find the combinations of columns for which we have to
    # compute the product
    m, n = x.shape

    combinations = {}

    for i in range(n * degree):
        if i < n:
            combinations[i] = [i]
        else:
            col_number = i - n
            cpt = 2
            while col_number >= n:
                col_number -= n
                cpt += 1
            combinations[i] = [col_number] * cpt

    # Now we can fill a new matrix with the produts of the columns
    # numbers computed previously
    eval_poly = np.zeros(shape=(m, n * degree))
    for i, c in combinations.items():
        eval_poly[:, i] = x[:, c].prod(1)

    return eval_poly


def add_constant_column(x):
    """ Prepend a column of 1 to the matrix """
    return np.hstack((np.ones((x.shape[0], 1)), x))


def na(x):
    """ Identifies missing values """
    return np.any(x == -999)


def pos(x):
    return np.all(x > 0)


def impute_data(data):
    """ Replace missing values (NA) by the most frequent value of the column"""
    for i in range(data.shape[1]):
        # If NA values in column
        if na(data[:, i]) or na(data[:, i]):
            msk_train = (data[:, i] != -999.)
            # Replace NA values with most frequent value
            values, counts = np.unique(data[msk_train, i], return_counts=True)
            data[~msk_train, i] = values[np.argmax(counts)]

    return data


def process_data(data):
    # Impute missing data
    data = impute_data(data)

    inv_log_cols = [0, 2, 5, 7, 9, 10, 13, 16, 19, 21, 23, 26]

    # Create inverse log values of features which are positive in value.
    data_inv_log_cols = np.log(1 / (1 + data[:, inv_log_cols]))
    data = np.hstack((data, data_inv_log_cols))

    return data
