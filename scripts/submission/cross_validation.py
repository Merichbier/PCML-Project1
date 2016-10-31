# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from proj1_helpers import predict_labels, compute_accuracy
from helpers import process_data


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, regression_method, **args):
    """ Completes k-fold cross-validation using the regression method passed as argument
    """
    # get k'th subgroup in test, others in train
    msk_test = k_indices[k]
    msk_train = np.delete(k_indices, (k), axis=0).ravel()

    x_train = x[msk_train, :]
    x_test = x[msk_test, :]
    y_train = y[msk_train]
    y_test = y[msk_test]

    # compute weights using gradient descent
    weights, loss = regression_method(y=y_train, tx=x_train, **args)

    # calculate the accuracy for train and test data
    y_train_pred = predict_labels(weights, x_train)
    acc_train = compute_accuracy(y_train_pred, y_train)

    y_test_pred = predict_labels(weights, x_test)
    acc_test = compute_accuracy(y_test_pred, y_test)

    return acc_train, acc_test


def cross_validation_visualization(lambds, acc_train, acc_test):
    """visualization the curves of acc_train and acc_test."""
    plt.semilogx(lambds, acc_train, marker=".", color='b', label='train error')
    plt.semilogx(lambds, acc_test, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("accuracy")
    plt.title("cross validation")
    plt.legend(loc=10)
    plt.grid(True)