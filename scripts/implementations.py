# -*- coding: utf-8 -*-
"""
Project 1 method implementations.
Authors: Victor Faramond, Dario Anongba Varela, Mathieu Schopfer
"""

import numpy as np
from costs import *
from helpers import *


def least_squares_gd(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using gradient descent
    """
    # Define parameters to store weight and loss
    loss = 0
    w = initial_w

    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)

        # update w by gradient
        w -= gamma * gradient

    return w, loss


def least_squares_sgd(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using stochastic gradient descent
    """
    # Define parameters of the algorithm
    batch_size = 1

    # Define parameters to store w and loss
    loss = 0
    w = initial_w

    for n_iter, [mb_y, mb_tx] in enumerate(batch_iter(y, tx, batch_size, max_iters)):
        # compute gradient and loss
        gradient = compute_gradient(mb_y, mb_tx, w)
        loss = compute_loss(y, tx, w)

        # update w by gradient
        w -= gamma * gradient

    return w, loss


def least_squares(y, tx):
    """ Least squares regression using normal equations
    """
    x_t = tx.T

    w = np.dot(np.dot(np.linalg.inv(np.dot(x_t, tx)), x_t), y)
    loss = compute_loss(y, tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """ Ridge regression using normal equations
    """
    x_t = tx.T
    lambd = lambda_ * 2 * len(y)

    w = np.dot(np.dot(np.linalg.inv(np.dot(x_t, tx) + lambd * np.eye(tx.shape[1])), x_t), y)
    loss = compute_loss(y, tx, w)

    return w, loss


def logistic_regression(y, tx, gamma, max_iters):
    # init parameters
    threshold = 1e-8
    losses = []
    loss = 0

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for i in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_newton_method(y, tx, w, gamma)
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, loss


def reg_logistic_regression(y, tx, lambda_, gamma, max_iters):
    # init parameters
    threshold = 1e-8
    losses = []
    loss = 0

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for i in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, loss
