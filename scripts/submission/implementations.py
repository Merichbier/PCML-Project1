# -*- coding: utf-8 -*-
"""
Project 1 method implementations.
Authors: Victor Faramond, Dario Anongba Varela, Mathieu Schopfer
"""

import numpy as np
from costs import compute_loss, compute_loss_neg_log_likelihood
from helpers import compute_gradient, batch_iter, sigmoid


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


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression"""
    w = initial_w
    sample_count = len(y)
    batch_size = 1000

    batch_count = int(sample_count / batch_size) * max_iters

    coef = gamma / batch_size
    for mini_y, mini_tx in batch_iter(y, tx, batch_size, batch_count, shuffle=False):
        grad = mini_tx.T @ (sigmoid(mini_tx @ w) - mini_y)
        w -= coef * grad

    return w, compute_loss_neg_log_likelihood(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression"""
    if lambda_ == 0:
        return logistic_regression(y, tx, initial_w, max_iters, gamma)

    w = initial_w
    sample_count = len(y)
    batch_size = 1000

    batch_count = int(sample_count / batch_size) * max_iters

    coef = gamma / batch_size
    for mini_y, mini_tx in batch_iter(y, tx, batch_size, batch_count, shuffle=False):
        grad = mini_tx.T @ (sigmoid(mini_tx @ w) - mini_y) - 2 * lambda_ * w
        w -= coef * grad

    return w, compute_loss_neg_log_likelihood(y, tx, w)
