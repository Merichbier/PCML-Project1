# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Stochastic Gradient Descent
"""
import costs
from helpers import batch_iter


def compute_stoch_gradient(y, tx, w):
    e = y - np.dot(tx, w)
    n = len(y)
    
    return -np.dot(tx.T, e) / n


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_epochs, gamma):    
    
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter, [mb_y, mb_tx] in enumerate(batch_iter(y, tx, batch_size, max_epochs)):
        # compute gradient and loss
        gradient = compute_stoch_gradient(mb_y, mb_tx, w)
        loss = compute_loss(y, tx, w)
        
        # update w by gradient
        w -= gamma * gradient
        
        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)

    return losses, ws