# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""
import numpy as np
import costs


def compute_gradient(y, tx, w):
    e = y - np.dot(tx, w)
    n = len(y)
    
    return -np.dot(tx.T, e) / n

	
def gradient_descent(y, tx, initial_w, max_iters, gamma): 
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
	
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        
        # update w by gradient
        w -= gamma * gradient
        
        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)
        
    return losses, ws
