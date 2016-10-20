# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lamb):
    tx_T = tx.T
    lambd = lamb * 2 * len(y)
    
    return np.dot(np.dot(np.linalg.inv(np.dot(tx_T, tx) + lambd * np.eye(tx.shape[1])), tx_T), y)