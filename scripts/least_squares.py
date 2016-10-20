# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    xT = tx.T
    return np.dot(np.dot(np.linalg.inv(np.dot(xT, tx)), xT), y)