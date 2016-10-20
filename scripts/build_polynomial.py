# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    matrix = np.zeros([len(x), degree+1])
    
    for i in range(len(x)):
        for j in range(degree+1):
            matrix[i, j] = x[i]**j
    
    return matrix
