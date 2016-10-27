# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_visualization(lambds, acc_train, acc_test):
    """visualization the curves of acc_train and acc_test."""
    plt.semilogx(lambds, acc_train, marker=".", color='b', label='train error')
    plt.semilogx(lambds, acc_test, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("accuracy")
    plt.title("cross validation")
    plt.legend(loc=10)
    plt.grid(True)