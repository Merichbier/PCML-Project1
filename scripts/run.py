# -*- coding: utf-8 -*-
""" Script file to run to obtain an exact submission
Authors: Victor Faramond, Dario Anongba Varela, Mathieu Schopfer
"""

# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *
from helpers import *

print('Script running... Please wait \n')

DATA_TRAIN_PATH = 'data/train.csv'
DATA_TEST_PATH = 'data/test.csv'

# Load the training data into feature matrix, class labels, and event ids:
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

# Pre-processing of input data
tX = process_data(tX)
tX, mean_tX, std_tX = standardize(tX)

# Ridge regression computation of the weights
# why we chose degree 7 is explained in the report
degree = 7

tX = build_poly(tX, degree)
tX = add_constant_column(tX)

# Why we chose lambda 0.01 is explained in the report
lambda_ = 0.01

weights, loss = ridge_regression(y, tX, lambda_)

# We give the name of the output file
OUTPUT_PATH = 'data/output_ridge_regression.csv'

# Load the test data into feature matrix, class labels, and event ids:
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# Process it
tX_test = process_data(tX_test)
tX_test, _, _ = standardize(tX_test, mean_tX, std_tX)
tX_test = build_poly(tX_test, degree)
tX_test = add_constant_column(tX_test)

# Generate predictions and save ouput in csv format for submission:
y_pred = predict_labels(weights, tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

print('Done !')
