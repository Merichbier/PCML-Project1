# -*- coding: utf-8 -*-
""" Script file to run to obtain an exact submission
Authors: Victor Faramond, Dario Anongba Varela, Mathieu Schopfer
"""

# Useful starting lines
from proj1_helpers import load_csv_data, predict_labels, create_csv_submission
from implementations import ridge_regression
from helpers import build_poly, process_data, add_constant_column

print('Script running... Please wait \n')

DATA_TRAIN_PATH = 'data/train.csv'
DATA_TEST_PATH = 'data/test.csv'

# Load the training data into feature matrix, class labels, and event ids:
y, tX_train, ids = load_csv_data(DATA_TRAIN_PATH)

# Load the test data into feature matrix, class labels, and event ids:
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# Pre-processing of data
tX_train, tX_test = process_data(tX_train, tX_test, False)

# The choice of degree is explained in the report
degree = 7

phi_train = build_poly(tX_train, degree)
phi_test = build_poly(tX_test, degree)

phi_train = add_constant_column(phi_train)
phi_test = add_constant_column(phi_test)

# The choice of lambda is explained in the report
lambda_ = 0.01

# We compute the weights using ridge regression
weights, loss = ridge_regression(y, phi_train, lambda_)

# We give the name of the output file
OUTPUT_PATH = 'data/output_ridge_regression.csv'

# Generate predictions and save ouput in csv format for submission:
y_pred = predict_labels(weights, phi_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

print('Done !')
