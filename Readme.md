## Principles of Machine Learning EPFL - Team #96
### Team members 
- Victor Faramond
- Dario Anongba Varela
- Mathieu Schopfer

This file explains the organisation and functions of the python scripts. For more information about the implementation, see the PDF report and the commented code.

### costs
Contain 3 different cost functions like:
- **calculate_mse**: Mean square error
- **calculate_mae**: Mean absolute error
- **compute_loss_neg_log_likelihood**: Negative log likelihood

### cross_validation
Contain helper methods for cross validation.
- **build_k_indices**: Builds k indices for k-fold cross validation
- **cross_validation_visualization**: Creates a plot showing the accuracy given a lambda value

### helpers
Contain multiple methods for data processing and utilitary methods necessary to achieve the regression methods:
- **standardize, buid_poly, add_constant_column, na, impute_data and process_data**: All the processing functions. See the report for explications about those functions.
- **compute_gradient**: Computes the gradient for gradient descent and stochastic gradient descent
- **batch_iter**: Generate a minibatch iterator for a dataset

### proj1_helpers
Contain functions used to load the data and generate a CSV submission file.

### implementations
Contain the 6 regression methods needed for this project
- **least_squares_gd**: Linear regression using gradient descent
- **least_squares_sgd**: Linear regression using stochastic gradient descent
- **least_squares**: Least squares regression using normal equations
- **ridge_regression**: Ridge regression using normal equations
- **logistic_regression**: using stochastic gradient descent
- **reg_logistic_regression**: Regularized logistic regression

### run
Script that generates the exact CSV file submitted on kaggle.

### project1.ipynb
Python notebook used for tests during this project.