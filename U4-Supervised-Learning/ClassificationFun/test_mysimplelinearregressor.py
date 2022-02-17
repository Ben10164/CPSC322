import numpy as np
from mysimplelinearregressor import MySimpleLinearRegressor
from sklearn.linear_model import LinearRegression

# with pytest, test modules and test functions start with test_


def test_mysiplelinearregressor_fit():
    np.random.seed(0)
    # we add 1 or more test cases
    # start with a simple/common test case (below)
    X_train = [[val] for val in range(100)]  # 2D
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]  # 1D
    lin_reg = MySimpleLinearRegressor()
    lin_reg.fit(X_train, y_train)

    # check with "desk calculation"
    slope_solution = 1.924917458430444
    intercept_solution = 5.211786196055144
    # assert against the solution
    assert np.isclose(lin_reg.slope, slope_solution)
    assert np.isclose(lin_reg.intercept, intercept_solution)

    # then check with a valuidated library
    sklearn_lin_reg = LinearRegression()
    sklearn_lin_reg.fit(X_train, y_train)
    assert np.isclose(lin_reg.slope, sklearn_lin_reg.coef_[0])
    assert np.isclose(lin_reg.intercept, sklearn_lin_reg.intercept_)

    # then do edge cases (like an empty list)


def test_mysiplelinearregressor_predict():
    np.random.seed(0)
    X_train = [[val] for val in range(100)]  # 2D
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]  # 1D
    X_test = [[-150], [0], [50], [150], [1000]]
    lin_reg = MySimpleLinearRegressor()
    lin_reg.fit(X_train, y_train)
    y_predicted = lin_reg.predict(X_test)

    # then check with a valuidated library
    sklearn_lin_reg = LinearRegression()
    sklearn_lin_reg.fit(X_train, y_train)
    sklearn_y_predicted = sklearn_lin_reg.predict(X_test)
    assert np.allclose(y_predicted, sklearn_y_predicted)
