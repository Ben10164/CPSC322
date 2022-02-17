from mysimplelinearregressor import MySimpleLinearRegressor
import numpy as np


def main():

    # staritng with PA4 and beyond
    # we ware going to implement ML algorithms
    # following the API design of commmon data science
    # ML libraries (e.g. sci-kit learn, tensorflow, etc...)

    # X: 2D feature matrix (feature AKA attribute)
    # this stores the features used to make predictions
    # the class attribute (y) is not stored in X
    # y: 1D target class vector (target AKA label, class attribute)
    # X and y are parallel
    # X0 is an instance and y0 is the label for that instance

    # each algorithm implemented as a class
    # each class will have the "public" API
    # (Application Programming Interface)
    # fit(X_train, y_train) -> None
    # typically you divide the data into training and test sets
    # X_train and y_train are parallel
    # X_test and y_test are parallel
    # fit(X_train, y_train) fits the modle/prepares the algorithm
    # using the training data (returns nothing)
    # predict(X_test) -> y_predicted(list)
    # predict() makes predictions, one for each instance in X_test
    # y_predicted is a list of the predicted labels for each of the values in X_test
    # X_test, y_test, y_predicted are parallel

    # we compare y_predicted and y_test to see
    # how well the algorithm/model did
    # regression: mean absolute error (average
    # of the absolute differences between pairs
    # in y_predicted and y_test)
    # classification: accuracy (number of matches
    # in y_predicted and y_test divided by total)

    np.random.seed(0)
    X_train = [[val] for val in range(100)]  # 2D
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]  # 1D
    # WHEN YOU SEE CAPITAL X, IT MUST BE IN A MATRIX (2D)
    X_test = [[150], [1000]]
    # now we make the MySimpleLienarRegressor object
    lin_reg = MySimpleLinearRegressor()
    lin_reg.fit(X_train, y_train)
    y_predict = lin_reg.predict(X_test)
    print(y_predict)
    # now, lets conver thtis example into unit tests


if __name__ == '__main__':
    main()
