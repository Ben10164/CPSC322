import numpy as np
from sklearn.linear_model import LinearRegression

from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from mysklearn.myclassifiers import MySimpleLinearRegressionClassifier,\
    MyKNeighborsClassifier,\
    MyDummyClassifier

# note: order is actual/received student value, expected/solution


def test_simple_linear_regression_classifier_fit():
    np.random.seed(0)

    # case 1

    # simple/common test case
    X_train = [[val] for val in range(100)]  # 2D
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]  # 1D

    def discretizer(x):
        return_list = []
        for val in x:
            if val >= 100:
                return_list.append('high')
            else:
                return_list.append('low')
        return return_list

    lin_reg = MySimpleLinearRegressionClassifier(discretizer)
    lin_reg.fit(X_train, y_train)

    # then check with a validated library
    sklearn_lin_reg = LinearRegression()
    sklearn_lin_reg.fit(X_train, y_train)

    assert np.allclose(lin_reg.regressor.slope,
                       sklearn_lin_reg.coef_)  # test the slope
    assert np.allclose(lin_reg.regressor.intercept,
                       sklearn_lin_reg.intercept_)  # test the intercept

    # case 2

    X_train = [[val] for val in range(500)]  # 2D
    y_train = [row[0] * 2 + np.random.normal(0, 50) for row in X_train]  # 1D

    def discretizer(x):
        return_list = []
        for val in x:
            if val >= 250:
                return_list.append('high')
            else:
                return_list.append('low')
        return return_list

    lin_reg = MySimpleLinearRegressionClassifier(discretizer)
    lin_reg.fit(X_train, y_train)

    # then check with a valuidated library
    sklearn_lin_reg = LinearRegression()
    sklearn_lin_reg.fit(X_train, y_train)

    assert np.allclose(lin_reg.regressor.slope,
                       sklearn_lin_reg.coef_)  # test the slope
    assert np.allclose(lin_reg.regressor.intercept,
                       sklearn_lin_reg.intercept_)  # test the intercept


def test_simple_linear_regression_classifier_predict():
    np.random.seed(0)

    # case 1

    # simple/common test case
    X_train = [[val] for val in range(100)]  # 2D
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]  # 1D

    def discretizer(x):
        return_list = []
        for val in x:
            if val >= 100:
                return_list.append('high')
            else:
                return_list.append('low')
        return return_list

    X_test = [[0], [50], [99], [101], [1000]]

    lin_reg = MySimpleLinearRegressionClassifier(discretizer)
    lin_reg.fit(X_train, y_train)
    y_predicted = lin_reg.predict(X_test)

    # then check with a valuidated library
    sklearn_lin_reg = LinearRegression()
    sklearn_lin_reg.fit(X_train, y_train)
    sklearn_y_predicted = sklearn_lin_reg.predict(X_test)

    sklearn_y_predicted_classified = [
        'high' if val >= 100 else 'low' for val in sklearn_y_predicted]

    assert np.array_equal(y_predicted, sklearn_y_predicted_classified)

    # case 2

    # simple/common test case
    X_train = [[val] for val in range(500)]  # 2D
    y_train = [row[0] * 2 + np.random.normal(0, 50) for row in X_train]  # 1D

    def discretizer(x):
        return_list = []
        for val in x:
            if val >= 250:
                return_list.append('high')
            else:
                return_list.append('low')
        return return_list

    X_test = [[250], [-10], [200], [500], [0]]

    lin_reg = MySimpleLinearRegressionClassifier(discretizer)
    lin_reg.fit(X_train, y_train)
    y_predicted = lin_reg.predict(X_test)

    # then check with a valuidated library
    sklearn_lin_reg = LinearRegression()
    sklearn_lin_reg.fit(X_train, y_train)
    sklearn_y_predicted = sklearn_lin_reg.predict(X_test)

    sklearn_y_predicted_classified = [
        'high' if val >= 250 else 'low' for val in sklearn_y_predicted]

    assert np.array_equal(y_predicted, sklearn_y_predicted_classified)


def test_kneighbors_classifier_kneighbors():

    # from in-class #1  (4 instances)
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    knn_clf = MyKNeighborsClassifier(n_neighbors=3)  # n_neighbors=k=3
    knn_clf.fit(X_train_class_example1, y_train_class_example1)
    X_test = [3, 2]  # TODO: GET FROM SLIDE
    knn_kneighbors = knn_clf.kneighbors(X_test)

    desk_solution = []  # BUG: GET FROM SLIDES

    print(knn_kneighbors)
    print(desk_solution)
    # assert np.allclose(knn_kneighbors, desk_solution)

    # from in-class #2 (8 instances)
    # assume normalized
    X_train_class_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]

    y_train_class_example2 = ["no", "yes",
                              "no", "no", "yes", "no", "yes", "yes"]

    knn_clf = MyKNeighborsClassifier(n_neighbors=3)  # n_neighbors=k=3
    knn_clf.fit(X_train_class_example2, y_train_class_example2)
    X_test = [2, 3]

    knn_kneighbors = knn_clf.kneighbors(X_test)

    # inclass solution
    inclass_sol = [[0, 1.4142135623730951],
                   [4, 1.4142135623730951],
                   [6, 2.0]]
    inclass_sol = ([1.4142135623730951, 1.4142135623730951, 2.0], [0, 4, 6])

    print(knn_kneighbors)
    print(inclass_sol)
    assert np.array_equal(knn_kneighbors, inclass_sol)

    # from Bramer
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",
                              "-", "-", "+", "+", "+", "-", "+"]

    knn_clf = MyKNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_train_bramer_example, y_train_bramer_example)

    X_test = [9.1, 11.0]

    knn_kneighbors = knn_clf.kneighbors(X_test)

    bramer_solution = ([0.6082762530298216, 1.2369316876852974, 2.202271554554525,
                       2.8017851452243794, 2.9154759474226513], [6, 5, 7, 4, 8])
    """
        9.2 11.6 0.608 6
        8.8 9.8 1.237 5
        10.8 9.6 2.202 7
        6.8 12.6 2.802 4
        11.8 9.9 2.915 8
        """

    print(knn_kneighbors)
    print(bramer_solution)
    assert np.array_equal(knn_kneighbors, bramer_solution)


def test_kneighbors_classifier_predict():
    # from in-class #1  (4 instances)
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    knn_clf = MyKNeighborsClassifier(n_neighbors=3)  # n_neighbors=k=3
    knn_clf.fit(X_train_class_example1, y_train_class_example1)
    X_test = [3, 2]  # TODO: GET FROM SLIDE

    knn_predict = knn_clf.predict(X_test)

    desk_solution = ["bad"]  # TODO: GET FROM SLIDES

    assert np.array_equal(knn_predict, desk_solution)

    # from in-class #2 (8 instances)
    # assume normalized
    X_train_class_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]

    y_train_class_example2 = ["no", "yes",
                              "no", "no", "yes", "no", "yes", "yes"]

    knn_clf = MyKNeighborsClassifier(n_neighbors=3)  # n_neighbors=k=3
    knn_clf.fit(X_train_class_example2, y_train_class_example2)
    X_test = [2, 3]

    knn_predict = knn_clf.predict(X_test)

    # inclass solution
    inclass_sol = ["no"]  # TODO: THIS

    assert np.array_equal(knn_predict, inclass_sol)

    # from Bramer
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",
                              "-", "-", "+", "+", "+", "-", "+"]

    knn_clf = MyKNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_train_bramer_example, y_train_bramer_example)

    X_test = [9.1, 11.0]

    knn_predict = knn_clf.predict(X_test)

    bramer_solution = ["-"]  # TODO: MAYBE
    """
        9.2 11.6 0.608 6
        8.8 9.8 1.237 5
        10.8 9.6 2.202 7
        6.8 12.6 2.802 4
        11.8 9.9 2.915 8
        """

    assert np.array_equal(knn_predict, bramer_solution)


def test_dummy_classifier_fit():
    # case 1
    # a list with 100 instances
    X_train = [i for i in range(100)]
    y_train = list(np.random.choice(
        ["yes", "no"], 100, replace=True, p=[0.7, 0.3]))

    dummy_cls = MyDummyClassifier()
    dummy_cls.fit(X_train, y_train)

    assert dummy_cls.most_common_label == "yes"

    # case 2
    y_train = list(np.random.choice(
        ["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))

    dummy_cls = MyDummyClassifier()
    dummy_cls.fit(X_train, y_train)

    assert dummy_cls.most_common_label == "no"

    # case 3
    y_train = list(np.random.choice(
        ["Yes", "No", "Maybe", "Undecided", "Unanswered"], 100, replace=True, p=[0.1, 0.2, 0.1, 0.2, 0.4]))
    dummy_cls = MyDummyClassifier()
    dummy_cls.fit(X_train, y_train)

    assert dummy_cls.most_common_label == "Unanswered"


def test_dummy_classifier_predict():
    # case 1
    # a list with 100 instances
    X_train = [i for i in range(100)]
    y_train = list(np.random.choice(
        ["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    X_test = [10, 20]

    dummy_cls = MyDummyClassifier()
    dummy_cls.fit(X_train, y_train)
    predicted = dummy_cls.predict(X_test)
    desk_calc = ["yes", "yes"]
    assert np.array_equal(predicted, desk_calc)

    # case 2
    y_train = list(np.random.choice(
        ["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))

    dummy_cls = MyDummyClassifier()
    dummy_cls.fit(X_train, y_train)
    predicted = dummy_cls.predict(X_test)
    desk_calc = ["no", "no"]
    assert np.array_equal(predicted, desk_calc)

    # case 3
    y_train = list(np.random.choice(
        ["Yes", "No", "Maybe", "Undecided", "Unanswered"], 100, replace=True, p=[0.1, 0.2, 0.1, 0.2, 0.4]))
    dummy_cls = MyDummyClassifier()
    dummy_cls.fit(X_train, y_train)
    predicted = dummy_cls.predict(X_test)
    desk_calc = ["Unanswered", "Unanswered"]
    assert np.array_equal(predicted, desk_calc)
