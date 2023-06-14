import numpy as np

from sklearn.linear_model import LinearRegression
from mysklearn.myclassifiers import (
    MyDecisionTreeClassifier,
    MyNaiveBayesClassifier,
    MySimpleLinearRegressionClassifier,
    MyKNeighborsClassifier,
    MyDummyClassifier,
)
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor

# TODO: copy your test_myclassifiers.py solution from PA4-6 here


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
                return_list.append("high")
            else:
                return_list.append("low")
        return return_list

    lin_reg = MySimpleLinearRegressionClassifier(discretizer)
    lin_reg.fit(X_train, y_train)

    # then check with a validated library
    sklearn_lin_reg = LinearRegression()
    sklearn_lin_reg.fit(X_train, y_train)

    assert np.allclose(lin_reg.regressor.slope,
                       sklearn_lin_reg.coef_)  # test the slope
    assert np.allclose(
        lin_reg.regressor.intercept, sklearn_lin_reg.intercept_
    )  # test the intercept

    # case 2

    X_train = [[val] for val in range(500)]  # 2D
    y_train = [row[0] * 2 + np.random.normal(0, 50) for row in X_train]  # 1D

    def discretizer2(x):
        return_list = []
        for val in x:
            if val >= 250:
                return_list.append("high")
            else:
                return_list.append("low")
        return return_list

    lin_reg = MySimpleLinearRegressionClassifier(discretizer2)
    lin_reg.fit(X_train, y_train)

    # then check with a valuidated library
    sklearn_lin_reg = LinearRegression()
    sklearn_lin_reg.fit(X_train, y_train)

    assert np.allclose(lin_reg.regressor.slope,
                       sklearn_lin_reg.coef_)  # test the slope
    assert np.allclose(
        lin_reg.regressor.intercept, sklearn_lin_reg.intercept_
    )  # test the intercept


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
                return_list.append("high")
            else:
                return_list.append("low")
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
        "high" if val >= 100 else "low" for val in sklearn_y_predicted
    ]

    assert np.array_equal(y_predicted, sklearn_y_predicted_classified)

    # case 2

    # simple/common test case
    X_train = [[val] for val in range(500)]  # 2D
    y_train = [row[0] * 2 + np.random.normal(0, 50) for row in X_train]  # 1D

    def discretizer2(x):
        return_list = []
        for val in x:
            if val >= 250:
                return_list.append("high")
            else:
                return_list.append("low")
        return return_list

    X_test = [[250], [-10], [200], [500], [0]]

    lin_reg = MySimpleLinearRegressionClassifier(discretizer2)
    lin_reg.fit(X_train, y_train)
    y_predicted = lin_reg.predict(X_test)

    # then check with a valuidated library
    sklearn_lin_reg = LinearRegression()
    sklearn_lin_reg.fit(X_train, y_train)
    sklearn_y_predicted = sklearn_lin_reg.predict(X_test)

    sklearn_y_predicted_classified = [
        "high" if val >= 250 else "low" for val in sklearn_y_predicted
    ]

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
        [1, 6],
    ]

    y_train_class_example2 = ["no", "yes",
                              "no", "no", "yes", "no", "yes", "yes"]

    knn_clf = MyKNeighborsClassifier(n_neighbors=3)  # n_neighbors=k=3
    knn_clf.fit(X_train_class_example2, y_train_class_example2)
    X_test = [2, 3]

    knn_kneighbors = knn_clf.kneighbors(X_test)

    # inclass solution
    inclass_sol = [[0, 1.4142135623730951], [4, 1.4142135623730951], [6, 2.0]]
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
        [19.6, 11.1],
    ]

    y_train_bramer_example = [
        "-",
        "-",
        "-",
        "+",
        "-",
        "+",
        "-",
        "+",
        "+",
        "+",
        "-",
        "-",
        "-",
        "-",
        "-",
        "+",
        "+",
        "+",
        "-",
        "+",
    ]

    knn_clf = MyKNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_train_bramer_example, y_train_bramer_example)

    X_test = [9.1, 11.0]

    knn_kneighbors = knn_clf.kneighbors(X_test)

    bramer_solution = (
        [
            0.6082762530298216,
            1.2369316876852974,
            2.202271554554525,
            2.8017851452243794,
            2.9154759474226513,
        ],
        [6, 5, 7, 4, 8],
    )
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
        [1, 6],
    ]

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
        [19.6, 11.1],
    ]

    y_train_bramer_example = [
        "-",
        "-",
        "-",
        "+",
        "-",
        "+",
        "-",
        "+",
        "+",
        "+",
        "-",
        "-",
        "-",
        "-",
        "-",
        "+",
        "+",
        "+",
        "-",
        "+",
    ]

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
    y_train = list(
        np.random.choice(["yes", "no", "maybe"], 100,
                         replace=True, p=[0.2, 0.6, 0.2])
    )

    dummy_cls = MyDummyClassifier()
    dummy_cls.fit(X_train, y_train)

    assert dummy_cls.most_common_label == "no"

    # case 3
    y_train = list(
        np.random.choice(
            ["Yes", "No", "Maybe", "Undecided", "Unanswered"],
            100,
            replace=True,
            p=[0.1, 0.2, 0.1, 0.2, 0.4],
        )
    )
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
    y_train = list(
        np.random.choice(["yes", "no", "maybe"], 100,
                         replace=True, p=[0.2, 0.6, 0.2])
    )

    dummy_cls = MyDummyClassifier()
    dummy_cls.fit(X_train, y_train)
    predicted = dummy_cls.predict(X_test)
    desk_calc = ["no", "no"]
    assert np.array_equal(predicted, desk_calc)

    # case 3
    y_train = list(
        np.random.choice(
            ["Yes", "No", "Maybe", "Undecided", "Unanswered"],
            100,
            replace=True,
            p=[0.1, 0.2, 0.1, 0.2, 0.4],
        )
    )
    dummy_cls = MyDummyClassifier()
    dummy_cls.fit(X_train, y_train)
    predicted = dummy_cls.predict(X_test)
    desk_calc = ["Unanswered", "Unanswered"]
    assert np.array_equal(predicted, desk_calc)


def test_naive_bayes_classifier_fit():
    # in-class Naive Bayes example (lab task #1)
    inclass_example_col_names = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5],  # yes
        [2, 6],  # yes
        [1, 5],  # no
        [1, 5],  # no
        [1, 6],  # yes
        [2, 6],  # no
        [1, 5],  # yes
        [1, 6],  # yes
    ]
    y_train_inclass_example = ["yes", "yes",
                               "no", "no", "yes", "no", "yes", "yes"]

    set1_posteriors_sol = {"yes": 1.0, "no": 0.5}
    set1_priors_sol = {"yes": 0.625, "no": 0.375}

    set1_naive_bayes_clf = MyNaiveBayesClassifier()
    set1_naive_bayes_clf.fit(X_train_inclass_example, y_train_inclass_example)

    set1_posteriors = set1_naive_bayes_clf.posteriors
    set1_priors = set1_naive_bayes_clf.priors

    assert np.array_equal(set1_posteriors, set1_posteriors_sol)
    assert np.array_equal(set1_priors, set1_priors_sol)

    # RQ5 (fake) iPhone purchases dataset
    iphone_col_names = ["standing", "job_status",
                        "credit_rating", "buys_iphone"]
    iphone_table = [
        [1, 3, "fair", "no"],
        [1, 3, "excellent", "no"],
        [2, 3, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [2, 1, "fair", "yes"],
        [2, 1, "excellent", "no"],
        [2, 1, "excellent", "yes"],
        [1, 2, "fair", "no"],
        [1, 1, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [1, 2, "excellent", "yes"],
        [2, 2, "excellent", "yes"],
        [2, 3, "fair", "yes"],
        [2, 2, "excellent", "no"],
        [2, 3, "fair", "yes"],
    ]
    iphone_X = [row[:3] for row in iphone_table]
    iphone_y = [row[3] for row in iphone_table]

    set2_posteriors_sol = {"yes": 0.28, "no": 0.16}
    set2_priors_sol = {"yes": 0.6666666666666666, "no": 0.3333333333333333}

    set2_naive_bayes_clf = MyNaiveBayesClassifier()
    set2_naive_bayes_clf.fit(iphone_X, iphone_y)

    set2_posteriors = set2_naive_bayes_clf.posteriors
    set2_priors = set2_naive_bayes_clf.priors

    assert np.array_equal(set2_posteriors, set2_posteriors_sol)
    assert np.array_equal(set2_priors, set2_priors_sol)

    # Bramer 3.2 train dataset
    train_col_names = ["day", "season", "wind", "rain", "class"]
    train_table = [
        ["weekday", "spring", "none", "none", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "high", "heavy", "late"],
        ["saturday", "summer", "normal", "none", "on time"],
        ["weekday", "autumn", "normal", "none", "very late"],
        ["holiday", "summer", "high", "slight", "on time"],
        ["sunday", "summer", "normal", "none", "on time"],
        ["weekday", "winter", "high", "heavy", "very late"],
        ["weekday", "summer", "none", "slight", "on time"],
        ["saturday", "spring", "high", "heavy", "cancelled"],
        ["weekday", "summer", "high", "slight", "on time"],
        ["saturday", "winter", "normal", "none", "late"],
        ["weekday", "summer", "high", "none", "on time"],
        ["weekday", "winter", "normal", "heavy", "very late"],
        ["saturday", "autumn", "high", "slight", "on time"],
        ["weekday", "autumn", "none", "heavy", "on time"],
        ["holiday", "spring", "normal", "slight", "on time"],
        ["weekday", "spring", "normal", "none", "on time"],
        ["weekday", "spring", "normal", "slight", "on time"],
    ]
    train_X = [row[:4] for row in train_table]
    train_y = [row[4] for row in train_table]

    set3_posteriors_sol = {
        "very late": 0.0,
        "late": 0.0,
        "cancelled": 0.0,
        "on time": 0.229592,
    }
    set3_priors_sol = {
        "very late": 0.15,
        "late": 0.1,
        "cancelled": 0.05,
        "on time": 0.7,
    }

    set3_naive_bayes_clf = MyNaiveBayesClassifier()
    set3_naive_bayes_clf.fit(train_X, train_y)

    set3_posteriors = set3_naive_bayes_clf.posteriors
    set3_priors = set3_naive_bayes_clf.priors

    print(set3_posteriors)
    print(set3_priors)

    assert np.array_equal(set3_posteriors, set3_posteriors_sol)
    assert np.array_equal(set3_priors, set3_priors_sol)


def test_naive_bayes_classifier_predict():
    # in-class Naive Bayes example (lab task #1)
    inclass_example_col_names = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5],  # yes
        [2, 6],  # yes
        [1, 5],  # no
        [1, 5],  # no
        [1, 6],  # yes
        [2, 6],  # no
        [1, 5],  # yes
        [1, 6],  # yes
    ]
    y_train_inclass_example = ["yes", "yes",
                               "no", "no", "yes", "no", "yes", "yes"]

    set1_X1 = [[2, 3]]
    # then we find the probability of the class "yes"
    set1_X1_sol = ["yes"]

    set1_naive_bayes_clf = MyNaiveBayesClassifier()
    set1_naive_bayes_clf.fit(X_train_inclass_example, y_train_inclass_example)

    set1_p_yes_X1 = set1_naive_bayes_clf.predict(set1_X1)

    assert set1_p_yes_X1 == set1_X1_sol

    # RQ5 (fake) iPhone purchases dataset
    iphone_col_names = ["standing", "job_status",
                        "credit_rating", "buys_iphone"]
    iphone_table = [
        [1, 3, "fair", "no"],
        [1, 3, "excellent", "no"],
        [2, 3, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [2, 1, "fair", "yes"],
        [2, 1, "excellent", "no"],
        [2, 1, "excellent", "yes"],
        [1, 2, "fair", "no"],
        [1, 1, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [1, 2, "excellent", "yes"],
        [2, 2, "excellent", "yes"],
        [2, 3, "fair", "yes"],
        [2, 2, "excellent", "no"],
        [2, 3, "fair", "yes"],
    ]

    iphone_X = [row[:3] for row in iphone_table]
    iphone_y = [row[3] for row in iphone_table]

    set2_X1 = [[2, 2, "fair"]]  # standing, job, credit rating
    set2_X2 = [[1, 1, "excellent"]]  # standing, job, credit rating

    set2_X1_pred_sol = ["yes"]
    set2_X2_pred_sol = ["yes"]

    set2_naive_bayes_clf = MyNaiveBayesClassifier()
    set2_naive_bayes_clf.fit(iphone_X, iphone_y)

    set2_X1_pred = set2_naive_bayes_clf.predict(set2_X1)
    set2_X2_pred = set2_naive_bayes_clf.predict(set2_X2)

    assert set2_X1_pred_sol == set2_X1_pred
    assert set2_X2_pred_sol == set2_X2_pred

    # Bramer 3.2 train dataset
    train_col_names = ["day", "season", "wind", "rain", "class"]
    train_table = [
        ["weekday", "spring", "none", "none", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "high", "heavy", "late"],
        ["saturday", "summer", "normal", "none", "on time"],
        ["weekday", "autumn", "normal", "none", "very late"],
        ["holiday", "summer", "high", "slight", "on time"],
        ["sunday", "summer", "normal", "none", "on time"],
        ["weekday", "winter", "high", "heavy", "very late"],
        ["weekday", "summer", "none", "slight", "on time"],
        ["saturday", "spring", "high", "heavy", "cancelled"],
        ["weekday", "summer", "high", "slight", "on time"],
        ["saturday", "winter", "normal", "none", "late"],
        ["weekday", "summer", "high", "none", "on time"],
        ["weekday", "winter", "normal", "heavy", "very late"],
        ["saturday", "autumn", "high", "slight", "on time"],
        ["weekday", "autumn", "none", "heavy", "on time"],
        ["holiday", "spring", "normal", "slight", "on time"],
        ["weekday", "spring", "normal", "none", "on time"],
        ["weekday", "spring", "normal", "slight", "on time"],
    ]
    train_X = [row[:4] for row in train_table]
    train_y = [row[4] for row in train_table]

    set3_X1 = [["weekday", "winter", "high", "heavy"]]
    set3_X1_pred_sol = ["on time"]

    set3_naive_bayes_clf = MyNaiveBayesClassifier()
    set3_naive_bayes_clf.fit(train_X, train_y)

    set3_X1_pred = set3_naive_bayes_clf.predict(set3_X1)

    assert set3_X1_pred_sol == set3_X1_pred


def test_decision_tree_classifier_fit():
    # use the 14 instance interview training set example
    # assert it against the tree constructer in the lab task 1

    # interview dataset
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"],
    ]
    y_train_interview = [
        "False",
        "False",
        "True",
        "True",
        "True",
        "False",
        "True",
        "False",
        "True",
        "True",
        "True",
        "True",
        "True",
        "False",
    ]

    tree_clf = MyDecisionTreeClassifier()
    tree_clf.fit(X_train_interview, y_train_interview)
    tree_interview = tree_clf.tree
    tree_interview_sol = [
        "Attribute",
        "att0",
        [
            "Value",
            "Senior",
            [
                "Attribute",
                "att1",
                ["Value", "Java", ["Leaf", "False", 2, 5]],
                [
                    "Value",
                    "Python",
                    [
                        "Attribute",
                        "att2",
                        ["Value", "no", ["Leaf", "False", 1, 2]],
                        ["Value", "yes", ["Leaf", "True", 1, 2]],
                    ],
                ],
                ["Value", "R", ["Leaf", "True", 1, 5]],
            ],
        ],
        ["Value", "Mid", ["Leaf", "True", 4, 14]],
        [
            "Value",
            "Junior",
            [
                "Attribute",
                "att3",
                ["Value", "no", ["Leaf", "True", 3, 5]],
                ["Value", "yes", ["Leaf", "False", 2, 5]],
            ],
        ],
    ]

    assert np.array_equal(tree_interview, tree_interview_sol)

    # use the bramer 4.1 figure 4.3 degrees dataset example
    # assert against the tree we made in 5.5 self-assessment exercise 1

    # bramer degrees dataset
    header_degrees = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
    X_train_degrees = [
        ["A", "B", "A", "B", "B"],
        ["A", "B", "B", "B", "A"],
        ["A", "A", "A", "B", "B"],
        ["B", "A", "A", "B", "B"],
        ["A", "A", "B", "B", "A"],
        ["B", "A", "A", "B", "B"],
        ["A", "B", "B", "B", "B"],
        ["A", "B", "B", "B", "B"],
        ["A", "A", "A", "A", "A"],
        ["B", "A", "A", "B", "B"],
        ["B", "A", "A", "B", "B"],
        ["A", "B", "B", "A", "B"],
        ["B", "B", "B", "B", "A"],
        ["A", "A", "B", "A", "B"],
        ["B", "B", "B", "B", "A"],
        ["A", "A", "B", "B", "B"],
        ["B", "B", "B", "B", "B"],
        ["A", "A", "B", "A", "A"],
        ["B", "B", "B", "A", "A"],
        ["B", "B", "A", "A", "B"],
        ["B", "B", "B", "B", "A"],
        ["B", "A", "B", "A", "B"],
        ["A", "B", "B", "B", "A"],
        ["A", "B", "A", "B", "B"],
        ["B", "A", "B", "B", "B"],
        ["A", "B", "B", "B", "B"],
    ]
    y_train_degrees = [
        "SECOND",
        "FIRST",
        "SECOND",
        "SECOND",
        "FIRST",
        "SECOND",
        "SECOND",
        "SECOND",
        "FIRST",
        "SECOND",
        "SECOND",
        "SECOND",
        "SECOND",
        "FIRST",
        "SECOND",
        "SECOND",
        "SECOND",
        "FIRST",
        "SECOND",
        "SECOND",
        "SECOND",
        "SECOND",
        "FIRST",
        "SECOND",
        "SECOND",
        "SECOND",
    ]

    tree_clf = MyDecisionTreeClassifier()
    tree_clf.fit(X_train_degrees, y_train_degrees)
    tree_degrees = tree_clf.tree
    # copilot made
    tree_degrees_sol = [
        "Attribute",
        "att0",
        [
            "Value",
            "A",
            [
                "Attribute",
                "att4",
                [
                    "Value",
                    "B",
                    [
                        "Attribute",
                        "att3",
                        ["Value", "B", ["Leaf", "SECOND", 7, 9]],
                        [
                            "Value",
                            "A",
                            [
                                "Attribute",
                                "att1",
                                ["Value", "B", ["Leaf", "SECOND", 1, 2]],
                                ["Value", "A", ["Leaf", "FIRST", 1, 2]],
                            ],
                        ],
                    ],
                ],
                ["Value", "A", ["Leaf", "FIRST", 5, 14]],
            ],
        ],
        ["Value", "B", ["Leaf", "SECOND", 12, 26]],
    ]

    assert np.array_equal(tree_degrees, tree_degrees_sol)

    # use the 15 instance iPhone training set from RQ5
    # assert against a tree we make with a desk check
    headers = ["standing", "job_status", "credit_rating", "buys_iphone"]
    data = [
        [1, 3, "fair", "no"],
        [1, 3, "excellent", "no"],
        [2, 3, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [2, 1, "fair", "yes"],
        [2, 1, "excellent", "no"],
        [2, 1, "excellent", "yes"],
        [1, 2, "fair", "no"],
        [1, 1, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [1, 2, "excellent", "yes"],
        [2, 2, "excellent", "yes"],
        [2, 3, "fair", "yes"],
        [2, 2, "excellent", "no"],
        [2, 3, "fair", "yes"],
    ]
    X_train_iphone = [row[:-1] for row in data]
    y_train_iphone = [row[-1] for row in data]

    tree_clf = MyDecisionTreeClassifier()
    tree_clf.fit(X_train_iphone, y_train_iphone)
    tree_iphone = tree_clf.tree

    tree_iphone_sol = [
        "Attribute",
        "att0",
        [
            "Value",
            1,
            [
                "Attribute",
                "att1",
                ["Value", 3, ["Leaf", "no", 2, 5]],
                [
                    "Value",
                    2,
                    [
                        "Attribute",
                        "att2",
                        ["Value", "fair", ["Leaf", "no", 1, 2]],
                        ["Value", "excellent", ["Leaf", "yes", 1, 2]],
                    ],
                ],
                ["Value", 1, ["Leaf", "yes", 1, 5]],
            ],
        ],
        [
            "Value",
            2,
            [
                "Attribute",
                "att1",
                ["Value", 3, ["Leaf", "yes", 3, 10]],
                [
                    "Value",
                    2,
                    [
                        "Attribute",
                        "att2",
                        ["Value", "fair", ["Leaf", "yes", 2, 4]],
                        ["Value", "excellent", ["Leaf", "yes", 2, 4]],
                    ],
                ],
                [
                    "Value",
                    1,
                    [
                        "Attribute",
                        "att2",
                        ["Value", "fair", ["Leaf", "yes", 1, 3]],
                        ["Value", "excellent", ["Leaf", "no", 2, 3]],
                    ],
                ],
            ],
        ],
    ]
    assert np.array_equal(tree_iphone, tree_iphone_sol)


def test_decision_tree_classifier_predict():
    # use the 14 instance interview training set example
    # assert against the two predictions made in lab task 2

    # interview dataset
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"],
    ]
    y_train_interview = [
        "False",
        "False",
        "True",
        "True",
        "True",
        "False",
        "True",
        "False",
        "True",
        "True",
        "True",
        "True",
        "True",
        "False",
    ]

    tree_clf = MyDecisionTreeClassifier()
    tree_clf.fit(X_train_interview, y_train_interview)
    X_test = [["Junior", "Java", "yes", "no"],
              ["Junior", "Java", "yes", "yes"]]
    y_pred = tree_clf.predict(X_test)
    assert np.array_equal(y_pred, ["True", "False"])

    # use the bramer dataset
    # test the set `[["B", "B", "B", "B", "B"], ["A", "A", "A", "A", "A"], ["A", "A", "A", "A", "B"]]`
    # check solution with a desk check

    # bramer degrees dataset
    header_degrees = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
    X_train_degrees = [
        ["A", "B", "A", "B", "B"],
        ["A", "B", "B", "B", "A"],
        ["A", "A", "A", "B", "B"],
        ["B", "A", "A", "B", "B"],
        ["A", "A", "B", "B", "A"],
        ["B", "A", "A", "B", "B"],
        ["A", "B", "B", "B", "B"],
        ["A", "B", "B", "B", "B"],
        ["A", "A", "A", "A", "A"],
        ["B", "A", "A", "B", "B"],
        ["B", "A", "A", "B", "B"],
        ["A", "B", "B", "A", "B"],
        ["B", "B", "B", "B", "A"],
        ["A", "A", "B", "A", "B"],
        ["B", "B", "B", "B", "A"],
        ["A", "A", "B", "B", "B"],
        ["B", "B", "B", "B", "B"],
        ["A", "A", "B", "A", "A"],
        ["B", "B", "B", "A", "A"],
        ["B", "B", "A", "A", "B"],
        ["B", "B", "B", "B", "A"],
        ["B", "A", "B", "A", "B"],
        ["A", "B", "B", "B", "A"],
        ["A", "B", "A", "B", "B"],
        ["B", "A", "B", "B", "B"],
        ["A", "B", "B", "B", "B"],
    ]
    y_train_degrees = [
        "SECOND",
        "FIRST",
        "SECOND",
        "SECOND",
        "FIRST",
        "SECOND",
        "SECOND",
        "SECOND",
        "FIRST",
        "SECOND",
        "SECOND",
        "SECOND",
        "SECOND",
        "FIRST",
        "SECOND",
        "SECOND",
        "SECOND",
        "FIRST",
        "SECOND",
        "SECOND",
        "SECOND",
        "SECOND",
        "FIRST",
        "SECOND",
        "SECOND",
        "SECOND",
    ]

    tree_clf = MyDecisionTreeClassifier()
    tree_clf.fit(X_train_degrees, y_train_degrees)
    X_test = [
        ["B", "B", "B", "B", "B"],
        ["A", "A", "A", "A", "A"],
        ["A", "A", "A", "A", "B"],
    ]
    y_pred = tree_clf.predict(X_test)

    assert np.array_equal(y_pred, ["SECOND", "FIRST", "FIRST"])

    # for the 15 instance iPhone training set from RQ5
    # assert against the two unseen instances from RQ5
    # standing = 2, job_status = 2, credit_rating=fair
    # standing = 1, job_status = 1, credit_rating=excellent
    # check the solution with our desk check
    # no ???
    # no ???

    headers = ["standing", "job_status", "credit_rating", "buys_iphone"]
    data = [
        [1, 3, "fair", "no"],
        [1, 3, "excellent", "no"],
        [2, 3, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [2, 1, "fair", "yes"],
        [2, 1, "excellent", "no"],
        [2, 1, "excellent", "yes"],
        [1, 2, "fair", "no"],
        [1, 1, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [1, 2, "excellent", "yes"],
        [2, 2, "excellent", "yes"],
        [2, 3, "fair", "yes"],
        [2, 2, "excellent", "no"],
        [2, 3, "fair", "yes"],
    ]
    X_train_iphone = [row[:-1] for row in data]
    y_train_iphone = [row[-1] for row in data]

    tree_clf = MyDecisionTreeClassifier()
    tree_clf.fit(X_train_iphone, y_train_iphone)
    X_test = [[2, 2, "fair"], [1, 1, "excellent"]]
    y_pred = tree_clf.predict(X_test)
    assert np.array_equal(y_pred, ["yes", "yes"])
