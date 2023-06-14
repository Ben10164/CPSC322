"""test_myevaluation.py

@author gsprint23
Note: do not modify this file
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score

from mysklearn import myevaluation

# note: order is actual/received student value, expected/solution
def test_train_test_split():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    X_1 = [[0, 1],
       [2, 3],
       [4, 5],
       [6, 7],
       [8, 9]]
    y_1 = [0, 1, 2, 3, 4]
    # then put repeat values in
    X_2 = [[0, 1],
       [2, 3],
       [5, 6],
       [6, 7],
       [0, 1]]
    y_2 = [2, 3, 3, 2, 2]
    test_sizes = [0.33, 0.25, 4, 3, 2, 1]
    for X, y in zip([X_1, X_2], [y_1, y_2]):
        for test_size in test_sizes:
            X_train_solution, X_test_solution, y_train_solution, y_test_solution =\
                train_test_split(X, y, test_size=test_size, random_state=0, shuffle=False)
            X_train, X_test, y_train, y_test = myevaluation.train_test_split(X, y, test_size=test_size, random_state=0, shuffle=False)

            assert np.array_equal(X_train, X_train_solution) # order matters with np.array_equal()
            assert np.array_equal(X_test, X_test_solution)
            assert np.array_equal(y_train, y_train_solution)
            assert np.array_equal(y_test, y_test_solution)

    # if get here, should have base algorithm implemented just fine
    # now test random_state and shuffle
    test_size = 2
    X_train0_notshuffled, X_test0_notshuffled, y_train0_notshuffled, y_test0_notshuffled =\
        myevaluation.train_test_split(X_1, y_1, test_size=test_size, random_state=0, shuffle=False)
    X_train0_shuffled, X_test0_shuffled, y_train0_shuffled, y_test0_shuffled =\
        myevaluation.train_test_split(X_1, y_1, test_size=test_size, random_state=0, shuffle=True)
    # make sure shuffle keeps X and y parallel
    for i, _ in enumerate(X_train0_shuffled):
        assert y_1[X_1.index(X_train0_shuffled[i])] == y_train0_shuffled[i]
    # same random_state but with shuffle= False vs True should produce diff folds
    assert not np.array_equal(X_train0_notshuffled, X_train0_shuffled)
    assert not np.array_equal(y_train0_notshuffled, y_train0_shuffled)
    assert not np.array_equal(X_test0_notshuffled, X_test0_shuffled)
    assert not np.array_equal(y_test0_notshuffled, y_test0_shuffled)
    X_train1_shuffled, X_test1_shuffled, y_train1_shuffled, y_test1_shuffled =\
        myevaluation.train_test_split(X_1, y_1, test_size=test_size, random_state=1, shuffle=True)
    # diff random_state should produce diff folds when shuffle=True
    assert not np.array_equal(X_train0_shuffled, X_train1_shuffled)
    assert not np.array_equal(y_train0_shuffled, y_train1_shuffled)
    assert not np.array_equal(X_test0_shuffled, X_test1_shuffled)
    assert not np.array_equal(y_test0_shuffled, y_test1_shuffled)

# test utility function
def check_folds(n, n_splits, train_folds, test_folds, train_folds_solution, test_folds_solution):
    """Utility function

    n(int): number of samples in dataset
    """
    all_test_indices = []
    all_train_indices = []
    all_train_indices_solution = []
    all_test_indices_solution = []
    for i in range(n_splits):
        # make sure all indices are accounted for in each split
        all_indices_in_fold = train_folds[i] + test_folds[i]
        assert len(all_indices_in_fold) == n
        for index in range(n):
            assert index in all_indices_in_fold
        all_test_indices.extend(test_folds[i])
        all_train_indices.extend(train_folds[i])
        all_train_indices_solution.extend(train_folds_solution[i])
        all_test_indices_solution.extend(test_folds_solution[i])

    # make sure all indices are in a test set
    assert len(all_test_indices) == n
    for index in range(n):
        assert index in all_indices_in_fold
    # make sure fold test on appropriate number of indices
    all_test_indices.sort()
    all_test_indices_solution.sort()
    assert all_test_indices == all_test_indices_solution

    # make sure fold train on appropriate number of indices
    all_train_indices.sort()
    all_train_indices_solution.sort()
    assert all_train_indices == all_train_indices_solution

def test_kfold_cross_validation():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

    Notes:
        The order does not need to match sklearn's split() so long as the implementation is correct
    """
    X = [[0, 1], [2, 3], [4, 5], [6, 7]]
    y = [1, 2, 3, 4]

    n_splits = 2
    for tset in [X, y]:
        train_folds, test_folds = myevaluation.kfold_cross_validation(tset, n_splits=n_splits)
        standard_kf = KFold(n_splits=n_splits)
        train_folds_solution = []
        test_folds_solution = []
        # convert all solution numpy arrays to lists
        for train_fold_solution, test_fold_solution in list(standard_kf.split(tset)):
            train_folds_solution.append(list(train_fold_solution))
            test_folds_solution.append(list(test_fold_solution))
        check_folds(len(tset), n_splits, train_folds, test_folds, train_folds_solution, test_folds_solution)

    # more complicated dataset
    table = [
        [3, 2, "no"],
        [6, 6, "yes"],
        [4, 1, "no"],
        [4, 4, "no"],
        [1, 2, "yes"],
        [2, 0, "no"],
        [0, 3, "yes"],
        [1, 6, "yes"]
    ]
    # n_splits = 2, ..., 8 (LOOCV)
    for n_splits in range(2, len(table) + 1):
        train_folds, test_folds = myevaluation.kfold_cross_validation(table, n_splits=n_splits)
        standard_kf = KFold(n_splits=n_splits)
        train_folds_solution = []
        test_folds_solution = []
        # convert all solution numpy arrays to lists
        for train_fold_solution, test_fold_solution in list(standard_kf.split(table)):
            train_folds_solution.append(list(train_fold_solution))
            test_folds_solution.append(list(test_fold_solution))
        check_folds(len(table), n_splits, train_folds, test_folds, train_folds_solution, test_folds_solution)

    # if get here, should have base algorithm implemented just fine
    # now test random_state and shuffle
    train_folds0_notshuffled, test_folds0_notshuffled = myevaluation.kfold_cross_validation(X, n_splits=2, random_state=0, shuffle=False)
    train_folds0_shuffled, test_folds0_shuffled = myevaluation.kfold_cross_validation(X, n_splits=2, random_state=0, shuffle=True)
    # same random_state but with shuffle= False vs True should produce diff folds
    for i, _ in enumerate(train_folds0_notshuffled):
        assert not np.array_equal(train_folds0_notshuffled[i], train_folds0_shuffled[i])
        assert not np.array_equal(test_folds0_notshuffled[i], test_folds0_shuffled[i])
    train_folds1_shuffled, test_folds1_shuffled = myevaluation.kfold_cross_validation(X, n_splits=2, random_state=1, shuffle=True)
    # diff random_state should produce diff folds when shuffle=True
    for i, _ in enumerate(train_folds0_shuffled):
        assert not np.array_equal(train_folds0_shuffled[i], train_folds1_shuffled[i])
        assert not np.array_equal(test_folds0_shuffled[i], test_folds1_shuffled[i])

# test utility function
def get_min_label_counts(y, label, n_splits):
    """Utility function
    """
    label_counts = sum([1 for yval in y if yval == label])
    min_test_label_count = label_counts // n_splits
    min_train_label_count = (n_splits - 1) * min_test_label_count
    return min_train_label_count, min_test_label_count

def test_stratified_kfold_cross_validation():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold

    Notes:
        This test does not test shuffle or random_state
        The order does not need to match sklearn's split() so long as the implementation is correct
    """
    # note: this test case does test order against sklearn's
    X = [[0, 1], [2, 3], [4, 5], [6, 4]]
    y = [0, 0, 1, 1]

    n_splits = 2
    train_folds, test_folds = myevaluation.stratified_kfold_cross_validation(X, y, n_splits=n_splits)
    stratified_kf = StratifiedKFold(n_splits=n_splits)
    train_folds_solution = []
    test_folds_solution = []
    # convert all solution numpy arrays to lists
    for train_fold_solution, test_fold_solution in list(stratified_kf.split(X, y)):
        train_folds_solution.append(list(train_fold_solution))
        test_folds_solution.append(list(test_fold_solution))
    # sklearn solution and order:
    # i=0: TRAIN: [1 3] TEST: [0 2]
    # i=1: TRAIN: [0 2] TEST: [1 3]
    check_folds(len(y), n_splits, train_folds, test_folds, train_folds_solution, test_folds_solution)
    for i in range(n_splits):
        # since the actual result could have folds in diff order, make sure this train and test set is in the solution somewhere
        # sort the train and test sets of the fold so the indices can be in any order within a set
        # make sure at least minimum count of each label in each split
        for label in [0, 1]:
            train_yes_labels = [y[j] for j in train_folds[i] if y[j] == label]
            test_yes_labels = [y[j] for j in test_folds[i] if y[j] == label]
            min_train_label_count, min_test_label_count = get_min_label_counts(y, label, n_splits)
            assert len(train_yes_labels) >= min_train_label_count
            assert len(test_yes_labels) >= min_test_label_count

    # note: this test case does not test order against sklearn's solution
    table = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]
    ]
    table_y = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    for n_splits in range(2, 5):
        train_folds, test_folds = myevaluation.stratified_kfold_cross_validation(table, table_y, n_splits=n_splits)
        stratified_kf = StratifiedKFold(n_splits=n_splits)
        train_folds_solution = []
        test_folds_solution = []
        # convert all solution numpy arrays to lists
        for train_fold_solution, test_fold_solution in list(stratified_kf.split(table, table_y)):
            train_folds_solution.append(list(train_fold_solution))
            test_folds_solution.append(list(test_fold_solution))
        check_folds(len(table), n_splits, train_folds, test_folds, train_folds_solution, test_folds_solution)

        for i in range(n_splits):
            # make sure at least minimum count of each label in each split
            for label in ["yes", "no"]:
                train_yes_labels = [table_y[j] for j in train_folds[i] if table_y[j] == label]
                test_yes_labels = [table_y[j] for j in test_folds[i] if table_y[j] == label]
                min_train_label_count, min_test_label_count = get_min_label_counts(table_y, label, n_splits)
                assert len(train_yes_labels) >= min_train_label_count
                assert len(test_yes_labels) >= min_test_label_count

    # if get here, should have base algorithm implemented just fine
    # now test random_state and shuffle
    train_folds0_notshuffled, test_folds0_notshuffled = \
        myevaluation.stratified_kfold_cross_validation(X, y, n_splits=2, random_state=0, shuffle=False)
    train_folds0_shuffled, test_folds0_shuffled = myevaluation.stratified_kfold_cross_validation(X, y, n_splits=2, random_state=0, shuffle=True)
    # same random_state but with shuffle= False vs True should produce diff folds
    for i, _ in enumerate(train_folds0_notshuffled):
        assert not np.array_equal(train_folds0_notshuffled[i], train_folds0_shuffled[i])
        assert not np.array_equal(test_folds0_notshuffled[i], test_folds0_shuffled[i])
    train_folds1_shuffled, test_folds1_shuffled = myevaluation.stratified_kfold_cross_validation(X, y, n_splits=2, random_state=1, shuffle=True)
    # diff random_state should produce diff folds when shuffle=True
    for i, _ in enumerate(train_folds0_shuffled):
        assert not np.array_equal(train_folds0_shuffled[i], train_folds1_shuffled[i])
        assert not np.array_equal(test_folds0_shuffled[i], test_folds1_shuffled[i])

# test utility function
def check_same_lists_regardless_of_order(list1, list2):
    """Utility function
    """
    assert len(list1) == len(list2) # same length
    for item in list1:
        assert item in list2
        list2.remove(item)
    assert len(list2) == 0
    return True

def test_bootstrap_sample():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html

    Notes:
        This test does not test shuffle or random_state
    """
    X = [[1., 0.], [2., 1.], [0., 0.]]
    y = [0, 1, 2]
    # X_sample, y_sample = resample(X, y, random_state=0) #n_samples = None means length of first dimension
    X_sample_solution = [[1., 0.], [2., 1.], [1., 0.]]
    X_out_of_bag_solution = [[0., 0.]]
    y_sample_solution = [0, 1, 0]
    y_out_of_bag_solution = [2]

    X_sample, X_out_of_bag, y_sample, y_out_of_bag = myevaluation.bootstrap_sample(X, y, random_state=0)
    check_same_lists_regardless_of_order(X_sample, X_sample_solution)
    check_same_lists_regardless_of_order(y_sample, y_sample_solution)
    check_same_lists_regardless_of_order(X_out_of_bag, X_out_of_bag_solution)
    check_same_lists_regardless_of_order(y_out_of_bag, y_out_of_bag_solution)

    # another example adapted from
    # https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/
    X = [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]]
    X_sample_solution = [[0.6], [0.4], [0.5], [0.1]]
    X_out_of_bag_solution = [[0.2], [0.3]]
    X_sample, X_out_of_bag, y_sample, y_out_of_bag = myevaluation.bootstrap_sample(X, y=None, n_samples=4, random_state=1)
    check_same_lists_regardless_of_order(X_sample, X_sample_solution)
    assert y_sample is None
    check_same_lists_regardless_of_order(X_out_of_bag, X_out_of_bag_solution)
    assert y_out_of_bag is None

def test_confusion_matrix():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]

    matrix_solution = [[2, 0, 0],
                [0, 0, 1],
                [1, 0, 2]]
    matrix = myevaluation.confusion_matrix(y_true, y_pred, [0, 1, 2])
    assert np.array_equal(matrix, matrix_solution)

    y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    matrix = myevaluation.confusion_matrix(y_true, y_pred, ["ant", "bird", "cat"])
    assert np.array_equal(matrix, matrix_solution)

    y_true = [0, 1, 0, 1]
    y_pred = [1, 1, 1, 0]

    matrix_solution = [[0, 2],[1, 1]]
    matrix = myevaluation.confusion_matrix(y_true, y_pred, [0, 1])
    assert np.array_equal(matrix, matrix_solution)

def test_accuracy_score():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    y_pred = [0, 2, 1, 3]
    y_true = [0, 1, 2, 3]

    # normalize=True
    score = myevaluation.accuracy_score(y_true, y_pred, normalize=True)
    score_sol =  accuracy_score(y_true, y_pred, normalize=True) # 0.5
    assert np.isclose(score, score_sol)

    # normalize=False
    score = myevaluation.accuracy_score(y_true, y_pred, normalize=False)
    score_sol =  accuracy_score(y_true, y_pred, normalize=False) # 2
    assert np.isclose(score, score_sol)