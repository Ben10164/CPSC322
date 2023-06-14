import operator
import numpy as np


def compute_euclidean_distance(v1, v2):
    """Computes the Euclidean distance between two vectors.
    Args:
        v1(list of numeric vals): The first vector
        v2(list of numeric vals): The second vector
    Returns:
        distance(float): The Euclidean distance between v1 and v2
    """
    total = 0
    for i in range(len(v1)):
        total += (v1[i] - v2[i]) ** 2
    return total ** (1/2)


def get_row_indexes_distances_sorted(X_train, X_test):
    """Computes the distances between each test instance in X_test and each
        training instance in X_train, and returns the sorted list of
        (index, distance) tuples.
    Args:
        X_train(list of list of numeric vals): The list of training instances (samples).
            The shape of X_train is (n_train_samples, n_features)
        X_test(list of list of numeric vals): The list of testing samples
            The shape of X_test is (n_test_samples, n_features)
    Returns:
        sorted_distances(list of (int, float)): The list of (index, distance) tuples
            sorted in ascending order by distance
    """
    row_indexes_distances = []
    for i, train_instance in enumerate(X_train):
        dist = compute_euclidean_distance(train_instance, X_test)
        row_indexes_distances.append([i, dist])
    # now we can sort the items by the distance (item[1])
    row_indexes_distances.sort(key=operator.itemgetter(-1))
    return row_indexes_distances


def get_mpg_rating(mpg):
    """
    Return the rating of the given mpg
    """
    if mpg >= 45:
        rating = 10
    elif 37 <= mpg < 45:
        rating = 9
    elif 31 <= mpg < 37:
        rating = 8
    elif 27 <= mpg < 31:
        rating = 7
    elif 24 <= mpg < 27:
        rating = 6
    elif 20 <= mpg < 24:
        rating = 5
    elif 17 <= mpg < 20:
        rating = 4
    elif 15 <= mpg < 17:
        rating = 3
    elif 14 <= mpg < 15:
        rating = 2
    else:
        rating = 1
    return rating


def randomize_in_place(alist, parallel_list=None, seed=None):
    """
    Randomize the order of the elements in alist.
    If parallel_list is not None, then it is also randomized in the same way while being kept parallel.
    """
    if seed is not None:
        np.random.seed(seed)

    for i in range(len(alist)):
        rand_index = np.random.randint(0, len(alist))  # [0, len(alist))
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]
    if parallel_list is not None:
        return alist, parallel_list
    else:
        return alist


def error_rate(predicted_labels, actual_labels):
    """
    Computes the error rate given predicted labels and actual labels.
    multiclass
    """
    num_errors = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] != actual_labels[i]:
            num_errors += 1
    return num_errors / len(predicted_labels)


def folds_to_train_test(X_train_folds, X_test_folds, y_train_folds, y_test_folds):

    # now with X_train_folds, we create X_train
    X_train = []
    for fold in X_train_folds:
        for instance in fold:
            X_train.append(instance)
    X_test = []
    for fold in X_test_folds:
        for instance in fold:
            X_test.append(instance)
    y_test = []
    for fold in y_test_folds:
        for instance in fold:
            y_test.append(instance)
    y_train = []
    for fold in y_train_folds:
        for instance in fold:
            y_train.append(instance)
    return X_train, X_test, y_train, y_test


def indexes_to_fold(X_train_indexes, X_test_indexes, X, y):
    # now find y_train and y_test
    y_train_folds = []
    y_test_folds = []
    for test in X_test_indexes:
        index_y_test = []
        for index in test:
            index_y_test.append(y[index])
        y_test_folds.append(index_y_test)

    for train in X_train_indexes:
        index_y_train = []
        for index in train:
            index_y_train.append(y[index])
        y_train_folds.append(index_y_train)
    X_test_folds = []
    for fold in X_test_indexes:
        index_X_test = []
        for index in fold:
            index_X_test.append(X[index])
        X_test_folds.append(index_X_test)

    X_train_folds = []
    for fold in X_train_indexes:
        index_X_train = []
        for index in fold:
            index_X_train.append(X[index])
        X_train_folds.append(index_X_train)
    return X_train_folds, X_test_folds, y_train_folds, y_test_folds


def stratify(y, folds, random_state=None):
    """
    Stratifies the given labels into the given number of folds.
    """
    # first we need to find the number of unique labels
    unique_labels = set(y)
    # now we create a list of lists, where each list is the indexes of the instances with the same label
    index_lists = []
    for label in unique_labels:
        index_lists.append([i for i, x in enumerate(y) if x == label])
    # now we need to randomize the indexes for each label
    for index_list in index_lists:
        randomize_in_place(index_list, seed=random_state)
    # now we need to split the indexes into the folds
    index_lists_folds = []
    for i in range(folds):
        index_lists_folds.append([])
    for index_list in index_lists:
        for i in range(folds):
            index_lists_folds[i].append(
                index_list[i*(len(index_list)//folds):(i+1)*(len(index_list)//folds)])
    return index_lists_folds


def stratify_in_place(X, y):
    """
    Stratifies the given labels into the given number of folds.

    REturns: X,y
    """
    # first we need to find the number of unique labels
    unique_labels = set(y)
    # now we create a list of lists, where each list is the indexes of the instances with the same label
    index_lists = []
    for label in unique_labels:
        index_lists.append([i for i, x in enumerate(y) if x == label])
    # now we need to randomize the indexes for each label
    for index_list in index_lists:
        randomize_in_place(index_list)
    # now we need to split the indexes into the folds
    index_lists_folds = []
    for i in range(len(index_lists[0])):
        index_lists_folds.append([])
    for index_list in index_lists:
        for i in range(len(index_lists[0])):
            index_lists_folds[i].append(
                index_list[i*(len(index_list)//len(index_lists)):(i+1)*(len(index_list)//len(index_lists))])
    X_folds = []
    y_folds = []
    for fold in index_lists_folds:
        X_fold = []
        y_fold = []
        for index_list in fold:
            for index in index_list:
                X_fold.append(X[index])
                y_fold.append(y[index])
        X_folds.append(X_fold)
        y_folds.append(y_fold)
    return X_folds, y_folds


# def stratified_kfold_split(X, y, n_splits):
#     """
#     Stratified k-fold split.
#     """
#     # first we need to find the number of unique labels
#     unique_labels = set(y)
#     # now we create a list of lists, where each list is the indexes of the instances with the same label
#     index_lists = []
#     for label in unique_labels:
#         index_lists.append([i for i, x in enumerate(y) if x == label])
#     # now we need to randomize the indexes for each label
#     for index_list in index_lists:
#         randomize_in_place(index_list)
#     # now we need to split the indexes into the folds
#     index_lists_folds = []
#     for i in range(n_splits):
#         index_lists_folds.append([])
#     for index_list in index_lists:
#         for i in range(n_splits):
#             index_lists_folds[i].append(
#                 index_list[i*(len(index_list)//n_splits):(i+1)*(len(index_list)//n_splits)])
#     return index_lists_folds


def group_by(data):
    # groupby_col_index = header.index(groupby_col_name)  # use this later
    group_names = sorted(list(set(data)))  # e.g. [75, 76, 77]
    group_subtables = [[] for _ in group_names]  # e.g. [[], [], []]

    for i in range(len(data)):
        # which subtable does this row belong?
        groupby_val_subtable_index = group_names.index(data[i])
        group_subtables[groupby_val_subtable_index].append(i)

    return group_names, group_subtables


def get_column(table, col_index):
    col = []
    for row in table:
        value = row[col_index]
        if value != "NA":
            col.append(value)
    return col
