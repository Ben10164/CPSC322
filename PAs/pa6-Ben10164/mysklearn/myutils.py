import math
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


# niave bayes
def p_value(X_train, new_instance, y_train, value):
    """
    Calculates the probability of a value given a training set

    Parameters
    ----------
    X_train : list
        A list of lists of strings. Each inner list represents a training
        instance.
    new_instance : list
        A list of strings. Each string represents an attribute.
    y_train : list
        A list of strings. Each string represents a class label.
    value : string
        A string representing a class label.

    Returns
    -------
    float
        The probability of a value given a training set
    """
    p_X = p_list(X_train, new_instance, y_train, value)
    p_X_Mult = multiply_list(p_X)
    p_yes = p_X_Mult * y_train.count(value)
    return p_yes


def multiply_list(data):
    """
    Multiplies the values in a list

    Parameters
    ----------
    data : list
        A list of floats.

    Returns
    -------
    float
        The product of the values in the list
    """
    product = 1
    for i in range(len(data)):
        product *= data[i]
    return product


def p_list(X_train, new_instance, y_train, value):
    """
    Calculates the probability of each attribute in a training set

    Parameters
    ----------
    X_train : list
        A list of lists of strings. Each inner list represents a training
        instance.
    new_instance : list
        A list of strings. Each string represents an attribute. (X_train[i])
    y_train : list
        A list of strings. Each string represents a class label.
    value : string
        A string representing a class label.

    Returns
    -------
    list
        A list of floats. Each float represents the probability of an
        attribute in a training set.
    """
    perdiction_list = []
    for i in range(len(new_instance)):
        attribute_matches_count = 0
        for j in range(len(X_train)):
            if (new_instance[i] == X_train[j][i]):
                if (y_train[j] == value):
                    attribute_matches_count += 1
        perdiction_list.append(
            attribute_matches_count/y_train.count(value))
    return perdiction_list


def gaussian(x, mean, sdev):
    """
    Calculates the probability of a value given a training set

    Parameters
    ----------
    x : float
        A float representing an attribute.
    mean : float
        A float representing the mean of a distribution.
    sdev : float
        A float representing the standard deviation of a distribution.

    Returns
    -------
    float
        The probability of a value given a training set
    """
    first, second = 0, 0
    if sdev > 0:
        first = 1 / (math.sqrt(2 * math.pi) * sdev)
        second = math.e ** (-((x - mean) ** 2) / (2 * (sdev ** 2)))
    return first * second


def get_priors(y_train):
    """
    Calculates the prior probability of each class label

    Parameters
    ----------
    y_train : list
        A list of strings. Each string represents a class label.

    Returns
    -------
    Dict
        A dictionary where the keys are class labels and the values are
        prior probabilities.
    """
    unique_labels = set(y_train)
    priors = {}
    for label in unique_labels:
        priors[label] = y_train.count(label)/len(y_train)
    return priors


def get_posteriors(X_train, y_train, priors):
    """
    Calculates the posterior probability of each class label

    Parameters
    ----------
    X_train : list
        A list of lists of strings. Each inner list represents a training
        instance.
    y_train : list
        A list of strings. Each string represents a class label.
    priors : list
        A list of floats. Each float represents the prior probability of a
        class label.

    Returns
    -------
    Dict
        A dictionary where the keys are class labels and the values are
        posterior probabilities.
    """
    unique_labels = set(y_train)
    posteriors = {}
    for label in unique_labels:
        posteriors[label] = p_value(
            X_train, X_train[0], y_train, label) * priors[label]
        # this is a really gross number so we will round it to be 5 decimal places
        posteriors[label] = round(posteriors[label], 6)
    return posteriors


def get_prediction_naive_bayes(row, posteriors, priors):
    """
    Calculates the prediction of a row using naive bayes

    Parameters
    ----------
    posteriors : Dict
        A dictionary where the keys are class labels and the values are
        posterior probabilities.

    Returns
    -------
    string
        A string representing the predicted class label.
    """
    max_posterior = 0
    prediction = ""
    for label in posteriors:
        if posteriors[label] > max_posterior:
            max_posterior = posteriors[label]
            prediction = label
    return prediction
