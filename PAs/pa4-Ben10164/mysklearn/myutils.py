from logging.handlers import RotatingFileHandler
import operator


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
