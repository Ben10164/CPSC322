import utils
import operator
import numpy as np
from scipy.spatial import distance

def main():
    header = ["att1", "att2"]
    X_train = [

        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]
    ]
    y_train = ["no", "yes", "no", "no", "yes",
               "no", "yes", "yes"]  # parallel to X_train

    X_test = [2, 3]

    row_indexes_distances = []
    for i, train_instance in enumerate(X_train):
        dist = compute_euclidean_distance(train_instance, X_test)
        row_indexes_distances.append([i, dist])

    print(row_indexes_distances)

    # now we can sort the items by the distance (item[1])
    row_indexes_distances.sort(key=operator.itemgetter(-1))
    k = 3
    top_k = row_indexes_distances[:k]
    print("Top k:")
    for row in top_k:
        print(row)

    # TODO: extract the y lables from y_train for the k bnearest neightbors
    # then figure out the majority class labvel, that is this
    # test instance's rediction
    freqs = utils.get_frequencies(y_train )
    print(freqs)



def test_compute_euclidean_distance():
    v1 = np.random.random(100)
    v2 = np.random.random(100)

    personal = compute_euclidean_distance(v1, v2)

    print(distance.euclidean(v1, v2))
    print(personal)
    # then check with a valuidated library
    scipy_euclidean = (distance.euclidean(v1, v2))
    assert np.isclose(personal, scipy_euclidean)


def compute_euclidean_distance(v1, v2):
    total = 0
    for i in range(len(v1)):
        total += (v1[i] - v2[i]) ** 2
    return (total ** (1/2))


if __name__ == '__main__':
    main()

    