import random


header = ["height(cm)", "weight(kg)", "size(t-shirt)"]
table = [
    [158, 58],  # "M"
    [158, 59],  # "M"
    [158, 63],  # "M"
    [160, 59],  # "M"
    [160, 60],  # "M"
    [163, 60],  # "M"
    [163, 61],  # "M"
    [160, 64],  # "L"
    [163, 64],  # "L"
    [165, 61],  # "L"
    [165, 62],  # "L"
    [165, 65],  # "L"
    [168, 62],  # "L"
    [168, 63],  # "L"
    [168, 66],  # "L"
    [170, 63],  # "L"
    [170, 64],  # "L"
    [170, 68],  # "L"
]
# TODO: normalize data before calculating distances


def perform_k_means_cluster(table, k):
    # how to represent clusers?
    # 1. a list of cluserts
    # 2. add a column to the table for the cluster nymber
    #   recommended because it is what sci kit lean uses
    table = [row + [None] for row in table]

    # step 2
    # select k random instances
    random_instances = random.sample(table, k)  # without replacement
    for i in range(len(random_instances)):
        random_instances[i][-1] = i
    for row in table:
        print(row)

    # step 3
    # assign each instance to the closest cluster
    # cluster_centers = compute_cluster_centers(table, k)
    # consider using groupby
    # for instance in table:
    #   nearest_cluster_num = find_nearst_cluster(instance, cluster_centers)
    #   update the instances cluster num
    #   instance[-1] = nearest_cluster_num

    # step 4
    # recalculate the centoids
    # new_cluster_centers = compute_cluster_centers(table, k)

    # step 5
    # moved = check_clusters_moved(cluster_centers, new_cluster_centers)
    # while moved: # this can be moved to be before step 3 and 4 :)
    #   repeat steps 3 and 4 until the centoids do not change

    # goal is to return two lists:
    # lables_ is a list of cluster numbers
    # cluster_centers) is a list of centroids
    return [], []  # TODO: fix this


# step 1
random.seed(0)
k = 2  # need to tune this parameter
lables_, cluster_centers_ = perform_k_means_cluster(table, k)
print("lables_:", lables_)
print("cluster_centers_:", cluster_centers_)
