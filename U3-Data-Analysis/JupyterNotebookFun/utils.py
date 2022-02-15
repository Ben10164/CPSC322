# THIS IS USED IN Matplotlib.ipynb
import numpy as np


def get_column(table, header, col_name):
    # col_index = header.index(col_name)
    # NOTE: Uses the new function for safety
    col_index = get_index(header, col_name)
    col = []
    for row in table:
        if row[col_index] != "NA":
            col.append(row[col_index])
    return col


# NOTE: This is an important function when not using pandas
def get_index(header, col_name):
    try:
        col_index = header.index(col_name)
        return col_index
    except ValueError:
        print("That col name is not in the header")


def get_frequencies(table, header, col_name):
    col = get_column(table, header, col_name)
    col.sort()  # inplace sort
    values = []
    counts = []
    for value in col:
        if value in values:
            # we have seen it before
            counts[-1] += 1
        else:
            values.append(value)
            counts.append(1)
    return values, counts


# header = ["CarName", "ModelYear", "MSRP"]
# msrp_table = [
#     ["ford pinto", 75, 2769],
#     ["toyota corolla", 75, 2711],
#     ["ford pinto", 76, 3025],
#     ["toyota corolla", 77, 2789],
# ]

def dumby_function():
    print("Hi!")


def group_by(table, header, groupby_col_name):
    groupby_col_index = header.index(groupby_col_name)  # use this later
    groupby_col = get_column(table, header, groupby_col_name)
    group_names = sorted(list(set(groupby_col)))  # e.g. [75, 76, 77]
    group_subtables = [[] for _ in group_names]  # e.g. [[], [], []]

    for row in table:
        groupby_val = row[groupby_col_index]  # e.g. this row's modelyear
        # which subtable does this row belong?
        groupby_val_subtable_index = group_names.index(groupby_val)
        group_subtables[groupby_val_subtable_index].append(
            row.copy())  # make a copy

    return group_names, group_subtables


# discretization lab
# 1
def compute_equal_width_cutoffs(values, num_bins):
    # we need to figure out the width of a bin first
    values_range = max(values) - min(values)
    bin_width = values_range / num_bins
    # now we have the width of the bins :) (also its a float)

    # range() works well with integer start stops and steps
    # np.arange() is for floating point start stop and steps
    cutoffs = list(np.arange(min(values), max(values), bin_width))
    cutoffs.append(max(values))  # exact max (because we do N + 1 cutoffs)
    # if your aopplication allows, convert cutoffs to ints
    # otherwise optionally round them to 2 decimal places
    cutoffs = [round(cutoff, 2) for cutoff in cutoffs]
    return cutoffs


def compute_bin_frequencies(values, cutoffs):
    freqs = [0 for _ in range(len(cutoffs) - 1)]

    for value in values:
        if value == max(values):
            freqs[-1] += 1  # increment the last bins freqiuency
        else:
            # now we need to figure out where this is
            # need to stop one early since we are indexing with i + 1
            for i in range(len(cutoffs) - 1):
                if cutoffs[i] <= value < cutoffs[i + 1]:  # left side closed, right side open
                    # we found it!
                    freqs[i] += 1
    return freqs


def compute_slope_intercept(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    m = 0
    numer = 0
    denom = 0

    for i in range(len(x)):
        x_i = x[i]
        y_i = y[i]
        x_mean_diff = x_i - x_mean
        y_mean_diff = y_i - y_mean
        numer += x_mean_diff * y_mean_diff

    for x_i in x:
        x_mean_diff = x_i - x_mean
        denom += x_mean_diff ** 2

    m = numer / denom
    b = y_mean - m * x_mean
    return m, b


def compute_slope(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numer = sum([(x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x))])
    denom = sum([(x_i - x_mean) ** 2 for x_i in x])

    m = numer / denom
    return m
