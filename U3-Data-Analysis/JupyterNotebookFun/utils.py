# THIS IS USED IN Matplotlib.ipynb

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
    groupby_col_index = header.index(groupby_col_name)  # will use this later
    groupby_col = get_column(table, header, groupby_col_name)
    # a way to get the unique values by converting it to a set {sets have no duplicates}
    # but then we need to convert it back to a list, then sort it
    group_names = list(set(groupby_col)).sort()
    group_subtables = [[] for _ in group_names]  # TODO: Fix

    for row in table:
        groupby_val = row[groupby_col_index]  # e.g. this row's modelyear
        # now we figure out which row this belongs to
        groupby_val_subtable_index = group_names.index(groupby_val)
        # this works because they are parallel
        group_subtables[groupby_val_subtable_index].append(row)

    return group_names, group_subtables
