def get_frequencies(table, header=None, col_name=None):
    if header is not None and col_name is not None:
        col = get_column(table, header, col_name)
    else:
        col = table
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
