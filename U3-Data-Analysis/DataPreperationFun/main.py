def get_column(table, header, col_name):
    # col_index = header.index(col_name)
    col_index = get_index(header, col_name)  # NOTE: Uses the new function for safety
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


def main():
    header = ["CarName", "ModelYear", "MSRP"]
    msrp_table = [
        ["ford pinto", 75, 2769],
        ["toyota corolla", 75, 2711],
        ["ford pinto", 76, 3025],
        ["toyota corolla", 77, 2789],
    ]

    print(get_column(msrp_table, header, "CarName"))
    print(get_column(msrp_table, header, "MSRP"))

    print(get_index(header, "CarName"))
    values, counts = get_frequencies(msrp_table, header, "ModelYear")
    print(values, counts)

    # more on attributes:
    # 1. what is the type of the attribute?
    #   e.g. how is it stored?
    #   int, float, str, list, ...
    # 2. what is the attribute's semantic type?
    #   e.g. what does the attribute (and it's value) represent?
    #   domain knowledge!!
    # 3. what is the attribute's measurement scale?
    #   categorical vs continuous scales
    #   nominal: categories without an inherent ordering, e.g. "yes", "no"
    #     Think names or eye color
    #   ordinal: categories with an inherent ordering, e.g. "low", "medium", "high"
    #     Think t-shirt sizes
    #   ratio-scaled: continuous where 0 means absense
    #     Think 0lbs or 0 kelvin
    #   interval: continuous without an inherence absence value
    #     Think 0 F (temp)

    # noisy vs invalid values
    # noisy: valid on the scale, but recorded incorrectly
    #   Think someone fatfingering 81 as their age instead of 18
    # invalid: not valid on the scale
    #   Think if someone responded "Bob" to their age

    # missing values
    # 2 main ways to deal with missing values:
    # 1. Discard them
    #   Generally you only want to consider this when your dataset is large and the missing values quantity is small
    #   NOTE: We never want to do this, throwing away data is bad :(
    # 2. Fill them
    #   2.A. Categorical attribute: use a majority vote system
    #       (e.g. fill with the most frequent value)
    #   2.B. Continuous attribute: use central tendency measure
    #       (e.g. mean, median, mode, etc...)
    #       (Later we will use machine learning and stuff)

    # summary stats
    # min, max
    # mid-range (mid-value, (min + max) / 2)
    # arithmetic mean (sum / amount) (sensitive to outliers)
    # median (mid value of a sorted list)
    # variance
    # low variance: the data is clustered around the mean
    # standard deviation is the square root of the variance
    # NOTE: use np.isclose to compare 2 floating point numbers, or np.allclose for list of floating point numbers


if __name__ == "__main__":
    main()
