def get_colum(table, header, col_name):
    col_index = header.index(col_name)
    col = []
    for row in table:
        if row[col_index] != "NA":
            col.append(row[col_index])
    return col


def main():
    header = ["CarName", "ModelYear", "MSRP"]
    msrp_table = [
        ["ford pinto", 75, 2769],
        ["toyota corolla", 75, 2711],
        ["ford pinto", 76, 3025],
        ["toyota corolla", 77, 2789],
    ]

    print(get_colum(msrp_table, header, "CarName"))
    print(get_colum(msrp_table, header, "MSRP"))

    # more on attributes:
    # 1. what is the type of the attribute?
    # e.g. how is it stored?
    # int, float, str, list, ...
    # 2. what is the attribute's semantic type?
    # e.g. what does the attribute (and it's value) represent?
    # domain knowledge!!
    # 3. what is the attribute's measurement scale?
    # categorical vs continuous scales
    # nominal: categories without an inherent ordering, e.g. "yes", "no"
    # ordinal: categories with an inherent ordering, e.g. "low", "medium", "high"


if __name__ == "__main__":
    main()




