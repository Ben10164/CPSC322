##############################################
# Programmer: Ben Puryear
# Class: CptS 322-02, Spring 2022
# Programming Assignment #2
# 2/8/2022
# I did not attempt the bonus...
#
# Description: This program completes all the requirements for Part 2 of PA2.
# This means it will utilize MyPyTable to do file I/O as well as compute stats
##############################################
import os

from mypytable import MyPyTable


def main():
    mpg_fname = os.path.join("input_data", "auto-mpg.txt")
    prices_fname = os.path.join("input_data", "auto-prices.txt")
    mpg_ofname = os.path.join("output_data", "auto-mpg-nodups.csv")
    prices_ofname = os.path.join("output_data", "auto-prices-nodups.csv")
    mpg_fname_clean = os.path.join("input_data", "auto-mpg-clean.csv")
    prices_fname_clean = os.path.join("input_data", "auto-prices-clean.csv")
    combined_data_path = os.path.join("output_data", "auto-data.csv")
    combined_data_path_removed_NA = os.path.join(
        "output_data", "auto-data-removed-NA.csv")
    combined_data_path_replaced_NA = os.path.join(
        "output_data", "auto-data-replaced-NA.csv")
    print(mpg_fname, prices_fname)

    # first we are going to opne up the auto-mpg.txt file and read it into a list
    # origin: 1 = US, 2 = Europe, 3 = Japan
    auto_mpg_file = open(mpg_fname, "r")
    auto_mpg_attributes = auto_mpg_file.readline().strip().split(",")

    auto_mpg_list = []
    for line in auto_mpg_file:  # now we can continue where we left off
        auto_mpg_list.append(line.strip().split(","))
    auto_mpg_file.close()

    # now we do the same thing but with the auto-prices.txt file
    auto_prices_file = open(prices_fname, "r")
    auto_prices_attributes = auto_prices_file.readline().strip().split(",")

    auto_prices_list = []
    for line in auto_prices_file:
        auto_prices_list.append(line.strip().split(","))
    auto_prices_file.close()

    auto_mpg = MyPyTable(auto_mpg_attributes, auto_mpg_list)
    auto_prices = MyPyTable(auto_prices_attributes, auto_prices_list)

    """
    duplicates found:
    auto-mpg.txt: [181]
    auto-prices.txt: [21]
        """
    dupe_stats(auto_mpg, "auto-mpg.txt")
    dupe_stats(auto_prices, "auto-prices.txt")
    empty_line()

    # now we remove the duplicate rows
    auto_mpg.drop_rows(auto_mpg.find_duplicates(["car name", "model year"]))
    auto_mpg.save_to_file(mpg_ofname)
    auto_prices.drop_rows(
        auto_prices.find_duplicates(["car name", "model year"]))
    auto_prices.save_to_file(prices_ofname)
    dupe_stats_after(auto_mpg, mpg_ofname)
    dupe_stats_after(auto_prices, prices_ofname)
    empty_line()

    # now i will do a manual clean
    """
    changes made:
    all lines that had NA were removed
    "autoprices":
        on the original outputs line 159, the car name was titled subaru, but no specific subaru car was specified so i deleted the row
        line 170 spelling of chevrolet was incorrect, so i changed it be correct
    auto-mpg:
        No changes :)
    Note: I dont know that much about old cars so i had no idea if a car just didnt exist so i just checked for spelling
    """
    # so now that im done i can open up the files and read them into a new MyPyTable
    auto_mpg_cleaned_file = open(mpg_fname_clean, "r")
    auto_mpg_cleaned_attributes = auto_mpg_cleaned_file.readline().strip().split(",")
    auto_mpg_cleaned_list = []
    for line in auto_mpg_cleaned_file:
        auto_mpg_cleaned_list.append(line.strip().split(","))
    auto_mpg_cleaned_file.close()

    auto_prices_cleaned_file = open(prices_fname_clean, "r")
    auto_prices_cleaned_attributes = auto_prices_cleaned_file.readline().strip().split(",")
    auto_prices_cleaned_list = []
    for line in auto_prices_cleaned_file:
        auto_prices_cleaned_list.append(line.strip().split(","))
    auto_prices_cleaned_file.close()

    auto_mpg_cleaned = MyPyTable(
        auto_mpg_cleaned_attributes, auto_mpg_cleaned_list)
    auto_prices_cleaned = MyPyTable(
        auto_prices_cleaned_attributes, auto_prices_cleaned_list)

    dupe_stats(auto_mpg_cleaned, "auto-mpg-clean.csv")
    dupe_stats(auto_prices_cleaned, "auto-prices-clean.csv")
    empty_line()

    # now we join them
    auto_data = auto_mpg_cleaned.perform_full_outer_join(
        auto_prices_cleaned, ["car name", "model year"])
    auto_data_backup = auto_mpg_cleaned.perform_full_outer_join(
        auto_prices_cleaned, ["car name", "model year"])

    auto_data.save_to_file(combined_data_path)

    print("--------------------------------------------------")
    print("combined table (saved to", combined_data_path, ")")
    print("--------------------------------------------------")
    print("No. of instances: ", auto_data.get_shape()[0])
    print("Duplicates: ", auto_data.find_duplicates(
        ["car name", "model year"]))

    # now onto the summary stats
    print("--------------------------------------------------")
    print("Summary stats")
    print("--------------------------------------------------")
    # first we need to remove rows that have NA
    auto_data.remove_rows_with_missing_values()
    # convert to numeric
    auto_data.convert_to_numeric()
    summary_stats = auto_data.compute_summary_statistics(
        ["mpg", "displacement", "horsepower", "weight", "acceleration", "model year", "msrp"])

    print("Summary stats:")
    summary_stats.pretty_print()

    # now we need to fill the missing values with the collumn average
    auto_data_backup.convert_to_numeric()
    numeric_cols = ["mpg", "displacement", "horsepower",
                    "weight", "acceleration", "model year", "msrp"]
    for col in numeric_cols:
        auto_data_backup.replace_missing_values_with_column_average(col)

    auto_data_backup.save_to_file(combined_data_path_removed_NA)

    print("--------------------------------------------------")
    print("combined table - rows w/missing values removed (saved as auto-data-removed-NA.txt)")
    print("--------------------------------------------------")
    # I already removed the NA
    print("No. of instances: ", auto_data.get_shape()[0])
    print("Duplicates: ", auto_data.find_duplicates(
        ["car name", "model year"]))
    print("Summary stats:")
    summary_stats.pretty_print()

    print("--------------------------------------------------")
    print("combined table - rows w/missing values replaced (saved as auto-data-replaced-NA.txt)")
    print("--------------------------------------------------")
    print("No. of instances: ", auto_data_backup.get_shape()[0])
    print("Duplicates: ", auto_data_backup.find_duplicates(
        ["car name", "model year"]))
    print("Summary stats:")
    summary_stats_replace = auto_data_backup.compute_summary_statistics(
        ["mpg", "displacement", "horsepower", "weight", "acceleration", "model year", "msrp"])
    summary_stats_replace.pretty_print()
    auto_data_backup.save_to_file(combined_data_path_replaced_NA)


def empty_line():
    """prints an empty line"""
    print("--------------------------------------------------")
    print()


def dupe_stats(mypytable, filename):
    """
    will print the number of duplicates in a table, formatted correctly
    """

    print("--------------------------------------------------")
    print(filename)
    print("--------------------------------------------------")
    # the [1] returned for get_shape is the rows size
    print("No. of instances: ", mypytable.get_shape()[0])
    print("Duplicates: ", mypytable.find_duplicates(
        ["car name", "model year"]))


def dupe_stats_after(mypytable, filename):
    """
    will print the number of duplicates in a table, but the string is different when stating it (takes place after)
    """
    print("--------------------------------------------------")
    print("duplicates removed (saved as", filename, ")")
    print("--------------------------------------------------")
    print("No. of instances: ", mypytable.get_shape()[0])
    print("Duplicates: ", mypytable.find_duplicates(
        ["car name", "model year"]))


if __name__ == "__main__":
    main()
