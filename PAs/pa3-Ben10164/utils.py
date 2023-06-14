##############################################
# Programmer: Ben Puryear
# Class: CptS 322-02, Spring 2022
# Programming Assignment #3
# 2/22/2022
# I did not attempt the bonus...
#
# Description: This file contains the general functions I use in my program.
##############################################

import mypytable


def open_file_to_attribute_data_list(file_name):
    """
    Open a file and return a list of lists of attributes and data
    """

    infile = open(file_name, "r")
    attributes = infile.readline().strip().split(",")
    data = []
    for line in infile:  # now we can continue where we left off
        data.append(line.strip().split(","))
    infile.close()

    return attributes, data


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


def list_to_mypytable(attributes, data):
    """
    Convert a list of lists of attributes and data to a MyPyTable object
    """
    table = mypytable.MyPyTable(attributes, data)
    return table


def get_count_of_attribute(table, attribute):
    """
    Count the number of instances of an attribute in a table with the given label
    """
    count = 0
    col = table.get_column(attribute)
    for val in col:
        if val == 1:
            count += 1
    return count


def convert_percentage_to_decimal(percentage):
    """
    Convert a percentage to a decimal
    """
    percentage = percentage.strip("%")
    return percentage


def get_correlation(list1, list2):
    """
    Calculate the correlation between two lists
    """
    # Calculate the mean of the lists
    mean1 = sum(list1) / len(list1)
    mean2 = sum(list2) / len(list2)

    # Calculate the numerator
    numerator = 0
    for i in range(len(list1)):
        numerator += (list1[i] - mean1) * (list2[i] - mean2)

    # Calculate the denominator
    denominator = 0
    for i in range(len(list1)):
        denominator += (list1[i] - mean1) ** 2

    # Calculate the correlation
    correlation = numerator / denominator

    return correlation


def open_file_to_mypytable(filename):
    """
    Open a file and return a MyPyTable object
    """

    return mypytable.MyPyTable().load_from_file(filename)
