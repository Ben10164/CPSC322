##############################################
# Programmer: Ben Puryear
# Class: CptS 322-02, Spring 2022
# Programming Assignment #3
# 2/22/2022
# I did not attempt the bonus...
#
# Description: This is the file that contains the functions that will be used to
# plot the data.
##############################################
import matplotlib.pyplot as plt


def plot_occurance_bar(mypytable, attribute, title=None, limit=None, rotation=0):
    """
    Plot a bar chart of the occurance of a single attribute
    """
    attribute_col = mypytable.get_column(attribute)
    if limit is not None:
        attribute_col = attribute_col[:limit]
    no_dupes = list(set(attribute_col))
    occurance = [attribute_col.count(x) for x in no_dupes]
    plt.bar(no_dupes, occurance)
    if title is not None:
        plt.title(title)
    plt.xlabel(attribute)
    plt.ylabel("Occurance")
    plt.xticks(rotation=rotation)
    plt.show()


def plot_percentage_of_total_pie(mypytable, attributes, title=None, limit=None):
    """
    Plot a pie chart of the percentage of multiple attributes
    """
    attribute_vals = [mypytable.get_column(x) for x in attributes]

    if limit is not None:
        attribute_vals = [x[:limit] for x in attribute_vals]
    attribute_sums = []
    for attribute in attribute_vals:
        attribute_total = 0
        for val in attribute:
            try:
                val = float(val)
            except ValueError:
                continue
            attribute_total += val
        attribute_sums.append(attribute_total)
    plt.pie(attribute_sums, labels=attributes, autopct="%1.1f%%")
    if title is not None:
        plt.title(title)
    plt.show()


def plot_occurance_bar_list(
    data, attribute=None, title=None, limit=None, rotation=0, x_labels=[]
):
    """
    Plot a bar chart of the occurance of a single attribute
    """
    no_dupes = list(set(data))
    occurance = [data.count(x) for x in no_dupes]
    plt.bar(no_dupes, occurance)
    if title is not None:
        plt.title(title)
    plt.xlabel(attribute)
    plt.ylabel("Occurance")
    plt.xticks(x_labels, rotation=rotation)
    plt.show()


def plot_histogram(data, attribute=None, title=None, limit=None, rotation=0):
    """
    Plot a histogram of the occurance of a single attribute
    """
    # 10 bins
    plt.hist(data, bins=10)
    if title is not None:
        plt.title(title)
    plt.xlabel(attribute)
    plt.ylabel("Occurance")
    plt.xticks(rotation=rotation)
    plt.show()


def plot_scatter(data, attribute_x, attribute_y, title=None, limit=None, rotation=0):
    """
    Plot a scatter plot of the occurance of a single attribute
    """
    plt.scatter(data.get_column(attribute_x), data.get_column(attribute_y))
    if title is not None:
        plt.title(title)
    plt.xlabel(attribute_x)
    plt.ylabel(attribute_y)
    plt.xticks(rotation=rotation)
    plt.show()


def plot_scatter_list(x, y, title=None, limit=None, rotation=0):
    """
    Plot a scatter plot of the occurance of a single attribute
    """
    plt.scatter(x, y)
    if title is not None:
        plt.title(title)
    plt.xticks(rotation=rotation)
    plt.show()


def plot_bar_dict(dictionary, title=None, limit=None, rotation=0):
    """
    Plot a bar chart of the occurance of a single attribute
    """
    plt.bar(dictionary.keys(), dictionary.values())
    if title is not None:
        plt.title(title)
    plt.xlabel("Attribute")
    plt.ylabel("Occurance")
    plt.xticks(rotation=rotation)
    plt.show()


def plot_box_and_whisker(data, attribute=None, title=None, limit=None, rotation=0):
    """
    Plot a box and whisker plot of an array
    """
    plt.boxplot(data)
    if title is not None:
        plt.title(title)
    plt.xlabel(attribute)
    plt.ylabel("Occurance")
    plt.xticks(rotation=rotation)
    plt.show()
