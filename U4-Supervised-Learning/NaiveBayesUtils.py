# these are some good functions
def p_value(X_train, new_instance, y_train, value):
    """
    Calculates the probability of a value given a training set

    Parameters
    ----------
    X_train : list
        A list of lists of strings. Each inner list represents a training
        instance.
    new_instance : list
        A list of strings. Each string represents an attribute.
    y_train : list
        A list of strings. Each string represents a class label.
    value : string
        A string representing a class label.

    Returns
    -------
    float
        The probability of a value given a training set
    """
    p_X = p_list(X_train, new_instance, y_train, value)
    p_X_Mult = multiply_list(p_X)
    p_yes = p_X_Mult * y_train.count(value)
    return p_yes


def multiply_list(data):
    """
    Multiplies the values in a list

    Parameters
    ----------
    data : list
        A list of floats.

    Returns
    -------
    float
        The product of the values in the list
    """
    product = 1
    for i in range(len(data)):
        product *= data[i]
    return product


def p_list(X_train, new_instance, y_train, value):
    """
    Calculates the probability of each attribute in a training set

    Parameters
    ----------
    X_train : list
        A list of lists of strings. Each inner list represents a training
        instance.
    new_instance : list
        A list of strings. Each string represents an attribute. (X_train[i])
    y_train : list
        A list of strings. Each string represents a class label.
    value : string
        A string representing a class label.

    Returns
    -------
    list
        A list of floats. Each float represents the probability of an
        attribute in a training set.
    """
    perdiction_list = []
    for i in range(len(new_instance)):
        attribute_matches_count = 0
        for j in range(len(X_train)):
            if (new_instance[i] == X_train[j][i]):
                if (y_train[j] == value):
                    attribute_matches_count += 1
        perdiction_list.append(
            attribute_matches_count/y_train.count(value))
    return perdiction_list
