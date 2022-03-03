import NaiveBayesUtils as utils
import numpy as np
np.random.seed(0)

# Lab Task 2


def labTask2():
    headers = ['outlook', 'temperature', 'humidity', 'windy']
    data = [
        ['Rainy', 'Hot', 'High', 'False'],
        ['Rainy', 'Hot', 'High', 'True'],
        ['Overcast', 'Hot', 'High', 'False'],
        ['Sunny', 'Mild', 'High', 'False'],
        ['Sunny', 'Cool', 'Normal', 'False'],
        ['Sunny', 'Cool', 'Normal', 'True'],
        ['Overcast', 'Cool', 'Normal', 'True'],
        ['Rainy', 'Mild', 'High', 'False'],
        ['Rainy', 'Cool', 'Normal', 'False'],
        ['Sunny', 'Mild', 'Normal', 'False'],
        ['Rainy', 'Mild', 'Normal', 'True'],
        ['Overcast', 'Mild', 'High', 'True'],
        ['Overcast', 'Hot', 'Normal', 'False'],
        ['Sunny', 'Mild', 'High', 'True']
    ]
    plays_golf = ["No", "No", "Yes", "Yes", "Yes", "No",
                  "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]

    new_instance = ['Sunny', 'Mild', 'High', 'False']

    # what is P(plays_golf = Yes|X)
    # p(result=yes|X) = p(X|result=yes)*P*result=yes)

    p_yes_X = utils.p_list(data, new_instance, plays_golf, 'Yes')
    print("p_yes_X:", p_yes_X)
    p_no_X = utils.p_list(data, new_instance, plays_golf, 'No')
    print("p_no_X:", p_no_X)

    # multiply the values in the list with each other
    p_yes = utils.multiply_list(p_yes_X) * plays_golf.count("Yes")
    print("p_yes:", p_yes)
    p_no = utils.multiply_list(p_no_X) * plays_golf.count("No")
    print("p_no:", p_no)
    print("-------------------")

    if(p_yes > p_no):
        print("Yes")
    else:
        print("No")

    print(utils.p_value(data, new_instance, plays_golf, 'Yes'))


def randomize_in_place(alist, parallel_list=None):
    for i in range(len(alist)):
        rand_index = np.random.randint(0, len(alist))  # [0, len(alist))
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]


def randomize_in_place_demo():
    data1 = [1, 2, 3, 4, 5, 6, 7, 8]
    data2 = [1, 2, 3, 4, 5, 6, 7, 8]
    randomize_in_place(data1, data2)
    print(data1)
    print(data2)


labTask2()
