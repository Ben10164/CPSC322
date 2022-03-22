import numpy as np
# this is some PA7 starter code

header = ["level", "lang", "tweets", "phd"]

X_train = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"]
]

y_train = ["False", "False", "True", "True", "True", "False",
           "True", "False", "True", "True", "True", "True", "True", "False"]

# how to represent a tree in python?
# 1. nested datastructure (dict, list, etc)
# 2. OOP (e.g. MyTree class)
# We will do a nested list representation
# at elem 0, we have the "data type" (attribute, value, leaf)
# at elem 1, we have the "data" (attribute name, attribute value, class label)
# at elem 2 or more, it depends on the data type:
# example: lets build a tree for the interview dataset

# interview_tree_solution = ["Attribute", "level",
#                                 ["Value","Junior",
#                                     ["Attribute","PHD",
#                                         ["Value","yes",
#                                             ["Leaf","False",2,5] # we put in the proportion values (not needed, but nice for debugging)
#                                         ]
#                                         ["Value","no",
#                                             ["Leaf","True",3,5]
#                                         ]
#                                     ]
#                                 ],
#                                 ["Value","Mid",
#                                     ["Leaf","True",4,14] # 4 of the 14 people are level MID
#                                 ],
#                                 ["Value","Senior",
#                                     ["Attribute","Tweets",
#                                         ["Value","yes",
#                                             ["Leaf","True",2,5]
#                                         ],
#                                         ["Value","no",
#                                             ["Leaf","False",3,5]
#                                         ]
#                                     ]
#                                 ]
#                             ]
interview_tree_solution = ["Attribute", "level",
                           ["Value", "Junior",
                            ["Attribute", "PHD",
                             ["Value", "yes",
                              # we put in the proportion values (not needed, but nice for debugging)
                              ["Leaf", "False", 2, 5]
                              ],
                                ["Value", "no",
                                 ["Leaf", "True", 3, 5]
                                 ]
                             ]
                            ],
                           ["Value", "Mid",
                               # 4 of the 14 people are level MID
                               ["Leaf", "True", 4, 14]
                            ],
                           ["Value", "Senior",
                               ["Attribute", "Tweets",
                                ["Value", "yes",
                                 ["Leaf", "True", 2, 5]
                                 ],
                                   ["Value", "no",
                                    ["Leaf", "False", 3, 5]
                                    ]
                                ]
                            ]
                           ]

# the order of the attribute values in the domain does matter
# also do them alphabetically
attribute_domains = {0: ["Junior", "Mid", "Senior"],
                     1: ["Java", "Python", "R"],  # lang
                     2: ["no", "yes"],  # tweets
                     2: ["no", "yes"]}  # PHD


def tdidt(current_instances, available_attributes):
    # basic approach (uses recursion!!):

    # select an attribute to split on
    attribute = select_attribute(current_instances, available_attributes)
    print("Splitting on:", attribute)
    available_attributes.remove(attribute)
    # this subtree
    tree = ["Attribute", attribute]
    # group data by attribute domains (creates pairwise disjoint partitions)
    #   this is a grouopby where you use the attribute domain, instead of the values
    partitions = partition_instances(current_instances, attribute)
    # for each partition, repeat unless one of the following occurs (base case)
    #    CASE 1: all class labels of the partition are the same => make a leaf node
    #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
    #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
    print(available_attributes)
    return None


def fit_starter_code():  # builds the tree
    # TODO: programmatically create a header
    #   (e.g. ["att0","att1",...] and create an attribute domains dictionary)
    # next, i advise stirching X_train and y_train together
    train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
    # we are gonna do this because we are going to recursivly traverse the base set, and in CASE 1, we look at the class labels
    # now, we can make a copy of the header, because the tdit algo is going to modify the list
    #   when we split on an attribute, we remove it from the available attributes, because ypu cant split on the same attribute twice
    available_attributes = header.copy()
    # recall: python is pass by object reference
    tree = tdidt(train, available_attributes)
    print("tree", tree)
    # note: the unit test for fit, will assert that the tree return == interview_tree_solution
    #   (mind the attribute value order)

    pass


def select_attribute(current_instances, available_attrbutes):
    # TODO:
    # use entropy to calculate and chose the attribute
    # with the smallest E_new

    # for now we will use random attribute selection
    rand_index = np.random.randint(0, len(available_attrbutes))
    print("The random index chosen is", rand_index)
    return available_attrbutes[rand_index]


def partition_instances(current_instances, attribute):
    # group by attribute domain
    #   use the attrobite_domains thing
    pass


fit_starter_code()
