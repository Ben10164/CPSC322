import numpy as np
import utils
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
#                                         ],
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
                                ["Value","Junior",
                                    ["Attribute","PHD",
                                        ["Value","yes",
                                            ["Leaf","False",2,5]
                                        ],
                                        ["Value","no",
                                            ["Leaf","True",3,5]
                                        ]
                                    ]
                                ],
                                ["Value","Mid",
                                    ["Leaf","True",4,14]
                                ],
                                ["Value","Senior",
                                    ["Attribute","Tweets",
                                        ["Value","yes",
                                            ["Leaf","True",2,5]
                                        ],
                                        ["Value","no",
                                            ["Leaf","False",3,5]
                                        ]
                                    ]
                                ]
                            ]

# the order of the attribute values in the domain does matter
# also do them alphabetically
attribute_domains = {0: ["Junior", "Mid", "Senior"],
                     1: ["Java", "Python", "R"],  # lang
                     2: ["no", "yes"],  # tweets
                     3: ["no", "yes"]}  # PHD


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
    # print(partitions)

    # for each partition, repeat unless one of the following occurs (base case)
    for att_value, att_partition in partitions.items():  # dictionary
        print("Current att_value", att_value)
        print("Length of partition", len(att_partition))
        value_subtree = ["Value", att_value]

        # CASE 1: all class labels of the partition are the same => make a leaf node
        if len(att_partition) > 0 and all_same_class(att_partition):
            print("CASE 1 all same class")
            # make leaf node

        # CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(att_partition) > 0 and len(available_attributes) == 0:
            print("CASE 2 no more attributes")
            # handle clash with the majority vote leaf node

        # CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(att_partition) == 0:
            print("CASE 3 empty partition")
            # backtrack and replace this attribute node with a majority vote leaf node
        else:
            # none of the previous conditions were true...
            # recurse :)
            # need a .copy here because we cant split on the same attribute twice
            print("Woah recursion")
            subtree = tdidt(att_partition, available_attributes.copy())
            # now that we have this subtree:
            # append subtree to value_subtree, and then tree appropriatly
    return tree


def all_same_class(att_partition):
    # look through all the [-1] and if they all are the same return true
    for i in range(len(att_partition)):
        try:
            if att_partition[i][-1] == att_partition[i+1][-1]:
                continue
            else:
                return False
        except IndexError:
            return True


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
    # print("tree", tree)
    # note: the unit test for fit, will assert that the tree return == interview_tree_solution
    #   (mind the attribute value order)

    pass


def select_attribute(current_instances, available_attrbutes):
    # TODO:
    # use entropy to calculate and chose the attribute
    # with the smallest E_new

    # for now we will use random attribute selection
    rand_index = np.random.randint(0, len(available_attrbutes))

    # * for each available attribute:
    #     * for each value in the attribute's domain (Seinor, Junior, etc...)
    #         * calculate the entropy of that value's partition (E_Seinor, E_Junior, etc...)
    #     * computer E_new, which is the weighted sum of the partition entropies
    # * chose to split on the attribute with the smallest E_new

    # print("The random index chosen is", rand_index)
    return available_attrbutes[rand_index]


# split_attribute is like the groupby attribute
def partition_instances(current_instances, split_attribute):
    # group by attribute domain
    #   use the attrobite_domains thing
    # key (attribute value)[junior, mid, seinor]: value (subtable) [values for junior, values for mid, values for senior]
    partitions = {}
    att_index = header.index(split_attribute)  # e.g. 0
    att_domain = attribute_domains[att_index]  # e.g. ["Junior","Senior","Mid"]
    for att_value in att_domain:
        # make an empty list at the key of att_value (Junior,Seinor, etc...)
        partitions[att_value] = []
        for instance in current_instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)

    # return a dictionary
    return partitions


fit_starter_code()


# tree pruning
# decision trees are notorious for overfitting to the training set
# this might seem like a good thing, but this means that the tree is overfitted to the training set,
#   and therefore the tree will respond well to unseen instances 
# typically to combat this, you post-prune the tree using
# prunning set
# no pruning coding on PA7 btw
#  