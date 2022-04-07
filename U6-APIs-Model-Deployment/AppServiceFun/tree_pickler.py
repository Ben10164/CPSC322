import pickle  # a standard libary

# pickling is used for object serialization ad deserialization
# pickle: write a biniary representation of an object to a file (for use later...)
# un/pickle: read a binary representation of an object from a file (to a python object in this memory)

# imagine you've just trained a MyDecisionTreeClassifier
# so you pickle the classifier object
# and unpickle it later in your web app code

# let's do this with the interv tree from DecisionTreeFun

header = ["level", "lang", "tweets", "phd"]
interview_tree_solution = ["Attribute", "level",
                           ["Value", "Junior",
                            ["Attribute", "phd",
                             ["Value", "yes",
                              ["Leaf", "False", 2, 5]
                              ],
                             ["Value", "no",
                              ["Leaf", "True", 3, 5]
                              ]
                             ]
                            ],
                           ["Value", "Mid",
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
packaged_obj = [header, interview_tree_solution]
outfile = open("tree.p", "wb")  # w for write, b for bnary
pickle.dump(packaged_obj, outfile)
outfile.close()
