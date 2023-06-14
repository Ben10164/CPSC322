from email.errors import NonPrintableDefect
import numpy as np

from mysklearn.myclassifiers import MyNaiveBayesClassifier

import numpy as np
from sklearn.linear_model import LinearRegression

from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from mysklearn.myclassifiers import MySimpleLinearRegressionClassifier,\
    MyKNeighborsClassifier,\
    MyDummyClassifier

# in-class Naive Bayes example (lab task #1)
inclass_example_col_names = ["att1", "att2"]
X_train_inclass_example = [
    [1, 5],  # yes
    [2, 6],  # yes
    [1, 5],  # no
    [1, 5],  # no
    [1, 6],  # yes
    [2, 6],  # no
    [1, 5],  # yes
    [1, 6]  # yes
]
y_train_inclass_example = ["yes", "yes",
                           "no", "no", "yes", "no", "yes", "yes"]

set1_X1 = [[2, 3]]
# then we find the probability of the class "yes"
set1_X1_sol = ["yes"]

set1_naive_bayes_clf = MyNaiveBayesClassifier()
set1_naive_bayes_clf.fit(X_train_inclass_example, y_train_inclass_example)

set1_p_yes_X1 = set1_naive_bayes_clf.predict(set1_X1)

assert set1_p_yes_X1 == set1_X1_sol

# RQ5 (fake) iPhone purchases dataset
iphone_col_names = ["standing", "job_status",
                    "credit_rating", "buys_iphone"]
iphone_table = [
    [1, 3, "fair", "no"],
    [1, 3, "excellent", "no"],
    [2, 3, "fair", "yes"],
    [2, 2, "fair", "yes"],
    [2, 1, "fair", "yes"],
    [2, 1, "excellent", "no"],
    [2, 1, "excellent", "yes"],
    [1, 2, "fair", "no"],
    [1, 1, "fair", "yes"],
    [2, 2, "fair", "yes"],
    [1, 2, "excellent", "yes"],
    [2, 2, "excellent", "yes"],
    [2, 3, "fair", "yes"],
    [2, 2, "excellent", "no"],
    [2, 3, "fair", "yes"]
]

iphone_X = [row[:3] for row in iphone_table]
iphone_y = [row[3] for row in iphone_table]

set2_X1 = [[2, 2, "fair"]]  # standing, job, credit rating
set2_X2 = [[1, 1, "excellent"]]  # standing, job, credit rating

set2_X1_pred_sol = ["yes"]
set2_X2_pred_sol = ["yes"]

set2_naive_bayes_clf = MyNaiveBayesClassifier()
set2_naive_bayes_clf.fit(iphone_X, iphone_y)

set2_X1_pred = set2_naive_bayes_clf.predict(set2_X1)
set2_X2_pred = set2_naive_bayes_clf.predict(set2_X2)

assert set2_X1_pred_sol == set2_X1_pred
assert set2_X2_pred_sol == set2_X2_pred

# Bramer 3.2 train dataset
train_col_names = ["day", "season", "wind", "rain", "class"]
train_table = [
    ["weekday", "spring", "none", "none", "on time"],
    ["weekday", "winter", "none", "slight", "on time"],
    ["weekday", "winter", "none", "slight", "on time"],
    ["weekday", "winter", "high", "heavy", "late"],
    ["saturday", "summer", "normal", "none", "on time"],
    ["weekday", "autumn", "normal", "none", "very late"],
    ["holiday", "summer", "high", "slight", "on time"],
    ["sunday", "summer", "normal", "none", "on time"],
    ["weekday", "winter", "high", "heavy", "very late"],
    ["weekday", "summer", "none", "slight", "on time"],
    ["saturday", "spring", "high", "heavy", "cancelled"],
    ["weekday", "summer", "high", "slight", "on time"],
    ["saturday", "winter", "normal", "none", "late"],
    ["weekday", "summer", "high", "none", "on time"],
    ["weekday", "winter", "normal", "heavy", "very late"],
    ["saturday", "autumn", "high", "slight", "on time"],
    ["weekday", "autumn", "none", "heavy", "on time"],
    ["holiday", "spring", "normal", "slight", "on time"],
    ["weekday", "spring", "normal", "none", "on time"],
    ["weekday", "spring", "normal", "slight", "on time"]
]
train_X = [row[:4] for row in train_table]
train_y = [row[4] for row in train_table]

set3_X1 = [["weekday", "winter", "high", "heavy"]]
set3_X1_pred_sol = ["on time"]

set3_naive_bayes_clf = MyNaiveBayesClassifier()
set3_naive_bayes_clf.fit(train_X, train_y)

set3_X1_pred = set3_naive_bayes_clf.predict(set3_X1)

assert set3_X1_pred_sol == set3_X1_pred

print(set3_X1_pred)
