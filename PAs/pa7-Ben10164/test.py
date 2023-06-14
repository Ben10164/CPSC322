# some useful mysklearn package import statements and reloads
from mysklearn.mypytable import MyPyTable
import mysklearn.myevaluation as myevaluation
import mysklearn.myevaluation
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier
import mysklearn.myclassifiers
import mysklearn.mypytable
import mysklearn.myutils as myutils
import importlib

import mysklearn.myutils
importlib.reload(mysklearn.myutils)

# uncomment once you paste your mypytable.py into mysklearn package
importlib.reload(mysklearn.mypytable)

# uncomment once you paste your myclassifiers.py into mysklearn package
importlib.reload(mysklearn.myclassifiers)

importlib.reload(mysklearn.myevaluation)


table = MyPyTable()
table.load_from_file("input_data/tournament_games2016-2021.csv")
data = table.data
header = table.column_names

X = data[:-1]
y = data[-1]


# dummy
# kNN
# naive bayes
# decision tree

# stratified k-fold cross validation(k=10)
X_test_folds, X_train_folds = myevaluation.kfold_cross_validation(
    X, 10, random_state=0)
# use sklearn to