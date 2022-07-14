# -*- coding: utf-8 -*-
"""Alura machine learning with sklearn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1H5zh1fjf-EuSTWiIUd3_W34lD6Tgd4Ig
"""

import pandas as pd

# with panda it is possible to get csv from uris
uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"
data = pd.read_csv(uri)

# you can get the first rows using ".head"
firstRow = data.head(n = 1) # n represents the quantity of rows to get (default is 5)
print(firstRow)

nColumns = {
    "home" : "x1",
    "how_it_works" : "x2",
    "contact" : "x3",
    "bought" : "y"
}

# panda can also rename the columns using a dict
# I decided to rename based on f(x) = y, where inputs will be named "x(n)"" and the output "y"
# but normally I can just let the columns be what they are
data = data.rename(columns = nColumns)

# to get only the desired columns you can do
x = data[["x1", "x2", "x3"]]
firstRowOfX = x.head(n = 1)
print(firstRowOfX)

# when you just need the values of one column, there is no need to use a array
# you can do the following
# y = data["y"]
y = data[["y"]]

firstRowOfY = y.head(n = 1)
print(firstRowOfY)

# to split an array you can just use "position:position"
# when the position is not defined, it will get from the first or up to the last
# array[:position] from the first up to the position
# array[position:] from the position up to the last
trainX = x[:75]
trainY = y[:75]
testX = x[75:]
testY = y[75:]

print("We will train with %d element and test with %d elements" % (len(trainX), len(testX)))

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

model = LinearSVC()
# the y argument in .fit, needs to be an 1d array
# so before/above, you could simply do:
# y = data["y"]
# but I didn't wanna to, so there is a need to specify the column like:
# trainY["y"] # that instead of 
# trainY[["y"]] # will return only an 1d array
model.fit(trainX, trainY["y"])
predictions = model.predict(testX)

accuracy = accuracy_score(testY, predictions) * 100
print("The accuracy is %.2f%%" % accuracy)

"""# Using the library to separate train from test"""

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# the seed is needed to avoid randomness
SEED = 20

trainX, testX, trainY, testY = train_test_split(x, y, random_state = SEED, test_size = 0.25)
print("Let's train with %d elements and test with %d elements" % (len(trainX), len(testX)))

model = LinearSVC()
model.fit(trainX, trainY["y"])
predictions = model.predict(testX)

accuracy = accuracy_score(testY, predictions) * 100
print("The accuracy is %.2f%%" % accuracy)

print(trainY.value_counts())
print(testY.value_counts())

"""# With stratify and test_size"""

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

SEED = 20

# add stratify and test_size
trainX, testX, trainY, testY = train_test_split(
    x,
    y,
    random_state = SEED,
    # test_size
    # represent the proportion of the dataset to include in the test split
    test_size = 0.25,
    # stratify
    # split the dataset into train and test sets in a way
    # that preserves the same proportions of examples in each class
    # as observed in the original dataset
    stratify = y
)

print("Let's train with %d elements and test with %d elements" % (len(trainX), len(testX)))

model = LinearSVC()
model.fit(trainX, trainY["y"])
predictions = model.predict(testX)

accuracy = accuracy_score(testY, predictions) * 100
print("The accuracy is %.2f%%" % accuracy)

print(trainY.value_counts())
print(testY.value_counts())