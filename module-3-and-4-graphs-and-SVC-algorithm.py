!pip install seaborn==0.9.0

import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"
data = pd.read_csv(uri)
data.head()

map = {
    0 : 1,
    1 : 0
}

data['finished'] = data.unfinished.map(map)
data.head()

data.tail()

import seaborn

# draw a graph (default config)
seaborn.scatterplot(x="expected_hours", y="price", data=data)

# draw a graph but split finished or not by color (hue option)
seaborn.scatterplot(x="expected_hours", y="price", data=data, hue="finished")

# split by color and put in different graphs
seaborn.relplot(x="expected_hours", y="price", data=data, hue="finished", col="finished")

data_input = data[['expected_hours', 'price']] # "dictionary" with arrays (DataFrame in this case)
data_output = data['finished'] # a simple array

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy

SEED = 5
# sklearn uses numpy, so it's easier to set here
numpy.random.seed(SEED)

train_input, test_input, train_output, test_output = train_test_split(data_input, data_output, test_size = 0.25, stratify = data_output)

print("We will train with %d elements and test with %d elements" % (len(train_input), len(test_input)))

model = LinearSVC()
# this gives a warning because data needs to be normalized
model.fit(train_input, train_output)
predictions = model.predict(test_input)


accuracy = accuracy_score(test_output, predictions) * 100

print("The accuracy is %.2f%%" % accuracy)

base_predictions = numpy.ones(540)
accuracy = accuracy_score(test_output, base_predictions) * 100
print("The baseline algorithm accuracy is %.2f%%" % accuracy)

# graph for test data (data that we know the answers)
seaborn.scatterplot(x="expected_hours", y="price", data=test_input, hue=test_output)

min_expected_hours = test_input.expected_hours.min() # min axis X
max_expected_hours = test_input.expected_hours.max() # max axis X
min_price = test_input.price.min() # min axis Y
max_price = test_input.price.max() # max axis Y

print("Expected hours: min(%d), max(%d)\nPrice: min(%d), max(%d)" % (min_expected_hours, max_expected_hours, min_price, max_price))

# let's "normalize"

# kinda arbitrary...
pixels = 100

graph_axis_x = numpy.arange(min_expected_hours, max_expected_hours, (max_expected_hours - min_expected_hours) / pixels)
graph_axis__y = numpy.arange(min_price, max_price, (max_price - min_price) / pixels)

print(graph_axis_x, graph_axis__y)

# not really sure to be honest, but something along the lines:
# let's put it together in the same Matrix to represent a graph?
xx, yy = numpy.meshgrid(graph_axis_x, graph_axis__y)
dots = numpy.c_[xx.ravel(), yy.ravel()]
dots

Z = model.predict(dots)
Z = Z.reshape(xx.shape)
Z

import matplotlib.pyplot as plt

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(test_input.expected_hours, test_input.price, c=test_output, s=1)

"""# DECISION BOUNDARY"""

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy

SEED = 5
# sklearn uses numpy, so it's easier to set here
numpy.random.seed(SEED)

train_input, test_input, train_output, test_output = train_test_split(data_input, data_output, test_size = 0.25, stratify = data_output)

print("We will train with %d elements and test with %d elements" % (len(train_input), len(test_input)))

model = SVC()
model.fit(train_input, train_output)
predictions = model.predict(test_input)

accuracy = accuracy_score(test_output, predictions) * 100
print("The accuracy is %.2f%%" % accuracy)

min_expected_hours = test_input.expected_hours.min() # min axis X
max_expected_hours = test_input.expected_hours.max() # max axis X
min_price = test_input.price.min() # min axis Y
max_price = test_input.price.max() # max axis Y

print("Expected hours: min(%d), max(%d)\nPrice: min(%d), max(%d)" % (min_expected_hours, max_expected_hours, min_price, max_price))

# let's "normalize"

# kinda arbitrary...
pixels = 100

graph_axis_x = numpy.arange(min_expected_hours, max_expected_hours, (max_expected_hours - min_expected_hours) / pixels)
graph_axis__y = numpy.arange(min_price, max_price, (max_price - min_price) / pixels)

xx, yy = numpy.meshgrid(graph_axis_x, graph_axis__y)
dots = numpy.c_[xx.ravel(), yy.ravel()]

Z = model.predict(dots)
Z = Z.reshape(xx.shape)

import matplotlib.pyplot as plt

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(test_input.expected_hours, test_input.price, c=test_output, s=1)

"""---"""

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

SEED = 5
numpy.random.seed(SEED)

raw_train_input, raw_test_input, train_output, test_output = train_test_split(data_input, data_output, test_size = 0.25, stratify = data_output)
print("We will train with %d elements and test with %d elements" % (len(raw_train_input), len(raw_test_input)))

# normalize the data for SVC
scaler = StandardScaler()
# Compute the mean and std to be used for later scaling.
scaler.fit(raw_train_input)
# normalize
train_input = scaler.transform(raw_train_input)
test_input = scaler.transform(raw_test_input)

model = SVC()
model.fit(train_input, train_output)
predictions = model.predict(test_input)

accuracy = accuracy_score(test_output, predictions) * 100
print("The accuracy is %.2f%%" % accuracy)

train_input

test_input_expect_hours = test_input[:,0]
test_input_price = test_input[:,1]

min_expect_hours = test_input_expect_hours.min() # min from axis X
max_expect_hours = test_input_expect_hours.max() # max from axis X
min_price = test_input_price.min() # min from axis Y
max_price = test_input_price.max() # max from axis Y

# some normalization to put in a graph (something related to the line?)
pixels = 100
eixo_x = numpy.arange(min_expect_hours, max_expect_hours, (max_expect_hours - min_expect_hours) / pixels)
eixo_y = numpy.arange(min_price, max_price, (max_price - min_price) / pixels)

xx, yy = numpy.meshgrid(eixo_x, eixo_y)
dots = numpy.c_[xx.ravel(), yy.ravel()]

Z = model.predict(dots)
Z = Z.reshape(xx.shape)

import matplotlib.pyplot as pyplot

pyplot.contourf(xx, yy, Z, alpha=0.3)
pyplot.scatter(test_input_expect_hours, test_input_price, c=test_output, s=1)