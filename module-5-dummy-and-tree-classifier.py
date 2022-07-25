!pip install graphviz==0.10
!apt-get install graphviz

import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv"
data = pd.read_csv(uri)

data.sold = data.sold.map({
    'no' : 0,
    'yes' : 1
})
data.head()

from datetime import datetime

actual_year = datetime.today().year

data['model_age'] = actual_year - data.model_year

data.head()

data['km_per_year'] = data.mileage_per_year * 1.60934
data.head()

import numpy as np
from sklearn.model_selection import train_test_split

data_input = data[["price", "model_age","km_per_year"]]
data_output = data["sold"]

SEED = 5
np.random.seed(SEED)
train_input, test_input, train_output, test_output = train_test_split(data_input, data_output, test_size = 0.25, stratify = data_output)

print("We will train with %d elements and test with %d elements" % (len(train_input), len(test_input)))

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

model = LinearSVC()
model.fit(train_input, train_output)
predictions = model.predict(test_input)

accuracy = accuracy_score(test_output, predictions) * 100
print("The accuracy is %.2f%%" % accuracy)

from sklearn.dummy import DummyClassifier

dummy_stratified = DummyClassifier(strategy="stratified")
dummy_stratified.fit(train_input, train_output)
accuracy = dummy_stratified.score(test_input, test_output) * 100

print("The DummyClassifier stratified accuracy is %.2f%%" % accuracy)

from sklearn.dummy import DummyClassifier

dummy_mostfrequent = DummyClassifier(strategy="most_frequent")
dummy_mostfrequent.fit(train_input, train_output)
accuracy = dummy_mostfrequent.score(test_input, test_output) * 100

print("The DummyClassifier most_frequent accuracy is %.2f%%" % accuracy)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SEED = 5
np.random.seed(SEED)

raw_train_input, raw_test_input, train_output, test_output = train_test_split(data_input, data_output, test_size = 0.25, stratify = data_output)

print("We will train with %d elements and test with %d elements" % (len(raw_train_input), len(raw_test_input)))

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

scaler = StandardScaler()
scaler.fit(raw_train_input)
train_input = scaler.transform(raw_train_input)
test_input = scaler.transform(raw_test_input)

model = SVC()
model.fit(train_input, train_output)
predictions = model.predict(test_input)

accuracy = accuracy_score(test_output, predictions) * 100
print("The accuracy is %.2f%%" % accuracy)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SEED = 5
np.random.seed(SEED)

raw_train_input, raw_test_input, train_output, test_output = train_test_split(data_input, data_output, test_size = 0.25, stratify = data_output)

print("We will train with %d elements and test with %d elements" % (len(raw_train_input), len(raw_test_input)))

from sklearn.tree import DecisionTreeClassifier

# max_depth is: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
model = DecisionTreeClassifier(max_depth=3)
model.fit(raw_train_input, train_output)
predictions = model.predict(raw_test_input)

accuracy = accuracy_score(test_output, predictions) * 100
print("The accuracy is %.2f%%" % accuracy)

from sklearn.tree import export_graphviz
import graphviz

features = data_input.columns
dot_data = export_graphviz(model, out_file=None,
                           filled = True, rounded = True,
                           feature_names = features,
                          class_names = ["no", "yes"])
graph = graphviz.Source(dot_data)
graph