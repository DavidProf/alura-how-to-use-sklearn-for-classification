# features (1 yes, 0 no)
# an array with information about the animal
# long hair?
# short leg?
# Does hoof noise?
pigs = [[0, 1, 0], [0, 1, 1], [1, 1, 0]]

dogs = [[0, 1, 1], [1, 0, 1], [1, 1, 1]]

# 1 => pig, 0 => dog
## x and y used as in f(x) = y
train_data_x = pigs + dogs
train_data_y = [1,1,1,0,0,0]

from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(train_data_x, train_data_y)

## predict what it is...
dog_or_pig = [0,1,1] # dog right?
prediction = model.predict([ dog_or_pig ])

print("dog" if prediction[0] == 0 else "pig")

data_to_test_x = [[1,1,1], [1,1,0], [0,1,1]]
# what the result should be
data_to_test_y = [0, 1, 1]

prediction = model.predict(data_to_test_x)

quantity_of_correct_predictions = (prediction == data_to_test_y).sum()

print("Accuracy %.2f%%" % (quantity_of_correct_predictions / len(data_to_test_y) * 100))