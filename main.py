import numpy as np
from sklearn import model_selection, neighbors, preprocessing
import pandas as pd

# data extraction from the file
df = pd.read_csv('breast-cancer-wisconsin.data.txt')

# data cleansing from the dataset provided
df.replace('?', -9999, inplace = True)
df.drop(['id'], 1, inplace=True)

#to get one value defined on the supervised learning algorithm
x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])


x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print(accuracy)

# Now we are testing our own example which is not is the dataset

testing_example = np.array([10,7,7,6,4,10,4,1,2])
testing_example = testing_example.reshape(1,-1)

prediction = clf.predict(testing_example)
print(prediction)