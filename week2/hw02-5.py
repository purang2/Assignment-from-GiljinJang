# ID: 2021220699
# NAME: Eunchan Lee 
# File name: hw02-5.py
# Platform: Python 3.7 on Windows 10 Spyder4
# Required Package(s): numpy pandas matplotlib scikit-learn

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class Perceptron:
    """
    Perceptron neuron
    """

    def __init__(self, learning_rate=0.1):
        """
        instantiate a new Perceptron

        :param learning_rate: coefficient used to tune the model
        response to training data
        """
        self.learning_rate = learning_rate
        self._b = 0.0  # y-intercept
        self._w = None  # weights assigned to input features
        # count of errors during each iteration
        self.misclassified_samples = []

    def fit(self, x: np.array, y: np.array, n_iter=10):
        """
        fit the Perceptron model on the training data

        :param x: samples to fit the model on
        :param y: labels of the training samples
        :param n_iter: number of training iterations 
        """
        self._b = 0.0
        self._w = np.zeros(x.shape[1])
        self.misclassified_samples = []

        for _ in range(n_iter):
            # counter of the errors during this training iteration
            errors = 0
            for xi, yi in zip(x, y):
                # for each sample compute the update value
                update = self.learning_rate * (yi - self.predict(xi))
                # and apply it to the y-intercept and weights array
                self._b += update
                self._w += update * xi
                errors += int(update != 0.0)

            self.misclassified_samples.append(errors)

    def f(self, x: np.array) -> float:
        """
        compute the output of the neuron
        :param x: input features
        :return: the output of the neuron
        """
        return np.dot(x, self._w) + self._b

    def predict(self, x: np.array):
        """
        
        The branch conditions should be modified 
        from 1,-1 to 3,2 for the correct prediction.
        """
        
        return np.where(self.f(x) >= 0, 3, 2)

#url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# download and convert the csv into a DataFrame
df = pd.read_csv("wine.csv", header=None)

#delete the first line[column names]
df = df[1:]
df.head()



'''

dataset info 

-X label
: 13 features 

-Y label 
Wine class -> 1,2,3
1 : 0~58
2 : 59~129
3 : 130~177

hw02-3.py: 1 vs 2
hw02-4.py: 1 vs 3
hw02-5.py: 2 vs 3

'''

# extract the label column
y = df.iloc[:, 0].values
# extract features
x = df.iloc[:, 1:].values


#slice to binary classification
x= np.concatenate([x[59:130],x[130:178]])
y= np.concatenate([y[59:130],y[130:178]])


#str to float
x=x.astype(np.float)
y=y.astype(np.float)


from sklearn.model_selection import train_test_split

#standardization for 13 features
for i in range(13):
    x[:, i] = (x[:, i] - x[:, i].mean()) / x[:, i].std()

'''
plt.hist(x[:, 0], bins=100)
plt.title("Features after standardization")
plt.show()
'''
# split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,
                                                    random_state=0)

# train the model
classifier = Perceptron(learning_rate=0.01)
classifier.fit(x_train, y_train)

# plot the number of errors during each iteration
plt.plot(range(1, len(classifier.misclassified_samples) + 1),
         classifier.misclassified_samples, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Errors')
plt.show()

#Accuracy 
from sklearn.metrics import accuracy_score
print("accuracy %f" % accuracy_score(classifier.predict(x_test), y_test))

