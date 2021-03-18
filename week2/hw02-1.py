# ID: 2021220699
# NAME: Eunchan Lee 
# File name: hw02-1.py
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
        convert the output of the neuron to a binary output
        :param x: input features
        :return: 1 if the output for the sample is positive (or zero),
        -1 otherwise
        """
        return np.where(self.f(x) >= 0, 1, -1)

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# download and convert the csv into a DataFrame
df = pd.read_csv(url, header=None)
df.head()


'''

dataset info 

-X label
: 4 features 

-Y label 
0~50: Iris-Setosa
51~100: Iris-Versicolour
101~150: Iris-Virginica


hw2-1.py => Setosa vs Virginica -> 0~50 + 100~150
hw2-2.py => Versicolour vs Virginica -> 50~100 +100~150

'''


# extract the label column
y = df.iloc[:, 4].values
# extract features
x = df.iloc[:, 0:4].values


#Concat for Setosa vs Virginica
y= np.concatenate([y[0:50],y[100:150]])
x= np.concatenate([x[0:50],x[100:150]])


from sklearn.model_selection import train_test_split

# map the labels to a binary integer value
# Setosa(1) or Virginica(-1)
y = np.where(y == 'Iris-setosa', 1, -1)


# standardization of the input features
plt.hist(x[:, 0], bins=100)
plt.title("Features before standardization")
plt.savefig("./before.png", dpi=300)
plt.show()

#standardization
x[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
x[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()
x[:, 2] = (x[:, 2] - x[:, 2].mean()) / x[:, 2].std()
x[:, 3] = (x[:, 3] - x[:, 3].mean()) / x[:, 3].std()

plt.hist(x[:, 0], bins=100)
plt.title("Features after standardization")
plt.show()

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


from sklearn.metrics import accuracy_score
print("accuracy %f" % accuracy_score(classifier.predict(x_test), y_test))