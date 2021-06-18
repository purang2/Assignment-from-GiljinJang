# -*- coding: utf-8 -*-
# ID: 2021220699
# NAME: Eunchan Lee 
# File name: hw03-1.py
# Platform: Python 3.7 on Windows 10 Spyder4
# Required Package(s): numpy pandas matplotlib scikit-learn



"""
This example provides how to load the database 
(by downloading from Internet), train the network
weights using scikit-learn’s MLPClassifier package.

"""
# tensorflow와 tf.keras를 임포트합니다
import tensorflow as tf
from tensorflow import keras

# 헬퍼(helper) 라이브러리를 임포트합니다
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)



import warnings

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

#Load data from https://www.openml.org/d/554
X,y = fetch_openml('mnist_784',version=1, return_X_y=True)
X = X /255

#rescale the data, use the traditional train/test split
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()



mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init = .1)

mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train,y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

#plotting filters
fig, axes = plt.subplots(4,4)
#use global min/max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28,28), cmap=plt.cm.gray, vmin=.5 *vmin, vmax=.5 *vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()
                    