# ID: 2021220699
# NAME: Eunchan Lee 
# File name: hw03-2.py
# Platform: Python 3.7 on Windows 10 Spyder4
# Required Package(s): numpy pandas matplotlib scikit-learn tensorflow

# -*- coding: utf-8 -*-

import warnings

# tensorflow와 tf.keras를 임포트합니다
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


#Use sklearn's digits data 
digits = load_digits()

#Separate train and test data by using train_test_split function
train_images, test_images, train_labels, test_labels = train_test_split(digits.data,
                                                                        digits.target,
                                                                        test_size =0.2)
'''
dataset info  @EunchanLee

digits data: 10-jinsu (0 to 9)

Features :integer 0-255 -> integer 0-16

train_images = (1437, 64) 1437(80%) images of 8x8
test_images  = (360,64) 360(20%) images of 8x8

train_labels = (1437,)      1437 label values of 1D array   
test_labels  = (360,)      360 label values of 1D array   


label 

0 ~ 9 


'''


#Reduce dimensions
train_images = train_images/17
test_images  = test_images/17

#Fitting variable names to fit the MLP model
X_train = train_images
y_train = train_labels
X_test = test_images
y_test = test_labels


mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.1)

# this example won't converge because of CI's time constraints, so we catch the
# warning and are ignore it here
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning,
                            module="sklearn")
    mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(8, 8), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()

