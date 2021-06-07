# ID: 2021220699
# NAME: Eunchan Lee 
# File name: hw03-1.py
# Platform: Python 3.7 on Windows 10 Spyder4
# Required Package(s): numpy pandas matplotlib scikit-learn tensorflow


import warnings

# tensorflow와 tf.keras를 임포트합니다
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier


#Keras library makes it easy to use Fashion MNIST datasets.

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

'''
dataset info  @EunchanLee

train_images = (60000,28,28) 60000 images of 28x28
test_images  = (10000,28,28) 10000 images of 28x28

train_labels = (60000,)      60000 label values of 1D array   
test_labels  = (10000,)      10000 label values of 1D array   


label 

0 ~ 9 exists

0 = T-shirt/top 1 = Trouser 
2 = Pullover    3 = Dress
4 = Coat        5 = Sandal
6 = Shirt       7 = Sneaker
8 = Bag         9 = Ankle boot

'''

# convert (n,28,28) to (n,784) by using reshape
train_images = train_images.reshape(60000,28*28)
test_images  = test_images.reshape(10000,28*28)

#Reduce dimensions
train_images = train_images/255
test_images  = test_images/255

#Fitting variable names to fit the MLP model
X_train = train_images
y_train = train_labels
X_test = test_images
y_test = test_labels


mlp = MLPClassifier(hidden_layer_sizes=(784,), max_iter=10, alpha=1e-4,
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
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()

