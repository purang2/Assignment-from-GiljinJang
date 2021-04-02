# ID: 2021220699
# NAME: Eunchan Lee 
# File name: hw04-1.py
# Platform: Python 3.7 on Windows 10 Spyder4
# Required Package(s): numpy pandas matplotlib scikit-learn tensorflow keras sys os

# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.pardir) #get 'par'ent 'dir'ectory

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

from tensorflow import keras
from sklearn.datasets import fetch_openml
#Keras library makes it easy to use Fashion MNIST datasets.

fashion_mnist = keras.datasets.fashion_mnist
#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#read Fashion-MNIST data 
(x_train, t_train), (x_test, t_test) = fashion_mnist.load_data()


# convert (n,28,28) to (n,784) by using reshape
x_train = x_train.reshape(60000,28*28)
x_test  = x_test.reshape(10000,28*28)

#Reduce dimensions
x_train = x_train/255
x_test  = x_test/255
'''
dataset info @EunchanLee
#4-1


x_train = (60000 , 784) -> 60000 images of FashionMNIST(28*28)
x_test =  (10000 , 784) -> 10000 images of FashionMNIST(28*28)

t_train = (60000,) -> 1D Array of sef of labels [ ... ]
t_test = (10000,)  -> 1D Array of sef of labels [ ... ]


batch_size =100

x_batch = (100, 784)
t_batch = (100, )

iter_per_epoch = 600 (60000/100) 

'''

#Fashion MNIST도 MNIST와 동일하게 size=874이므로 동일하게 가져감 
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num, train_size , batch_size, learning_rate = 10000, x_train.shape[0], 100, 0.1
train_loss_list, train_acc_list, test_acc_list = [], [], []
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
  batch_mask = np.random.choice(train_size, batch_size)
  x_batch = x_train[batch_mask]
  t_batch = t_train[batch_mask]
  
  #computing gradients
  grad = network.gradient(x_batch, t_batch) #backpropagation (faster)
  
  #update
  for key in ('W1', 'b1', 'W2', 'b2'):
    network.params[key] -= learning_rate * grad[key]
   
  
  loss = network.loss(x_batch, t_batch)
  train_loss_list.append(loss)
  
  if i % iter_per_epoch == 0:
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    #print(train_acc, test_acc)
    #formatting : 소숫점 4째자리 반올림 (보기좋게)
    print(f"[Accuracy] Train(%): {round(train_acc*100,4)}", end=' ')
    print(f"Test(%): {round(test_acc*100,4)}")


