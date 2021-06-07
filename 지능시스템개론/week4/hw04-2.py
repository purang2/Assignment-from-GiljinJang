# ID: 2021220699
# NAME: Eunchan Lee 
# File name: hw04-2.py
# Platform: Python 3.7 on Windows 10 Spyder4
# Required Package(s): numpy pandas matplotlib scikit-learn tensorflow keras

# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.pardir) #get 'par'ent 'dir'ectory

import numpy as np
from two_layer_net import TwoLayerNet

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
#Use sklearn's digits data 
digits = load_digits()

#Separate train and test data by using train_test_split function
train_images, test_images, train_labels, test_labels = train_test_split(digits.data,digits.target,test_size =0.2)          

#Reduce dimensions
train_images = train_images/17
test_images  = test_images/17

#Fitting variable names to fit the MLP model
x_train = train_images
t_train = train_labels
x_test = test_images
t_test = test_labels
                                                        
'''
dataset info @EunchanLee
#4-2


digits data: 10-jinsu (0 to 9)
Features :integer 0-255 -> integer 0-16 (it means a range of Fixel-value )


x_train = (1437 , 64) -> 80%(1437) images of digits data(8*8)
x_test =  (360 , 64) -> 20%(360) images of digits data(8*8)

t_train = (1437,) -> 1D Array of sef of labels [ ... ]
t_test = (360,)  -> 1D Array of sef of labels [ ... ]


batch_size =100

x_batch = (100, 64)
t_batch = (100, )

iter_per_epoch = 14 (= int(1437/100)) 

Test Acc(%) : 96.6667%
'''


#Digits Data는 input size=64이다 
#hidden_size는 데이터가 작아짐에 따라 임의로 50->24로 줄여보았는데 실험결과 test accuracy 성능이 좋아졌다.
network = TwoLayerNet(input_size=64, hidden_size=24, output_size=10)



#iters_num의 경우 10000번을 반복해서 돌려보니 
#Test acc가 96정도에 계속 수렴하여 과적합을 피하기 위해 2000이 적합하다고 판단함

iters_num, train_size , batch_size, learning_rate = 2000, x_train.shape[0],100, 0.1
train_loss_list, train_acc_list, test_acc_list = [], [], []
iter_per_epoch = max(int(train_size / batch_size), 1)

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

