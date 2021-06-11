# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 11:25:23 2021


@지능시스템 설계 Final Project

"""
import pandas as pd 
import numpy as np


import sys 
sys.path.append('..')
from common.optimizer import SGD
from common.trainer import Trainer 
from common.layers import Affine, Sigmoid, SoftmaxWithLoss

class Adam:
    '''
    Adam (http://arxiv.org/abs/1412.6980v8)
    '''
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])
            
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)
            
            
class TwoLayerNet:
    
    def __init__(self, input_size, hidden_size, output_size):
        i,h,o = input_size, hidden_size, output_size
        
        
        W1 = 0.01 * np.random.randn(i,h)
        b1 = np.zeros(h)
        W2 = 0.01 * np.random.randn(h,o)
        b2 = np.zeros(o)
        
        
        self.layers = [
            Affine(W1,b1),
            Sigmoid(),
            Affine(W2,b2)]
        self.loss_layer = SoftmaxWithLoss() #⭐
        
        self.params, self.grads =[],[]
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
        
    
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss 
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
        
        
#데이터 분석 [EDA, Exploratory Data Analysis] 

data = pd.read_csv('data/features_30_sec.csv')

x = data.drop(['filename','length','label'], axis=1) #Features, #axis = 1을 통해 column 제거 
y = data['label'] #Label



#Label to int(0~9) 
genres = list(set(y))

t = []

for i in range(len(y)):
    
    one_hot_vector = [0]*10 
    one_hot_vector[genres.index(y.loc[i])] = 1
    #y[i] = np.array(one_hot_vector)
    t.append(np.array(one_hot_vector))
    #y.loc[i] = genres.index(y.loc[i])
#y = y.to_numpy()   
t = np.array(t)


x = x.to_numpy()

#y = y.to_numpy(dtype="int")
#y = y.to_numpy(dtype="array")

#신경망 설계 [Neuralnet Modeling]
model = TwoLayerNet(input_size=57, hidden_size =50, output_size =10)
optimizer = SGD(lr=1)
#optimizer = Adam()

#교재 제공 Trainer Class를 사용하여 Model Training 수행 
GTZAN_MLP = Trainer(model, optimizer)
#GTZAN_MLP.fit(x,y, max_epoch=30)
GTZAN_MLP.fit(x,t, max_epoch=30)



