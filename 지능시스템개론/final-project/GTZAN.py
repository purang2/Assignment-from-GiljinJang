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

x = data.drop(['filename','length','label'], axis=1) #Features
y = data['label'] #Label



#신경망 설계 [Neuralnet Modeling]




