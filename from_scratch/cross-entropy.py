# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 11:41:45 2022

@author: PC

Cross Entropy, How to code? 
learn by Practice coding


CE = - sigma( 1 to k ){ t_k * log (y_k) }

code by numpy

t, y = num array

t는 정답 레이블 

log 0 = -inf 대비 delta (epsilon) 추가
"""

import numpy as np

def cross_entropy(y, t):
    return -np.sum(t*np.log(y))

def cross_entropy_delta(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(delta+y))

#test array, by textbook
t = [0,0,0,1,0,0,0]
y = [0.1,0.05,0.05,0.6,0.07,0.07,0.06]

print(cross_entropy(np.array(y), np.array(t)))
print(cross_entropy_delta(np.array(y), np.array(t)))
