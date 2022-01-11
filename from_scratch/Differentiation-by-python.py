# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:43:07 2022

@author: PC



미분(Differentiation) to Python! 


"""


import numpy as np 

#[pseudo-code of numerical Diff]

def numerical_diff(f,x):
    h = 1e-4 #0.0001
    return (f(x+h)-f(x-h))/ (2*h)
    

#[편미분 함수(partial derivation)]


#y(x_0,x_1) = x_0^2 + x_1^2
def function_2_variable(x):
    return x[0]**2 + x[1]**2

# delf/del_x0 ex:x0=3, x1=4

def function_tmp(x0):
    return x0*x0 + 4.0**2.0

print(numerical_diff(function_tmp , 3.0))
