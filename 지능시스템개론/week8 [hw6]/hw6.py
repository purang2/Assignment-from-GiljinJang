# ID: 2021220699
# NAME: Eunchan Lee 
# File name: hw6.py
# Platform: Python 3.8 on Windows 10 Spyder4
# Required Package(s): sys os numpy pandas matplotlib sklearn(scikit-learn)



# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

def train_neuralnet_iris(x_train, t_train, x_test, t_test, 
                         input_size=4, hidden_size=10, output_size=3, 
                         iters_num = 1000, batch_size = 10, learning_rate = 0.1,
                         verbose=True):
    
    network = TwoLayerNet(input_size, hidden_size, output_size)

    # Train Parameters
    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size / batch_size, 1)

    train_loss_list, train_acc_list, test_acc_list = [], [], []

    for step in range(1, iters_num+1):
        # get mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 기울기 계산
        #grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
        grad = network.gradient(x_batch, t_batch) # 오차역전파법 방식(압도적으로 빠르다)

        # Update
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        # loss
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if verbose and step % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print('Step: {:4d}\tTrain acc: {:.5f}\tTest acc: {:.5f}'.format(step, 
                                                                            train_acc,
                                                                            test_acc))
    tracc, teacc = network.accuracy(x_train, t_train), network.accuracy(x_test, t_test)
    if verbose:
        print('Optimization finished!')
        print('Training accuracy: %.2f' % tracc)
        print('Test accuracy: %.2f' % teacc)
    return tracc, teacc



def train_neuralnet_digits(x_train, t_train, x_test, t_test, 
                         input_size=64, hidden_size=128, output_size=10, 
                         iters_num = 5000, batch_size = 64, learning_rate = 0.1,
                         verbose=True):
    
    network = TwoLayerNet(input_size, hidden_size, output_size)

    # Train Parameters
    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size / batch_size, 1)

    train_loss_list, train_acc_list, test_acc_list = [], [], []

    for step in range(1, iters_num+1):
        # get mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 기울기 계산
        #grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
        grad = network.gradient(x_batch, t_batch) # 오차역전파법 방식(압도적으로 빠르다)

        # Update
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        # loss
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if verbose and step % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print('Step: {:4d}\tTrain acc: {:.5f}\tTest acc: {:.5f}'.format(step, 
                                                                            train_acc,
                                                                            test_acc))
    tracc, teacc = network.accuracy(x_train, t_train), network.accuracy(x_test, t_test)
    if verbose:
        print('Optimization finished!')
        print('Training accuracy: %.2f' % tracc)
        print('Test accuracy: %.2f' % teacc)
    return tracc, teacc





'''

HoldOut 1. Incorrect 
'''


digits = load_digits(); nsamples = digits.data.shape[0]


#iris = datasets.load_iris(); nsamples = iris.data.shape[0]
ntestsamples = nsamples * 4 // 10  # `//' is integer division
ntrainsamples = nsamples - ntestsamples	  # 4:6 test:train split
testidx = range(0,ntestsamples); trainidx = range(ntestsamples,nsamples)


train_neuralnet_digits(digits.data[trainidx,:], digits.target[trainidx],
                     digits.data[testidx], digits.target[testidx])


'''

HoldOut 2 . Per-Class Holdout Split

Setosa      1~50 -> 1~20 Test / 21~50 Training
Versicolour 51~100 -> 51~70 Test / 71~100 Training
Virginica   101~150 -> 101~120 Test / 121~50 Training
 

'''


'''
digits = load_digits()
ntestsamples = len(digits.target) * 4 // 10  # '//' integer division , 연산은 40%를 말함
ntestperclass = ntestsamples // 10 


# allocate indices for test and training data
# Bte: boolean index for test data;  ~Bte: logical not, for training data
Bte = np.zeros(len(digits.target),dtype=bool)   # initially, False index
#for c in range(0,10): Bte[range(c*50,c*50+ntestperclass)] = True



train_neuralnet_digits(digits.data[~Bte,:], digits.target[~Bte],
                     digits.data[Bte,:], digits.target[Bte])


'''



'''

HoldOut 3. Holdout Split by Random Sampling

'''



#iris = datasets.load_iris()
#nsamples = iris.data.shape[0]

digits = load_digits(); nsamples = digits.data.shape[0]
ntestsamples = nsamples * 4 // 10  # 4:6 test:train split
# random permutation (shuffling)
Irand = np.random.permutation(nsamples)
Ite = Irand[range(0,ntestsamples)]
Itr = Irand[range(ntestsamples,nsamples)]



train_neuralnet_digits(digits.data[Itr,:], digits.target[Itr],
                     digits.data[Ite,:], digits.target[Ite])








'''

HoldOut 5. Stratified(층으로 나누는) Random Sampling 

Setosa -> Random Subsampling
Versicolour -> Random Subsampling
Virginica -> Random Subsampling

'''


X, y = load_digits(return_X_y=True)


# per-class random sampling by passing y to variable stratify, 
Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.4, shuffle=True, stratify=y)

# check number of samples of the individual classes
print('test: %d %d %d,  '%(sum(yte==0),sum(yte==1),sum(yte==2)),end='')
print('training: %d %d %d'%(sum(ytr==0),sum(ytr==1),sum(ytr==2)))

# due to the random initialization of the weights, the performance varies
# so we have to set the random seed for TwoLayerNet's initialization values
np.random.seed(len(y))


train_neuralnet_digits(Xtr,ytr,Xte,yte)



'''
Part2. Cross Validation -> Holdout Method의 한계
'''

'''
Random Subsampling: K Data Splits
'''


#X, y = datasets.load_iris(return_X_y=True)

X, y = load_digits(return_X_y=True)

# due to the random initialization of the weights, the performance varies
# so we have to set the random seed for TwoLayerNet's initialization values
np.random.seed(len(y))

K = 20
Acc = np.zeros([K,2], dtype=float)
for k in range(K):
    # stratified random sampling
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, shuffle=True, random_state=None, stratify=y)
    Acc[k,0], Acc[k,1] = train_neuralnet_digits(Xtr,ytr,Xte,yte,
                                  verbose = False)
    
  
    
    print('Trial(K) %d: accuracy %.3f %.3f' % (k, Acc[k,0], Acc[k,1]))





