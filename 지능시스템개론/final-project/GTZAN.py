# -*- coding: utf-8 -*-
"""
@지능시스템 설계 Final Project
"""

#필요한 라이브러리
import pandas as pd 
import numpy as np
import sys 
sys.path.append('..')
from common.optimizer import SGD
from common.trainer import Trainer 
from common.layers import Affine, Sigmoid, SoftmaxWithLoss

# 교재에서 가져온 Source Code [Adam, TwoLayerNet]
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
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy 
    

        
#데이터 분석 [EDA, Exploratory Data Analysis] 
data = pd.read_csv('data/features_30_sec.csv')

x = data.drop(['filename','length','label'], axis=1) #Features, #axis = 1을 통해 column 제거 
y = data['label'] #Label
x = x.to_numpy()


#레이블을 원핫벡터로 표현: Ex) 'Pop' = [0 0 0 1 0 0 0 0 0 0]

genres = list(set(y))
t = []
for i in range(len(y)): 
    one_hot_vector = [0]*10 # [0 0 0 0 0 0 0 0 0 0]
    one_hot_vector[genres.index(y.loc[i])] = 1
    t.append(np.array(one_hot_vector))  
t = np.array(t)


#데이터 정규화[Data Normalize] : 다차원 array indexing 기법 중 column 인덱스를 추출하는 문법을 사용 
x_column_len = len(x[0,:]) # (1000,57) -> x_column_len = 57
for i in range(x_column_len):
    x[:,i] = x[:,i]/max(x[:,i])
    

#데이터 분리[Train-test-Validation spliting] Train 75: Test 25
Test_set = np.random.choice(1000,250,replace=False)
Train_set = np.delete(np.arange(1000),Test_set)
np.random.shuffle(Train_set)

x_train = x[Train_set]
x_test = x[Test_set]
t_train = t[Train_set]
t_test = t[Test_set]




#신경망 설계 [Neuralnet Modeling]
#model = TwoLayerNet(input_size=57, hidden_size =5000, output_size =10)
model = TwoLayerNet(input_size=57, hidden_size = 114, output_size =10)

#optimizer = Adam()
optimizer =Adam(lr=0.018)
#교재 제공 Trainer Class를 사용하여 Model Training 수행 
GTZAN_MLP = Trainer(model, optimizer)
GTZAN_MLP.fit(x,t,x_train, t_train, x_test, t_test, max_epoch=250,batch_size=128)

'''
from sklearn.metrics import confusion_matrix
# Confusion Matrix

preds = model.predict(x_test)
confusion_matr = confusion_matrix(t_test, preds) #normalize = 'true'
plt.figure(figsize = (16, 9))
sns.heatmap(confusion_matr, cmap="Blues", annot=True, 
            xticklabels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
           yticklabels=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]);
plt.savefig("conf matrix")
'''