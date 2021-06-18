# -*- coding: utf-8 -*-
"""
@지능시스템 설계 Final Project @이은찬
"""

#필요한 라이브러리
import pandas as pd 
import numpy as np
import sys 
sys.path.append('..')
import time
import matplotlib.pyplot as plt
import seaborn as sns 
GPU = False 



'''
1. Source Code [교재를 응용]

'''
## 옵티마이저 [SGD, Adam]

class SGD:
    
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class Adam:
   
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
            
 
    
 

 
## MLP 신경망 [TwoLayerNet]             
 
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






## 신경망 레이어 관련 코드 [Affine, SoftmaxWithLoss, Sigmoid, softmax,cross_entropy_error]

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        '''정답 레이블이 원핫 벡터일 경우 정답의 인덱스로 변환'''
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    '''정답 데이터가 원핫 벡터일 경우 정답 레이블 인덱스로 변환'''
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

  

      
## 신경망 모델 트레이닝 코드 [Trainer, remove_duplicate, def clip_grads]

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, x_train, t_train, x_test, t_test, max_epoch=10, batch_size=32, max_grad=None, eval_interval=31):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0
        
        accuracy_list = []
        train_acc_list = []
        test_acc_list = []
        start_time = time.time()
        for epoch in range(max_epoch):
            # 뒤섞기
            idx = np.random.permutation(np.arange(data_size))
            x = x[idx]
            t = t[idx]
            
            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]

                # 기울기 구해 매개변수 갱신
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)  # 공유된 가중치를 하나로 모음
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                #model.accuracy(batch_x, batch_t)
                
                # 평가
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    acc = model.accuracy(batch_x, batch_t)
                    accuracy_list.append(100*acc)
                    train_acc = model.accuracy(x_train, t_train)
                    test_acc = model.accuracy(x_test, t_test)
                    train_acc_list.append(train_acc*100)
                    test_acc_list.append(test_acc*100)
                    
                    #formatting : 소숫점 4째자리 반올림 (보기좋게)
                    print(f"[Accuracy] Train(%): {round(train_acc*100,4)}", end=' ')
                    print(f"Test(%): {round(test_acc*100,4)}")
                    
                    #print('| 에폭 %d |  반복 %d / %d | 시간 %d[s] | 손실 %.2f'
                    print('| 에폭 %d |  반복 %d / %d | 시간 %d[s] | 손실 %.2f | 정확도 %.3f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss, 100*acc
))                  
                    
                    
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0
            
            self.current_epoch += 1
        
        ''' Plot the Accuracy'''
        
        sns.set_style("darkgrid") #Seaborn UI
        plt.title("Model Perfomance!")
        #plt.plot(accuracy_list)
        plt.plot(train_acc_list,label='train')
        plt.plot(test_acc_list, label='test')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()
 
    def plot(self, ylim=None):
        x = numpy.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('반복 (x' + str(self.eval_interval) + ')')
        plt.ylabel('손실')
        plt.show()

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate

def remove_duplicate(params, grads):
    '''
    매개변수 배열 중 중복되는 가중치를 하나로 모아
    그 가중치에 대응하는 기울기를 더한다.
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 가중치 공유 시
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 경사를 더함
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 가중치를 전치행렬로 공유하는 경우(weight tying)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads






'''

2. 데이터 분석 및 전처리


'''


## csv 파일 Read 

data = pd.read_csv('data/features_30_sec.csv')



## Feature(x), Label(y)로 분류하는 작업 

x = data.drop(['filename','length','label'], axis=1) '''Features, #axis = 1을 통해 column 제거 '''
y = data['label'] #Label
x = x.to_numpy()



## 레이블을 원핫벡터로 표현: Ex) 'Pop' = [0 0 0 1 0 0 0 0 0 0]

genres = list(set(y))
t = []
for i in range(len(y)): 
    one_hot_vector = [0]*10 # [0 0 0 0 0 0 0 0 0 0]
    one_hot_vector[genres.index(y.loc[i])] = 1
    t.append(np.array(one_hot_vector))  
t = np.array(t)



## 데이터 정규화[Data Normalize] : 다차원 array indexing 기법 중 column 인덱스를 추출하는 문법을 사용 
x_column_len = len(x[0,:]) # (1000,57) -> x_column_len = 57
for i in range(x_column_len):
    x[:,i] = x[:,i]/max(x[:,i])
    


## 데이터 분리[Train-test-Validation spliting] Train 75 : Test 25
Test_set = np.random.choice(1000,250,replace=False)
Train_set = np.delete(np.arange(1000),Test_set)
np.random.shuffle(Train_set)

x_train = x[Train_set]
x_test = x[Test_set]
t_train = t[Train_set]
t_test = t[Test_set]




'''

3. 신경망 설계 및 훈련 [Neuralnet Modeling and Training]


'''


## 모델, 옵티마이저 세팅 
''' HIDDEN-SIZE, LEARNING-RATE(LR)는 지속적인 수정을 통해 좋은 결과를 내는 값으로 지정하였다 '''
model = TwoLayerNet(input_size=57, hidden_size = 114, output_size =10)
optimizer =Adam(lr=0.018)


## 교재 제공 Trainer Class를 사용하여 Model Training 수행 
GTZAN_MLP = Trainer(model, optimizer)



## Training 실행 
GTZAN_MLP.fit(x,t,x_train, t_train, x_test, t_test, max_epoch=250,batch_size=128)



