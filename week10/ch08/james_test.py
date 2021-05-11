# coding: utf-8
import sys, os
# tensorflow와 tf.keras를 임포트합니다
import tensorflow as tf
from tensorflow import keras

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from deep_convnet import DeepConvNet
#from dataset.mnist import load_mnist



from deep_convnet import DeepConvNet
from common.trainer import Trainer


from sklearn.datasets import fetch_openml
#Fashion MNIST를 불러온다.
fashion_mnist = keras.datasets.fashion_mnist
(x_train, t_train), (x_test, t_test) = fashion_mnist.load_data()


#입력 데이터 전처리 
#차원 추가를 통해 3D -> 4D & reshape 과정 (60000,28,28) -> (60000,1 ,28,28)
x_train= x_train[np.newaxis]
x_test= x_test[np.newaxis]

x_train = x_train.reshape(len(x_train[0]),1,28,28)
x_test = x_test.reshape(len(x_test[0]),1,28,28)



x_train = x_train/255
x_test = x_test/255
#Train Fashion MNIST!
network = DeepConvNet()  



## 가중치들 조절하면서 성능 뽑아내기 
# ex) epochs = 12 , batch_size =128 ... 

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=12, mini_batch_size=128,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# 매개변수 보관
network.save_params("fMNIST_deep_convnet_params.pkl")
print("Saved Network Parameters!")






'''

network = DeepConvNet()
network.load_params("fMNIST_deep_convnet_params.pkl")

print("calculating test accuracy ... ")



sampled = 1000
#x_test = x_test[:sampled]
#t_test = t_test[:sampled]

classified_ids = []

acc = 0.0
batch_size = 100

for i in range(int(x_test.shape[0] / batch_size)):
    tx = x_test[i*batch_size:(i+1)*batch_size]
    tt = t_test[i*batch_size:(i+1)*batch_size]
    y = network.predict(tx, train_flg=False)
    y = np.argmax(y, axis=1)
    classified_ids.append(y)
    acc += np.sum(y == tt)
    
acc = acc / x_test.shape[0]
print("test accuracy:" + str(acc))

classified_ids = np.array(classified_ids)
classified_ids = classified_ids.flatten()
 
max_view = 20
current_view = 1

fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2)

mis_pairs = {}
for i, val in enumerate(classified_ids == t_test):
    if not val:
        ax = fig.add_subplot(4, 5, current_view, xticks=[], yticks=[])
        ax.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
        mis_pairs[current_view] = (t_test[i], classified_ids[i])
            
        current_view += 1
        if current_view > max_view:
            break

print("======= misclassified result =======")
print("{view index: (label, inference), ...}")
print(mis_pairs)

plt.show()
'''