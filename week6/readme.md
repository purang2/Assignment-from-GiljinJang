# 지능 시스템 설계 Homework 5

[pdf readme를 클릭하세요!](https://github.com/purang2/DeepLearning_Prac/blob/main/week6/hw5-doc.pdf)

> ***Dep: Electronics***
***ID: 2021220699
Name: Eunchan Lee***


---

## **What to do?**


**딥러닝 교재 [밑바닥부터 시작하는 딥러닝] Chapter 6. 학습 관련 기술들의 예제 코드**들은  ***MNIST dataset***에 대한 아래 내용들을 담고 있습니다.

- ***SGD, Momentum, AdaGrad, Adam을 비교***
- **가중치(W) 초기값 잘 정하는 방법**
- **배치 정규화**
- **오버피팅 탈출법 → 가중치 감소(weight decay), 드롭아웃(Dropout)**
- **적절한 하이퍼파라미터 [layer size, batch size, learning rate 등] 값 찾기**



[**6장의 정리]
이번 장에서 배운 것**


**매개변수 갱신 방벙에는 확률적 경사 하강법(SGD) 외에도 모멘텀, AdaGrad, Adam 등이 있음
가중치 초기값을 정하는 방법은 올바른 학습을 하는데 매우 중요
가중치의 초기값은 Xavier 초기값(Sigmoid, tanh)과 He 초기값(ReLU)이 효과적
배치 정규화(normalization)를 이용하면 학습을 빠르게 진행, 초기값에 영향을 덜 받게 됨
오버피팅을 억제하는 정규화(regularization) 기술로는 가중치 감소와 드롭아웃이 있음
하이퍼파라미터 값 탐색은 최적 값이 존재할 법한 범위를 점차 좁히면서 하는 것이 효과적**


위의 예제 코드들을 


- ***Fashion MNIST***
- ***Scikit-learn digits***


**위 두가지** **Dataset에 대해 적용하는 코드를 *ipynb → py*로 만들고** 


 **Code Review 등의 형식으로 Report화 하기**
 
 
 
 ---
 
 
 
## 1. for Fashion-MNIST              [code: hw5-1.py]                 
- hw5-1.py/hw5-doc.pdf를 참조



## 2. for digits              [code: hw5-2.py]                 
- hw5-2.py/hw5-doc.pdf를 참조
 
 
 
 ---

## ☘결론

**데이터 상관없이 데이터를 잘 분석해서 데이터에 맞게 Neural Network의 모델링을 잘하면 잘 된다...** 


**이 과제를 통해서 단순한 딥러닝 오픈소스 모델 이용에 그치치 않고** 

- **모델의 하이퍼파라미터를 잘 짜는 법**
- **모델의 오버피팅을 해소하는 몇가지 기법**
- **모델의 옵티마이저를 고르는 기준**

**등을 실제로 프로그래밍 시에 깐깐히 생각하면서 더 효율을 높이면서 프로그래밍해 볼 수 있을 것 같다.**

## (와 딥러닝 너무 재밌다....😂😂)
