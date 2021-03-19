## week 3 Homework 
- hw1 사이킷런 MLP 모델을 통해서 Fashion-MNIST 분류를 하는 모델을 구현함

### 🙄의문점?
- Hidden Layer의 Size는 어떻게 정하는게 맞는가?-> 휴리스틱?..



-------


### 🐷 Let's find the best number of Hidden layer size!

<img src="images/ttb.PNG" width="70%" height="70%">

|Rank#|HiddenLayer Size|Traning Score|Test Score|
|----|----|----|-----|
|⭐1|784|0.919183|0.885600|
|⭐2|3136|0.917667|0.883200|
|⭐3|100|0.911950|0.878100|
|4|256|0.911900|0.877300|
|5|128|0.911367|0.875700|
|6|50|0.903200|0.866500|
|7|25|0.876283|0.853200|
|8|32|0.873917|0.848900|

- TRY1 : Size 50
<img src="images/TRY1_50.PNG" width="60%" height="60%">

- TRY2 : Size 100
<img src="images/TRY2_100.PNG" width="80%" height="80%">

- TRY3 : Size 25
<img src="images/TRY3_25.PNG" width="80%" height="80%">

- TRY4 : Size 128
<img src="images/TRY4_128.PNG" width="80%" height="80%">

- TRY5 : Size 784
<img src="images/TRY5_784.PNG" width="80%" height="80%">

- TRY6 : Size 3136
<img src="images/TRY6_3136.PNG" width="60%" height="60%">

- TRY7 : Size 32

<img src="images/TRY7_32.PNG" width="80%" height="80%">

😅Google's pick, but..
<img src="images/google_pick.PNG" width="50%" height="50%">
 
 
- TRY8 : Size 256

<img src="images/TRY8_256.PNG" width=width="80%" height="80%">






