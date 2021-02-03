#### 210203 모두를 위한 딥러닝 강좌 시즌
## Lec 04 - multi-variable linear regression

# multi-variable : 시험 n개의 점수 등등
# # Hypothesis
# H(x1, x2, x3, ...) = w1x1 + w2x2 + w3x3 + ... + b

# # Cost function
# cost(w,b) = (H(x1, x2, x3) - y)^2의 평균

# n개면 수식이 엄청 길어짐
# by Matrix (행렬) - Dot Product (내적)

# # Hypothesis using Matrix
# -> H(X) = XW : 행렬의 곱셈으로 간단하게 표현 가능
# -> Matrix를 쓰면 각각 계산이 아니라 한번에 input 넣고 연산 가능


## Lab 04-1 - multi-variable Linear Regression을 TensorFlow에서 구현하기

import tensorflow as tf
