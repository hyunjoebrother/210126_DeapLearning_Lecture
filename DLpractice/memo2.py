#### 210128 모두를 위한 딥러닝 강좌 - 기초부터 이론 공부 + 메모장 기록
### https://www.youtube.com/watch?v=BS6O0zOGX4E&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=1

## Lec 02 - Linear Regression의 Hypothesis와 cost 설명

# Hypothesis 가설 
# 어떤 data가 linear한 것다 -> linear한 값을 찾는다
# -> H(x) = Wx + b 일차 방정식 형태가 될 것이라는 '가설'을 세운다

# Cost(loss) function
# 실제 data와 H(x) 사이의 거리 측정
# -> ( H(x) - y )^2 값으로 check (부호 상관없도록 제곱)

# 모든 제곱값에 대한 평균, 일반화 = cost(W,b) 로 표현한다

# 목표 : cost값이 가장 작은minimize W와 b 값을 구해서 training하자


#### 210129 모두를 위한 딥러닝 강좌
## Lab 02 - TensorFlow로 간단한 Linear Regression을 구현

# Hypothesis and Cost function

# 주어진 x에 대해서 예측을 어떻게 할 것인가
# 가설 H(x) = Wx + b
# 예측 cost(W,b) -> 실제 true 값 y 사용 -> cost를 minimize하자

# TensorFlow Machanics

import tensorflow as tf

# step1) TF를 이용하여 graph build

# X and Y data
x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# Hypothesis
hypothesis = x_train * W + b

# Cost Function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# step2) sess.run(op, feed_dict = {x:x_data}) 으로 graph operation 실행
# step3) 그 결과로 graph속 어떤 결과값 return

