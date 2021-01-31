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

# Session module 지원 안해서 추가
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# # step1) TF를 이용하여 graph build

# # 학습할 X and Y data 주어짐 : x가 1일 때 y가 1
# x_train = [1,2,3]
# y_train = [1,2,3]

# # W와 b 값을 정의하자 - tf가 사용할 variable -> trainable한 변수값
# W = tf.Variable(tf.random_normal([1]), name = 'weight') # random값인 shape와 rank 지정
# b = tf.Variable(tf.random_normal([1]), name = 'bias')

# # Hypothesis
# hypothesis = x_train * W + b

# # Cost Function
# cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # reduce_mean : 평균

# # Minimize - GradientDescent 방식
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
# train = optimizer.minimize(cost) # 일종의 Node


# # step2) sess.run(op, feed_dict = {x:x_data}) 으로 graph operation 실행

# # Launch the graph in a session
# sess = tf.Session()

# # Initializes global variables(W, b) in the graph
# sess.run(tf.global_variables_initializer())


# # step3) 그 결과로 graph속 어떤 결과값 return

# # Fit the line
# for step in range(2001):
#     sess.run(train) # Node 실행
#     if step % 20 == 0: # step을 20번에 한번씩 출력해라
#         print(step, sess.run(cost), sess.run(W), sess.run(b))


## 실행 결과 (step - cost - W - b)

# 첫 실행 때 cost가 아주 크고, W와 b는 random

# 학습이 진행될수록 cost 값이 작아지고
# W = 1, b = 0에 수렴된다

# -> 자동적으로 optimize를 실행시키면 
# -> train이 일어나서 TF가 W와 b값을 조절한다

## data 가설 H(x) = x -> W = 1, b = 0 꼴임 -> ok


## Placeholders
# 직접 data 주지 않고 필요할 때 data 던져줌


X = tf.placeholder(tf.float32, shape = [None]) # x_train, y_train 대신
Y = tf.placeholder(tf.float32, shape = [None]) # None : 아무 값

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = X * W + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001): # _ : train 값은 필요없다
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                # 위에서 Linear Regression model 만들고 여기서 data 준다
                feed_dict = {X: [1,2,3,4,5], Y: [2.1,3.1,4.1,5.1]})    

                ## data 가설 H(x) = 1x + 1.1 -> W = 1, b = 1.1 꼴임 -> ok 

    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

# Testing our model
print(sess.run(hypothesis, feed_dict = {X : [5]}))
print(sess.run(hypothesis, feed_dict = {X : [2.5]}))
print(sess.run(hypothesis, feed_dict = {X : [1.5, 3.5]}))