#### 210131 모두를 위한 딥러닝 강좌
## Lec 03 - Linear Regression의 cost 최소화 알고리즘의 원리 설명
## How to minimize cost?

# Simplified Hypothesis - 설명을 위해 간단하게 가정
# H(x) = Wx 이라고 하자 -> b = 0으로 취급
# -> W 값에 따라서 cost(W) 값 그래프를 그려보자


## Gradient descent Algorithm 
# : cost(W) function에서 min값을 찾도록 하는 알고리즘
# -> 기울기가 0인 지점을 찾자

# * How it works?
# : (0,0)부터 시작하여 W와 b를 조절하여 기울기를 계산
# -> by 미분



#### 210201 모두를 위한 딥러닝 강좌
## Lec 03 - Linear Regression의 cost 최소화의 TensorFlow 구현
## How to minimize cost?

# # Simplified Hypothesis - Linear Regression

import tensorflow as tf
import matplotlib.pyplot as plt

# Session module 지원 안해서 추가
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# X = [1,2,3]
# Y = [1,2,3]
# W = tf.placeholder(tf.float32)
# hypothesis = X * W # Simplified

# # Cost function
# cost = tf.reduce_mean(tf.square(hypothesis - Y))

# sess = tf.Session()
# sess.run(tf.global_variables_initializer()) # 변수 초기화

# # Variables for plotting cost function
# W_val = [] # 값을 저장할 list -> to draw graph
# cost_val = []
# for i in range(-30, 50):
#     feed_W = i * 0.1
#     curr_cost, curr_W = sess.run([cost, W], feed_dict = {W:feed_W})

#     # list에 값 넣기
#     W_val.append(curr_W)
#     cost_val.append(curr_cost)

# # Show the cost function
# plt.plot(W_val, cost_val)
# plt.show()


# x_data = [1,2,3]
# y_data = [1,2,3]

# W = tf.Variable(tf.random_normal([1]), name = 'weight')
# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)

# hypothesis = X * W # Simplified
# cost = tf.reduce_mean(tf.square(hypothesis - Y))


# # # Gradient descent Algorithm
# # 수동으로 직접 Minimize
# # W -= Learinig rate * derivative

# learning_rate = 0.1 # alpha 값
# gradient = tf.reduce_mean((W * X - Y) * X) # 알파에 곱해진 수식 (평균)
# descent = W - learning_rate * gradient
# update = W.assign(descent) # 새로운 W값으로


# sess = tf.Session()
# sess.run(tf.global_variables_initializer()) # 변수 초기화

# for step in range(21):
#     sess.run(update, feed_dict = {X: x_data, Y: y_data})
#     print(step, sess.run(cost, feed_dict = {X: x_data, Y: y_data}), sess.run(W))


#### 210202 모두를 위한 딥러닝 강좌

# 미분 수식이 복잡할 때 -> by 명령 optimize 선언
# 미분하지 않아도 가능
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
# train = optimizer.minimize(cost)

## W = 5일 때 optimize 시험해보자


X = [1,2,3]
Y = [1,2,3]

W = tf.Variable(5.0)
hypothesis = X * W # Simplified
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer()) # 변수 초기화

for step in range(100):
    print(step, sess.run(W))
    sess.run(train)