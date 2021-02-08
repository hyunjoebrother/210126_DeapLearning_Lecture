#### 210205 정통 21학번 신입생 정모로 pass

#### 210206 모두를 위한 딥러닝 강좌 시즌

#### 210207 모두를 위한 딥러닝 강좌 시즌
## Lec 05-1 - Logistic Classification의 가설 함수 정의

# 복습 - Regression (결과값 y 예측하기)
# Hypothesis -> Cost -> Gradient descent


# ## (Binary) Classification
# : 단순 숫자 예측이 아닌 2개 중 하나를 정해줌
# ex. Spam Email Detection spam/ham, Facebook Feed show/hide
# -> 0/1 encoding
# ex. spam(1) or ham(0), show(1) or hide(0)

# -> y값을 0~1 사이로 만들어주는 함수 g(z) : sigmoid function
# : z=WX라 하면, 새로운 H(x) = g(z) 라고 가정하자

# # Logistic Hypothesis
# : Classification의 가설 함수가 된다


####210208 모두를 위한 딥러닝 강좌 시즌
## Lec 05-2 - Logistic Regression의 cost 함수 정의

# ## New Cost function for Logistic
# : cost(W) = C(H(x), y) 값의 평균으로 하자
# -> 형태가 log함수
# if) y=1 -> H(x) = 1이면 cost = 0, H(x) = 0이면 cost 값 무한에 가까워짐 (잘못 예측)
# if) y=0 -> H(x) = 0이면 cost = 0, H(x) = 1이면 cost 값 무한에 가까워짐 (잘못 예측)

# ## Minimize Cost
# -> By Gradient descent algorithm
# -> 똑같이 미분하고 alpha값으로 learning rate 후 값을 update
# * tf.train.GradientDescentOptimizer(a)라는 라이브러리 사용한다


## Lab 05 - TensorFlow로 Logistic Classification의 구현하기

import tensorflow as tf

# Session module 지원 안해서 추가
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Training Data

x_data = [
    [1,2], [2,3], [3,1], [4,3], [5,3], [6,2] # Study hours
]
y_data = [
    [0], [0], [0], [1], [1], [1] # Binary 값 Fail & Pass
] 

X = tf.placeholder(tf.float32, shape = [None, 2]) # 총 N개, x_data 2개씩
Y = tf.placeholder(tf.float32, shape = [None, 1])
W = tf.Variable(tf.random_normal([2,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# Hypothesis using sigmoid function
hypothesis = tf.sigmoid(tf.matmul(X,W) + b) # 행렬 X와 W 곱
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

# Accuracy computation -> 0.5 기준으로 True & False (Binary 식으로)
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))

## Train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict = {X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val) # Finish Training

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict = {X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)


# ## Classifying diabetes 당뇨병
# : csv 파일을 불러오고 학습 후에 예측해보자

import numpy as np

xy = np.loadtext('data~ .csv', delimiter = ',', dtype = np.float32)
# , 로 seperate하고 data type도 지정해줌

# Slicing - list에서 원하는 범위만큼 get
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# placeholder shape check하고 나머지 방법은 동일

X = tf.placeholder(tf.float32, shape = [None, 8])
Y = tf.placeholder(tf.float32, shape = [None, 1])

W = tf.Variable(tf.random_normal([8,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')