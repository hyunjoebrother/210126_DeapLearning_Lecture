####210208 모두를 위한 딥러닝 강좌 시즌
## Lec 06-1 - Softmax Regression 기본 개념 소개

# # Multinomial Classification
# : 여러 개의 class가 있을 때 예측
# ex. 공부 시간 hours, 출석 attendence  -> 학점 grade (A, B, C)
# -> class : 학점 A/B/C or NOT
# -> 독립된 Matrix 서로 곱해서 한번에 Hypothesis 계산 가능
# -> 각각 sigmoid 적용하여 Hypothesis 계산하는데 귀찮다? -> 다음 단원


## Lec 06-2 - Softmax classifier의 cost 함수

# # Sigmoid : 0~1 사이 값으로 줄어준다
# # Softmax : 나오는 모든 probabilities 합이 1이 나오도록 계산
# -> "ONE-HOT Encoding" 기법으로 확률 가장 큰 값 -> 하나만 출력

# # Cross Entropy Cost Function
# : 예측이 틀리면 cost가 큰 값
# D(S, L) = - (log함수 합) -> 실질적으로 logistic cost와 같은 꼴
# -> 전체 거리 distance 계산 후 평균을 낸다

# # Gradient descent
# : In same way... 미분하고 learning rate 계산


#### Lab 06-1 - Tensorflow로 Softmax Classificaiton의 구현하기

import tensorflow as tf

# Session module 지원 안해서 추가
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np


# x_data = [
#     [1,2,1,1], [2,1,3,2], [3,1,3,4], [4,1,5,5], [1,7,5,5], [1,2,5,6], [1,6,6,6], [1,7,7,7]
# ]
# y_data = [ # ONE-HOT Encoding -> 2/1/0 이런 식으로 label check
#     [0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [1,0,0]
# ]

# X = tf.placeholder("float", [None, 4])
# Y = tf.placeholder("float", [None, 3]) # label(class) 0/1/2 
# nb_classes = 3

# W = tf.Variable(tf.random_normal([4, nb_classes]), name = 'weight')
# b = tf.Variable(tf.random_normal([nb_classes]), name = 'bias')


# # Softmax function
# hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# # Cross entropy cost function
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)


# # Launch graph
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())

#     for step in range(2001):
#         sess.run(optimizer, feed_dict = {X: x_data, Y: y_data})
#         if step % 200 == 0:
#             print(step, sess.run(cost, feed_dict = {X: x_data, Y: y_data}))


# # Testing & ONE-HOT encoding
# a = sess.run(hypothesis, feed_dict = {X: [
#     [1,11,7,9]
# ]})

# # 확률 가장 큰 값을 check -> 몇 번째 argument가 가장 max인지
# print(a, sess.run(tf.arg_max(a,1)))


# # 여러 개도 가능
# all = sess.run(hypothesis, feed_dict = {X: [
#     [1,11,7,9], [1,3,4,3], [1,1,0,1]
# ]})
# print(all, sess.run(tf.arg_max(all,1)))


####210211 모두를 위한 딥러닝 강좌 시즌
#### Lab 06-2 - Tensorflow로 Fancy Softmax Classificaiton의 구현하기

# softmax_cross_entropy_with_logits 함수
# logits (=score) -> softmax 통과시키면 최종 확률 hypothesis 계산
# -> ONE-HOT인 Y값 label과 logits 값을 함수에 넣어주고 평균을 구한다


## Animal Classification -> 어떤 특징을 통해서 종을 분류

# Predicting animal type based on various features
xy = np.loadtxt('data-04-zoo.csv', delimiter = ',', dtype = np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]] # 마지막 열이 y_data

nb_classes = 7 # 0~6 숫자로 구성

X = tf.placeholder(tf.float32, [None, 16]) # x 특징 16가지
Y = tf.placeholder(tf.int32, [None, 1]) # 0~6, shape = (?,1)

# one_hot으로 바꿔주자 - 0~6, class 7개 중에서 check
Y_one_hot = tf.one_hot(Y, nb_classes) # one hot shape = (?,1,7)

# reshape로 rank하나 줄여서 우리가 원하는 값으로 바꿔준다
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) # shape = (?,7)


W = tf.Variable(tf.random_normal([16, nb_classes]), name = 'weight') # X 입력 16개
b = tf.Variable(tf.random_normal([nb_classes]), name = 'bias') # Y 7개

# Softmax
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# Cross Entropy cost
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y_one_hot)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

# Start Training 
prediction = tf.argmax(hypothesis, 1) # 확률을 label 0~6사이로 만든다
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch Graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        sess.run(optimizer, feed_dict = {X: x_data, Y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict = {X: x_data, Y: y_data})
            print("Step : {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

    # Lets see if we can predict
    pred = sess.run(prediction, feed_dict = {X: x_data})
    # y_data : (N, 1) = fLattten => (N, ) matches pred.shape
    
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
    # True & False로 나타낸다