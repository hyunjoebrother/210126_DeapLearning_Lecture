####210214 모두를 위한 딥러닝 강좌 시즌
#### Lec 07 - ML model 동작 Application & TIPs

#### Lec 07-1 - 학습 rate, Overfitting, 그리고 일반화 (Regularization)

# Standaraization (Normalizing)

# Overfitting
# : test dataset에 안 맞는 model
# -> 줄여보자 (Solution)
# - more training data
# - reduce the number of features
# - Regularization (일반화)

# Regularization
# : Lets not have too big numbers in the weight
# -> cost를 최적화시킨다 -> by 상수 regularization strength


#### Lec 07-2 - Training/Testing 데이타 셋

# # Data (Original Set)
# : Data 중에서 70% 정도 training set / 나머지 30%는 testing set
# -> Training set으로 model을 train -> testing set으로 test

# # Validation Set 
# : Training set = Training + Validation (training rate, strength 등)
# -> Validation set으로 모의 test

# # Online Learning
# : 많은 data를 그룹으로 쪼개서 train -> 각 그룹의 학습 결과가 남도록 함
# -> 있는 data에 추가로 학습 가능

# # Accuracy
# : 실제 data Y & model이 예측한 data Y hat 값을 비교



#### Lab 07-1 - training/test dataset, learning rate, normalization

import tensorflow as tf

# Session module 지원 안해서 추가
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np


# # Training & Test datasets으로 나누자
# x_data = [
#     [1,2,1], [1,3,2], [1,3,4], [1,5,5], [1,7,5], [1,2,5], [1,6,6], [1,7,7]
# ]
# y_data = [
#     [0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [1,0,0], [1,0,0]
# ]

# # Evaluation our model using this test dataset -> train 완료 후 model test작업
# x_test = [
#     [2,1,1], [3,1,2], [3,3,4]
# ]
# y_test = [
#     [0,0,1], [0,0,1], [0,0,1]
# ]

# X = tf.placeholder("float", [None, 3])
# Y = tf.placeholder("float", [None, 3])
# W = tf.Variable(tf.random_normal([3,3]))
# b = tf.Variable(tf.random_normal([3]))

# hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

# # Correct prediction Test model
# prediction = tf.arg_max(hypothesis, 1)
# is_correct = tf.equal(prediction, tf.arg_max(Y,1))
# accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# # Launch Graph
# with tf.Session() as sess:
#     # Initialize TensorFlow variables
#     sess.run(tf.global_variables_initializer())

#     for step in range(201):
#         cost_val, W_val, _ = sess.run(
#             [cost, W, optimizer], feed_dict = {X: x_data, Y: y_data} # Training set
#         )
#         print(step, cost_val, W_val)

#     # Training 끝 -> Testing set 작업 ON

#     # Prediction
#     print("Prediction : ", sess.run(prediction, feed_dict = {X: x_test}))

#     # Calculate the Accuracy 
#     print("Accuracy : ", sess.run(accuracy, feed_dict = {X: x_test, Y: y_test}))


# Learning rate : NaN!
# : 기존 수업 연습처럼 0.1이나 0.01이 아니라
# Large하면 Overshooting (발산),
# Small하면 train 안 먹히거나, trapping in local minimize 된다

# -> 실습에서 learning_rate 값을 변경하고 Prediction & Accuracy를 확인해보자
# if) 값이 1.5 -> 학습 포기 Cost값이 nan처리
# if) 값이 1e-10 -> Cost값이 동일 -> 학습 중단

# # Non-normalizied inputs 
# -> NaN 발생 -> MinMaxScaler() 함수를 이용하여 data 값을 0~1사이로 조정


#### Lab 07-2 - Meet MNIST Dataset

# : 실제 MNIST Dataset으로 실습해보자! 
# : 우체국 숫자 필기체 인식하기 위한 training

# Reading data & Set variables
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

nb_classes = 10

# MNIST data image of shape 28*28 = 784 pixels
X = tf.placeholder(tf.float32, [None, 784])
# 예측값 : 0~9 digits recongition = 10 classes (by ONE-HOT)
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))


# Softmax
hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

# Test model
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))

# Calcurrate Accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


# Training epoch/batch - Data가 많아서 batch로 나눈다
# parameters
training_epochs = 15 # 전체 dataset을 한 번 돌리는 것 : 1 epoch
batch_size = 100 # 한번에 몇 개씩 train할까

with tf.Session() as sess:
    # Initialize TensorFlow Variables
    sess.run(tf.global_variables_initializer())

    # Training Cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        # iteration 몇 번 돌까 loop
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) # read
            c, _ = sess.run([cost, optimizer], feed_dict = {X: batch_xs, Y: batch_ys})
            avg_cost +=  c / total_batch
            # loop 끝나면 1 epoch 끝난다

    print('Epoch : ', '%04d' % (epoch + 1),
            'Cost = ', '{:.9f}'.format(avg_cost))

# Report results on testing dataset
print("Accuracy: ", accuracy.eval(session = sess, # 유사 sess.run()
feed_dict = {X: mnist.test.images, Y: mnist.test.labels}))

# 출력 결과
# epoch가 지날수록 cost 값이 줄어든다