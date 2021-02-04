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

# Session module 지원 안해서 추가
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# x1_data = [73., 93., 89., 96., 73.]
# x2_data = [80., 88., 91., 98., 66.]
# x3_data = [75., 93., 90., 100., 70.]
# y_data = [152., 185., 180., 196., 142.]

# x1 = tf.placeholder(tf.float32)
# x2 = tf.placeholder(tf.float32)
# x3 = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)

# w1 = tf.Variable(tf.random_normal([1]), name = 'weight1')
# w2 = tf.Variable(tf.random_normal([1]), name = 'weight2')
# w3 = tf.Variable(tf.random_normal([1]), name = 'weight3')
# b = tf.Variable(tf.random_normal([1]), name = 'bias')

# hypothesis = x1*w1 + x2*w2 + x3*w3 + b

# cost = tf.reduce_mean(tf.square(hypothesis - Y))

# # Minimize - Need a very small learning rate for this data set)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5) # 0.001?
# train = optimizer.minimize(cost)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# for step in range(2001):
#     cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
#         feed_dict = {x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
    
#     if step % 10 == 0:
#         print(step, "Cost: ", cost_val, "\nPrediction: \n", hy_val)


# -> training 결과 : Cost가 감소하며 Prediction 값이 Y에 수렴한다
# -> instance 많으면 복잡

## by Matrix로 더 간단하게 구현하자

x_data = [
    [73., 80., 75.], [93., 88., 93.], [89., 91., 90.],
    [96., 98., 100.], [73., 66., 70.]
]
y_data = [
    [152.], [185.], [180.], [196.], [142.]
]

# Shape - None : 원하는만큼 n개 줄 수 있다. 각 element 값은 3 or 1개이다
X = tf.placeholder(tf.float32, shape = [None, 3])
Y = tf.placeholder(tf.float32, shape = [None, 1])

W = tf.Variable(tf.random_normal([3,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = tf.matmul(X,W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5) # 0.001?
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
        feed_dict = {X: x_data, Y: y_data})
    
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction: \n", hy_val)



#### 210204 모두를 위한 딥러닝 강좌 시즌
## Lab 04-2 - TensorFlow로 파일에서 data 읽어오기

# data가 많아지니까 코드에 직접 쓰기 힘들다 

import numpy as np

xy = np.loadtxt('data-01-test-score.csv', delimiter = ',', dtype = np.float32)
# , 로 seperate하고 data type도 지정해줌

# Slicing - list에서 원하는 범위만큼 get
x_data = xy[:, 0:-1] # : (전체 n개)
y_data = xy[:, [-1]]

# data 확인용
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)


## Queue Runners 
# numpy로도 힘든 data -> Queue에 필요한만큼 쌓고 꺼내서 쓴다


# step1 : file들을 리스트로 만든다
filename_queue = tf.train.string_input_producer(
    ['data-01-test-score.csv', '', ],
    shuffle = False, name = 'filename-queue')

# step2 : file을 읽어오는 reader 
reaer = tf.TextLineReader()
key, value = reader.read(filename_queue)

# step3 : value값을 이해(파싱)하기 위한 decode_csv 가져오고 data type 정함
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults = record_defaults)

# tf.train.batch : x_data와 y_data 가져오고 size 지정
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size = 10)

# batch는 tf므로 sess.run으로 실행시키고, feed-dict로 값을 넘겨준다
for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
