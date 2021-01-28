#### 210126 모두를 위한 딥러닝 강좌 - 기초부터 이론 공부 + 메모장 기록
### https://www.youtube.com/watch?v=BS6O0zOGX4E&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=1
## Lec 00 - ML/DL 수업 개요와 일정
## Lect 01 - 기본적인 ML의 용어와 개념 설정

####210127 모두를 위한 딥런이 강좌 시즌
## Lab 01 - TensorFlow의 설치 및 기본적인 operations


import tensorflow as tf

# Session module 지원 안해서 추가
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# Create a constant op
hello = tf.constant("Hello, TensorFlow!") #hello라는 Node 이름을 준다

# seart a TF session -> 실행하기 위해 session 생성
sess = tf.Session()

# run the op and get result
print(sess.run(hello))

## 출력하면 b'Hello~' 출력 -> b는 Bytes literals String


### Computational Graph

## ex) Node A와 B가 plus로 연결 

## TensorFlow Machanics

#step1) TF를 이용하여 graph build

# Node 생성
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # tf.float32는 implictly
# plus Node
node3 = tf.add(node1, node2) # Node3 = Node1 + Node2

print("node1 : ", node1, "node2 : ", node2)
print("node3 : ", node3)
## 이렇게 하면 Tensor 형태로 나오기 때문에 원하는 결과값이 나오지 않는다
## -> Session을 만든다

sess = tf.Session()

# step2) sess.run으로 graph operation 실행
# sess.run으로 graph를 실행시킨다 -> node가 하나의 상수 취급

# step3) 그 결과로 graph속 어떤 결과값 return
print("sess.run(node1, node2) : ", sess.run([node1, node2]))
print("sess.run(node3): ", sess.run(node3))


### Placeholder
## graph 실행시키는 단계에서 값 주고 싶을 때 (아까는 값이 먼저 주어짐)
## Node를 만들 때 Placeholder라는 특별한 node로 만든다

a = tf.placeholder(tf.float32) # a가 constant 상수가 아닌 placeholder임
b = tf.placeholder(tf.float32)
adder_node = a + b 

sess = tf.Session()

# feed_dict : placeholder에서 받을 값 -> graph 실행
# sess.run(op, feed_dict = {x: x_data})으로 graph 실행
print(sess.run(adder_node, feed_dict = {a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict = {a: [1,3], b: [2,4]})) # 여러 값도 가능