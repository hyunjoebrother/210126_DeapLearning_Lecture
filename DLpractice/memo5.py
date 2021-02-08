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
# -> 똑같이 미분하고 alpha값으로 training rate 후 값을 update
# * tf.train.GradientDescentOptimizer(a)라는 라이브러리 사용한다


## Lab 05 - TensorFlow로 Logistic Classification의 구현하기
