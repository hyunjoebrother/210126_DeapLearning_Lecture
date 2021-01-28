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

