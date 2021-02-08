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