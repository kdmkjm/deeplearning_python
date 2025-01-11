import sys, os
# path가 아나콘다 설치 위치로 되어있기 때문에 주의
sys.path.append('D:\Github\deeplearning_python')
import numpy as np
from common.functions import sigmoid

# 항등 함수(출력층의 활성화 함수)
def identity_function(x):
    return x

# 0층에서 1층으로 가는 신호 전달
X = np.array([1.0, 0.5])
W1 = np.array([[0.1,0.3,0.5], [0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1

Z1 = sigmoid(A1)

print(A1)
print(Z1)

# 1층에서 2층으로 가는 신호 전달
W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2 = np.array([0.1,0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

print(A2)
print(Z2)

W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1,0.2])

print(Z2.shape)
print(W3.shape)
print(B3.shape)

A3 = np.dot(Z2,W3) + B3
Y = identity_function(A3)
print(Y)