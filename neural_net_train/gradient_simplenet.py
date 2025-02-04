import sys, os
# 노트북 로컬 주소
sys.path.append('C:/Users/1/OneDrive/Documents/GitHub/deeplearning_python')
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)

print(dW)

net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

# 최댓값의 인덱스
np.argmax(p)
print(np.argmax(p))

# 정답레이블
t = np.array([0, 0, 1])
net.loss(x, t)
print(net.loss(x, t))

def f(W):
    return net.loss(x, t)
# f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
print(dW)