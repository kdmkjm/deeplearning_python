import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

input_data = np.random.randn(1000, 100) # 1000개의 데이터
node_num = 100 # 각 은닉층의 노드 수
hidden_layer_size = 5 # 은닉층의 갯수
activations = {} # 활성화 결과 저장

x = input_data
for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # 초깃값 변경 테스트
    #w = np.random.randn(node_num, node_num) * 1
    #w = np.random.randn(node_num, node_num) * 0.01
    #w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num) # Xavier 초깃값
    w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num) # ReLU 에서 가장 Best

    a = np.dot(x, w)

    # 활성화 함수 변경 테스트
    #z = sigmoid(a)
    z = ReLU(a)
    #z = tanh(a)

    activations[i] = z

# 히스토그램 그리기(도수 분포표 중 하나)
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    plt.xlim(-0.1, 1)
    plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()