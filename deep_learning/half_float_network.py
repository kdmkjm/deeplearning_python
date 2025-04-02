import sys, os
sys.path.append('D:\Github\deeplearning_python')
import numpy as np
import matplotlib.pyplot as plt
from deep_convnet import DeepConvNet
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()
network.load_params("./deep_learning/deep_convnet_params.pkl")

sampled = 10000
x_test = x_test[:sampled]
t_test = t_test[:sampled]

print("calcurate accuracy (float64) ...")
print(network.accuracy(x_test, t_test))

# float16로 형변환
x_test = x_test.astype(np.float16)
for param in network.params.values():
    param[...] = param.astype(np.float16)

print("calcurate accuracy (float16) ...")
print(network.accuracy(x_test, t_test))