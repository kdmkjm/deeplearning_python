import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from gradient_1d import numerical_diff
def function_2(x):
    return np.sum(x**2)
#    return x[0]**2 + x[1]**2
'''
def numerical_diff(f, x):
    h = 1e-4 # 10의 마이너스 4승 0.0001
    return (f(x + h) - f(x - h)) / (2*h)
'''
