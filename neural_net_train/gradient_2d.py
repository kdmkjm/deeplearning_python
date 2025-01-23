import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
#from gradient_1d import numerical_diff

def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)    
    else:
        return np.sum(x**2, axis=1)
#    return x[0]**2 + x[1]**2
'''
def numerical_diff(f, x):
    h = 1e-4 # 10의 마이너스 4승 0.0001
    return (f(x + h) - f(x - h)) / (2*h)
'''

def _numercial_gradient_no_batch(f, x):
    h = 1e-4
    grad = np.zeros_like(x)    # x와 형상이 같은 배열 생성

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x + h)계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x - h)계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val    # 값 복원
    return grad

def numercial_gradient(f, X):
    if X.ndim == 1:
        return _numercial_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numercial_gradient_no_batch(f, x)

        return grad
    
def tangent_line(f, x):
    d = numercial_gradient(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)

    X = X.flatten()
    Y = Y.flatten()

    grad = numercial_gradient(function_2, np.array([X, Y]))

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()