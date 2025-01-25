import numpy as np
import matplotlib.pylab as plt
from gradient_2d import numercial_gradient

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    # 100번 반복
    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numercial_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)

def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])

lr = 0.1
lr2 = 10.0
lr3 = 1e-10
step_num = 100
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)
x2, x2_history = gradient_descent(function_2, init_x, lr=lr2, step_num=step_num)
x3, x3_history = gradient_descent(function_2, init_x, lr=lr3, step_num=step_num)

plt.plot([-5, 5], [0, 0], '--b')
plt.plot([0, 0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'bo')
plt.plot(x2_history[:,0], x2_history[:,1], 'go')
plt.plot(x3_history[:,0], x3_history[:,1], 'ro')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()