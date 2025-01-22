import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f, x):
    h = 1e-4 # 10의 마이너스 4승 0.0001
    return (f(x + h) - f(x - h)) / (2*h)

def function_1(x):
    return 0.01 * x * x + 0.1 * x

def target_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

x = np.arange(0.0,  20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = target_line(function_1, 5)
tf2 = target_line(function_1, 10)
y2 = tf(x)
y3 = tf2(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.plot(x, y3)
plt.show()