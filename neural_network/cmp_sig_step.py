import numpy as np
import matplotlib.pylab as plt

def sigmond(x):
    return 1 / (1 + np.exp(-x))

def step(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmond(x)
y2 = step(x)

plt.plot(x, y1)
plt.plot(x, y2, 'k--')
plt.ylim(-0.1, 1.1)
plt.show()