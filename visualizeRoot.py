import numpy as np
from scipy.optimize import fsolve
from database_pt111 import *

import matplotlib.pyplot as plt

def f(x, y):
    k1_plus = 13e4
    k1_minus = 6e4
    k2_plus = 6e7
    k2_minus = 8e8
    k3_plus = 2e1
    thetastar = 1 - x - y

    return k1_plus * pCO * (1 - x - y) - k3_plus * x * y

def g(x, y):
    k1_plus = 13e4
    k1_minus = 6e4
    k2_plus = 6e7
    k2_minus = 8e8
    k3_plus = 2e1

    thetastar = 1 - x - y
    return 2 * k2_plus * pO2 * pow((1 - x - y), 2) - k3_plus * x * y


x = np.linspace(0.9, 1.1, 500)

pCO = 0.1
pO2 = 1

@np.vectorize
def fy(x):
    x0 = 0.0
    def tmp(y):
        return f(x, y)
    y1, = fsolve(tmp, x0)
    return y1

@np.vectorize
def gy(x):
    x0 = 0.0
    def tmp(y):
        return g(x, y)
    y1, = fsolve(tmp, x0)
    return y1


plt.plot(x, fy(x), x, gy(x))
plt.xlabel('x')
plt.ylabel('y')
plt.ylim([-2, 2])
plt.legend(['fy', 'gy'])
plt.show()
# plt.savefig('images/continuation-1.png')