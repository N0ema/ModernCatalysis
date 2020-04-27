import numpy as np
import scipy as sp
from math import *
import sys

from scitools.numpyutils import iseq


def integrate(T, n, u0, f):
    """"Solve simple ODE
        T = Time
        n = steps
        u0 = initial value"""
    t = np.linspace(0, T, n+1)
    h = T/float(n)
    I = f(t[0])
    for k in iseq(1, n-1, 1):
        I += 2*f(t[k])
    I += f(t[-1])
    I *= h/2
    I += u0

    return I

# from scitools.StringFunction import StringFunction


f = lambda t: eval(sys.argv[1])
T = eval(sys.argv[2])
u0 = eval(sys.argv[3])
n = int(sys.argv[4])
# f = StringFunction(f_formula, independent_variables='t')
print("Numerical solution of u'(t)=t**3: ", integrate(T, n, u0, f))