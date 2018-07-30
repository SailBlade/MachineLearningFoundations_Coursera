import numpy as np
import matplotlib.pyplot as mp
import scipy
import math
import sympy
from decimal import Decimal


def DoQuestion6And7():
    loopNum = 5
    array = np.zeros((2))

    def GetPartialD(array):
        # E(u,v) = exp(u) + exp(2v) + exp(uv) + u^2 - 2uv + 2 * (v^2) - 3*u - 2*v
        u = array[0]
        v = array[1]
        uOfPartialD = np.exp(u) + v * np.exp(u * v) + 2 * u - 2 * v - 3
        vOfPartialD = 2 * np.exp(2 * v) + u * np.exp(u * v) -2 * u + 4 * v - 2
        return np.array((uOfPartialD,vOfPartialD))

    eta = 0.01
    print ('Question6 Answer is:' , GetPartialD(array))
    for i in range(loopNum):
        array = array - eta * GetPartialD(array)

    def CalcError(array):
        u = array[0]
        v = array[1]
        return np.exp(u) + np.exp(2*v) + np.exp(u*v) + np.power(u,2) - 2*u*v + 2 * (np.power(v,2)) - 3*u - 2*v

    print ('Question7 Answer is:' , CalcError(array))

if  __name__ == "__main__":
    DoQuestion6And7()