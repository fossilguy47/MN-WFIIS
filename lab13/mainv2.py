import numpy as np
from numpy.polynomial import legendre
from numpy.polynomial import laguerre
from numpy.polynomial import hermite
import math
import matplotlib.pyplot as plt

def leggauss_ab(n=20, a=0, b=2):
    assert(n>0)
    x, w = np.polynomial.legendre.leggauss(n)
    x = (b-a) * 0.5 * x+(b+a) * 0.5
    w = w * (b-a) * 0.5
    return x, w


def f(x):
    return x/(4*(x**2) + 1)


def f1(x, k=10):
    return x**k


def f2(x):
    return (math.sin(x))**2


def f3(x):
    return math.sin(x)**4

def analitic_integratef(a=0, b=2):
    return np.log(4*b**2 + 1)/8 - np.log(4*a**2 + 1)/8


def analitic_integratef1(k=10):
    return math.factorial(k)

gauss = []
#podpunkt1
print("#1")
for i in range(2, 20):
    w, t = leggauss_ab(i)
    gauss.append(sum(w*f(t)))
gauss_analitic = analitic_integratef()
#print(gauss)
wdokl1 = gauss_analitic
ar2 = []
#podpunkt2
print('#2')
for i in range(2, 20):
    t, w = np.polynomial.laguerre.laggauss(i)
    ar2.append(sum(w*f1(t)))
wdokl2 = analitic_integratef1()
#podpunkt3
ar3 = []
print('#3')
f2_v1 = np.vectorize(f2)
f3_v1 = np.vectorize(f3)
for i in range(2, 15):
    t, w = np.polynomial.hermite.hermgauss(i)
    ar3.append(sum(w*f2_v1(t))*sum(w*f3_v1(t)))
wdokl3 = 0.1919832644
x_1 = [i for i in range(2, 20)]
x_2 = [i for i in range(2, 15)]
#plt.plot(x_1, gauss)
#plt.xlabel("n")
#plt.ylabel("f1(n)")
#plt.show()
#plt.plot(x_1, [np.math.fabs(gauss[i] - wdokl1) for i in range(len(gauss))])
#plt.yscale("log")
#plt.xlabel("n")
#plt.ylabel("|$f_1-c_{1,a}|$")
#plt.show()
plt.plot(x_1, ar2)
plt.xlabel("n")
plt.ylabel("f2(n)")
plt.show()
plt.plot(x_1, [np.math.fabs(ar2[i] - wdokl2) for i in range(len(ar2))])
plt.yscale("log")
plt.xlabel("n")
plt.ylabel("|$f_2-c_{2,a}|$")
plt.show()
#plt.plot(x_2, ar3)
#plt.xlabel("n")
#plt.ylabel("f3(n)")
#plt.show()
#plt.plot(x_2, [np.math.fabs(ar3[i] - wdokl3) for i in range(len(ar3))])
#plt.yscale("log")
#plt.xlabel("n")
#plt.ylabel("$|f_3-c_{3,a}|$")
#plt.show()
