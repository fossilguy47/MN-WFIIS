import matplotlib.pyplot as plt
import math


def f1(x_):
    return math.log(x_ ** 5 + 3 * (x_ ** 2) + x_ + 9)


def f2(x_):
    return x_ ** 6


def iloraz_roznicowy1(x1, x2, f):
    return (f(x2) - f(x1)) / (x2 - x1)


def iloraz_roznicowy2(x1, x2, x3, f):
    return (iloraz_roznicowy1(x2, x3, f) - iloraz_roznicowy1(x1, x2, f)) / (x3 - x1)


h = 0.01
n = 10
x = [-0.5, -0.5 + h, -0.5 + 2*h]
xm_tab = []
ym_tab = []
x_tab = []
y_tab = []
i_tab = []
il1_tab = []
il2_tab = []
for i in range(n):
    i_tab.append(i + 1)
    il1 = iloraz_roznicowy1(x[0], x[1], f1)
    il2 = iloraz_roznicowy2(x[0], x[1], x[2], f1)
    il1_tab.append(il1)
    il2_tab.append(il2)
    xm = (x[0] + x[1]) / 2 - il1 / (2 * il2)
    xm_tab.append(xm)
    ym_tab.append(f1(xm))
    for j in range(len(x)):
        x[j] += h
    if abs(xm - x[0]) > abs(xm - x[2]):
        x[0] = xm
    else:
        x[2] = xm
    x.sort()
for i in range(-150, 150, 1):
    x_tab.append(float(i / 100))
    y_tab.append(f1(i / 100))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x_tab, y_tab, label="f(x)")
plt.scatter(xm_tab, ym_tab, label="f(x)", color="red")
plt.show()
plt.xlabel('k')
plt.ylabel('iloraz roznicowy')
plt.plot(i_tab, il1_tab, label="F(x1, x2)", marker=".")
plt.plot(i_tab, il2_tab, label="F(x1, x2, x3)", marker=".")
plt.legend()
plt.show()
