import numpy as np
import math
import matplotlib.pyplot as plt

n = 2
m = 8
n_1 = int(n+m / 2) + 1
c_ = np.zeros(n_1)
for i in range(n_1):
    c_[i] = (-1) ** i / math.factorial(i)
m_1 = int(m / 2)
n_1 = int(n / 2)
a_ = np.zeros((m_1, m_1))
for i in range(m_1):
    for j in range(m_1):
        a_[i][j] = c_[n_1 - m_1 + i + j + 1]
y = np.zeros(m_1)
for i in range(m_1):
    y[i] = -c_[n_1 + 1 + i]
x_ = np.linalg.solve(a_, y)
b_ = np.zeros(m_1 + 1)
b_[0] = 1
for i in range(m_1):
    b_[m_1 - i] = x_[i]
a_ = np.zeros(n_1 + 1)
for i in range(n_1 + 1):
    for j in range(i + 1):
        a_[i] += c_[i - j] * b_[j]
r_ = []
x_ = np.arange(-5, 5, 0.01)
for i in x_:
    p = 0
    q = 0
    for j in range(len(a_)):
        p += a_[j] * i ** (2 * j)
    for j in range(len(b_)):
        q += b_[j] * i ** (2 * j)
    r_.append(p/q)
plt.plot(x_, r_, label='R(x)')
plt.plot(x_, np.exp(-x_ ** 2), label='f(x)')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
