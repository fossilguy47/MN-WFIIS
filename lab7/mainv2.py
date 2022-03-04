import numpy as np
import matplotlib.pyplot as plt


def count_wzo_New(a, x, i_x, size):
    sum_temp = 0.0
    for i in range(size):
        it_temp = 1.0
        for j in range(i):
            it_temp = it_temp * (i_x - x[j])
        sum_temp = sum_temp + a[i][i] * it_temp
    return sum_temp


def count_y(x):
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = 1/(1+x[i]**2)
    return y


def set_x(size, _min= -5, _max=5):
    step = (_max - _min) / (size - 1)
    x = [_min + (step * i) for i in range(size)]
    return x


def set_x_cz(size, _min = - 5, _max=5):
    x = [(0.5 * ((_min - _max) * np.cos(np.pi * (2 * i + 1) / (2 * (size - 1) + 2))) + (_min + _max)) for i in range(size)]
    return x


n = 5
x_min = -5
x_max = 5
#x_ = set_x(n + 1)
x_= set_x_cz(n+1)
#print(x_)
out_arr=[]
y_ = count_y(x_)
a = np.zeros(shape=(n+1, n+1))
for i in range(n+1):
    a[i][0] = y_[i]
print(y_)
#print(a)
for j in range(1,n+1):
    for i in range(j,n+1):
        a[i][j] = (a[i][j-1] - a[i-1][j-1])/(x_[i]-x_[i-j])
print('\n')
print(a)
for i in np.arange(x_min, x_max + 0.01, 0.1):
    out_arr.append(count_wzo_New(a, x_, i, n + 1))
print('\n')
plt.figure()

plt.scatter(x_, y_, marker='o')
plt.plot(np.arange(x_min, x_max + 0.01, 0.1),  out_arr, label='W(x)')
x = np.arange(-5, 5, 0.01)
y = np.zeros(len(x))
for i in range(len(x)):
    y[i] = 1 / (x[i]**2 + 1)
plt.plot(x, y, label='f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()
