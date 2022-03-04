import numpy as np
import matplotlib.pyplot as plt


def get_y_1(i):
    return 1/(1+i**2)


def get_second_derivative(fun, arr):
    dx = 0.01
    return [(fun(x - dx) - 2 * fun(x) + fun(x + dx)) / pow(dx, 2) for x in arr]


def get_s(y, x, m, X, n):
    i = 0
    for j in range(n - 1):
        if x[j] <= X <= x[j + 1]:
            i = j + 1
    return m[i - 1] * pow((x[i] - X), 3) / (6 * (x[i]-x[i-1])) + m[i] * pow((X - x[i - 1]), 3) / (6 * (x[i]-x[i-1])) + ((y[i] - y[i-1])/(x[i]-x[i-1]) - (x[i]-x[i-1])/6*(m[i]-m[i-1])) * (X - x[i - 1]) + (y[i-1] - m[i-1]*(x[i]-x[i-1])**2 / 6)


def A_matrix_set(x_, n):
    a = np.zeros(shape=(n, n))
    h=[]
    for i in range(0, n):
        h.append(x_[i]-x_[i-1])
    for i in range(n):
        a[i][i] = 2
    a[0][0] = 1
    a[n-1][n-1] = 1
    for i in range(1, n-1):
        a[i][i+1] = h[i]/(h[i]+h[i-1])
        a[i][i-1] = 1 - a[i][i+1]
    a[1][0] = 0.5
    a[1][2] = 0.5
    return a, h


def set_x(size, _min=-5, _max=5):
    step = (_max - _min) / (size - 1)
    x = [_min + (step * i) for i in range(size)]
    return x


def set_y_1(x):
    y = [1/(1+x[i]**2) for i in range(len(x))]
    return y


def set_y_2(x):
    y = [np.cos(2*x[i]) for i in range(len(x))]
    return y


def set_d(size, h, y):
    d = []
    d.append(0)
    for i in range(1, size-1):
        d.append((6/(h[i]+h[i+1]))*(((y[i+1]-y[i])/h[i+1])-((y[i]-y[i-1])/h[i])))
    d.append(0)
    return d


def set_m(A, d):
    return np.linalg.solve(A, d)


n = 5
x_min = -5
x_max = 5
x = set_x(n, x_min, x_max)
A, h = A_matrix_set(x, n)
y = set_y_1(x)
d = set_d(n, h, y)
m = set_m(A, d)
s = [get_s(y, x, m, i, n) for i in x]
t_arr = np.arange(x_min, x_max, 0.01)
# derivative = get_second_derivative(get_y_1, t_arr)
plt.plot(t_arr, set_y_1(t_arr), label='f(x)')
plt.scatter(x, s)
plt.plot(t_arr, [get_s(y, x, m, i, n) for i in t_arr], label='interpolacja')
# plt.scatter(x, m, label='wektor m')
# plt.plot(t_arr, derivative,color='green', label='pochodne analityczne')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
