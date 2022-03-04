import numpy as np
import matplotlib.pyplot as p


def norm(vec):
    sum_1 = 0
    for i in range(len(vec)):
        sum_1 = sum_1 + vec[i]**2
    return np.sqrt(sum_1)


def set_a(matrix_a, L, N):
    delta_x = 2*L/N
    for i in range(N):
        x_i = -L+i*delta_x
        matrix_a[i][i] = (delta_x**-2)+(x_i**2)/2
        if i > 0:
            matrix_a[i-1][i] = -1/(2*delta_x**2)
            matrix_a[i][i-1] = -1 / (2 * delta_x ** 2)


n = 50
l = 5
A = np.zeros((n, n))
set_a(A, l, n)
pp = 10000
kp = 0
for i in range(n):
    s = 0
    for j in range(n):
        s += abs(A[i][j])
    s -= abs(A[i][i])
    if pp > A[i][i] - s:
        pp = A[i][i] - s
    if kp < A[i][i] + s:
        kp = A[i][i] + s
z = kp
lamb = [kp / 2]
for i in range(5):
    kp = z
    pp = -z
    while (kp - pp) > 1e-7:
        count = 0
        omega = [1, A[0][0] - lamb[-1]]
        for j in range(2, len(A)):
            omega.append((A[j][j] - lamb[-1]) * omega[j - 1] - (A[j - 1][j - 2] ** 2 * omega[j - 2]))
        for v in range(1, len(omega)):
            if omega[v - 1] * omega[v] < 0:
                count += 1
        if count <= i:
            pp = lamb[-1]
        else:
            kp = lamb[-1]
        lamb[-1] = (pp + kp) / 2
    lamb.append(z / 2)

lamb.pop()
print(lamb)
vector = [[] for _ in range(len(lamb))]
for i in range(len(lamb)):
    vector[i].append(1)
    vector[i].append((lamb[i] - A[0][0]) / A[0][1])
    for j in range(1, len(A) - 2):
        vector[i].append(((lamb[i] - A[j][j]) * vector[i][j] - A[j][j + 1] * vector[i][j - 1]) / A[j + 1][j + 2])
for v in vector:
    norma = norm(v)

    for i in range(len(v)):
        v[i] /= norma
    p.plot(v, marker='o', markersize=3)
p.title('pięc pierwszych wartości własnych')
p.grid()
p.show()
