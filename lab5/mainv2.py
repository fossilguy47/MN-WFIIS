import numpy as np
import matplotlib.pyplot as plt


def set_a(matrix_a, L, N):
    delta_x = 2*L/N
    for i in range(N):
        x_i = -L+i*delta_x
        matrix_a[i][i] = (delta_x**-2)+(x_i**2)/2
        if i > 0:
            matrix_a[i-1][i] = -1/(2*delta_x**2)
            matrix_a[i][i-1] = -1 / (2 * delta_x ** 2)


def hauss_QR(A, m):
    A = A.astype(float)
    V = np.zeros((m, m))
    for k in range(0, m):
        el = np.zeros((m-k))
        el[0] = 1
        x = A[k:m, k]
        V[k:m, k] = (np.sign(m)*np.linalg.norm(x)*el+x)
        V[k:m, k] = V[k:m, k]/np.linalg.norm(V[k:m, k])
        pom = V[k:m, [k]].T@A[k:m, k:m]
        A[k:m, k:m] = A[k:m, k:m]-2*V[k:m, [k]]@pom
    I = np.identity(m)
    for k in range(n-1, -1, -1):
        pom = 2*V[k:m, [k]]@V[k:m, [k]].T
        I[k:m, :] = I[k:m, :]-pom@I[k:m, :]
    return -I[:, :m], -A[:m, :]


n = 50
l = 5
A = np.zeros((n, n))
set_a(A, l, n)
print(A)
Q, R = hauss_QR(A, n)
A_test = np.matmul(Q, R)

X = A[:]
print(X)
P = np.eye(len(A))
for i in range(100):
    q, r = hauss_QR(X, n)
    P = P @ q
    X = r @ q
lamb = np.diag(X)[::-1]
for i in range(1, 6):
    plt.plot(P[:, -i], marker='.', markersize=10)
plt.grid()
plt.show()

