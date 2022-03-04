import numpy as np
import matplotlib.pyplot as plt

def hor_sch(poly, x):
    n = len(x)
    result = [0]*n
    for j in range(0, n):
        for i in range(n-1, -1, -1):
            result[j] = result[j] * x[j] + poly[i]
    return result


def matrix_mul(m_1, m_2):
    m_end = [[0 for col in range(len(m_1))] for rows in range(len(m_2[0]))]
    for i in range(len(m_1)):
        for k in range(len(m_2[0])):
            temp_sum = 0
            for j in range(len(m_1[0])):
                temp_sum = temp_sum + m_1[i][j]*m_2[j][k]
            m_end[i][k] = temp_sum
    return m_end


def matrix_table_mul(m_1, m_2):
    m_end = [0] * len(m_1)
    for i in range(len(m_1)):
        for j in range(len(m_1[0])):
            m_end[i] += m_1[i][j] * m_2[j]
    return m_end


def print_matrix2d(tab):
    for row in tab:
        for elem in row:
            print(elem, end=' ')
        print()


def set_a(tab):
    tab_end = [[0 for col in range(6)] for rows in range(6)]
    for i in range(6):
        for j in range(6):
            tab_end[i][j] = pow(tab[i], j)
    return tab_end


def gauss_lu(A, L):
    N = len(A)
    U = np.zeros((N, N))
    for i in range(N):
        L[i][i] = 1
    for j in range(N):
        for i in range(j + 1):
            U[i][j] = A[i][j] - sum(U[k][j] * L[i][k] for k in range(i))
        for i in range(j, N):
            L[i][j] = (A[i][j] - sum(U[k][j] * L[i][k] for k in range(j))) / U[j][j]
    return U


matrix_start_x = [-2, -1, 1, 2, 3, 4]
size = len(matrix_start_x)
matrix_set_x = set_a(matrix_start_x)
matrix_start_c = [1, 3, 3, 5, 4, 2]
matrix_l = np.zeros((size, size))
matrix_u = gauss_lu(matrix_set_x, matrix_l)
matrix_y = matrix_table_mul(matrix_u, matrix_start_c)
matrix_w_lu = matrix_table_mul(matrix_l, matrix_y)
print(matrix_w_lu)
matrix_w_horn = hor_sch(matrix_start_c, matrix_start_x)
print("\n \n \n ")
print(matrix_w_horn)

x = np.linspace(-2,2)
y = 1 + x*3 + 3*x**2 + 5*x**3 + 4*x**4+2*x**5
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.plot(x, y, label="w(x)")
plt.scatter(matrix_start_x, matrix_w_lu, color='red', label='y_from_LU')
plt.legend(loc='upper left')
plt.show()
