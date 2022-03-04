import matplotlib.pyplot as plt
import numpy as np
import math


def solve_gj(basic_matrix, solution_matrix):
    n = 5
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            gj_divider = (basic_matrix[j, i] / basic_matrix[i, i])
            basic_matrix[j] -= basic_matrix[i] * gj_divider
            solution_matrix[j] -= solution_matrix[i] * gj_divider
    solved_matrix = [0] * n
    solved_matrix[n - 1] = matrix_b[n - 1] / basic_matrix[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        sum_temp = 0
        for j in range(i + 1, n):
            sum_temp += basic_matrix[i, j] * solved_matrix[j]
        solved_matrix[i] = (solution_matrix[i] - sum_temp) / basic_matrix[i, i]
    return np.matrix(solved_matrix).T


x = []
y = []
for q in np.arange(0.2, 5, 0.05):
    matrix_a = np.array([[q * pow(10, -4), 1, 6, 9, 10],
                  [pow(10, -4), 1, 6, 9, 10],
                  [1, 6, 6, 8, 6],
                  [5, 9, 10, 7, 10],
                  [3, 4, 9, 7, 9]
                  ], float)
    matrix_b = np.array([10, 2, 9, 9, 3], float)
    solution = solve_gj(matrix_a, matrix_b)
    matrix_c = matrix_a @ solution
    Sum = 0
    for k in range(5):
        Sum += pow((matrix_c[k] - matrix_b[k]), 2)
    deflection = 1 / 5 * math.sqrt(Sum)
    x.append(q)
    y.append(deflection)


plt.plot(x, y, label="o(g)")
plt.xlabel('q')
plt.ylabel("Odchylenie o(q)")
plt.yscale('log')
plt.show()
