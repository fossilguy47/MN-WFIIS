import math
import matplotlib.pyplot as plt
from decimal import Decimal


def rozw(a, b, k, m, n):
    a_sum = 0
    b_sum = 0
    result = []
    for i in range(n):
        a_sum +=((-1)**i)*((k*a)**(2*i+2+m))/((k**(m+1))*(2*i+m+2)*math.factorial(2*i+1))
        b_sum += (((-1) ** i) * ((k * b) ** (2 * i + 2+m))) / ((k ** (m + 1)) * (2 * i + m + 2) * math.factorial(2 * i + 1))
        result.append(b_sum - a_sum)
    return result


def simpson(f, a, b, n):
    h = (b-a)/n
    k = 0.0
    x = a + h
    result = []
    result.append((h / 3) * (f(a) + f(b) + k))
    for i in range(1, int(n/2 + 1)):
        k += 4*f(x)
        x += 2*h
        result.append((h / 3) * (f(a) + f(b) + k))
    x = a + 2*h
    for i in range(1, int(n/2)):
        k += 2*f(x)
        x += 2*h
        result.append((h/3)*(f(a)+f(b)+k))
    return result


m = 5
k = 5
n= 30
l = 210
def function(x): return (x**m)*(math.sin(k*x))


CI = []
tempsimp = simpson(function, 0, math.pi, l)
temprozw = rozw(0, math.pi, k, m, n)
x_val2 = [11, 21, 51, 101, 201]
for j in x_val2:
    CI.append(abs(tempsimp[j]-temprozw[-1]))
print(simpson(function, 0, math.pi, l))
print("\n\n")
print(rozw(0, math.pi, k, m, n))
x_val = [i for i in range(n)]
plt.plot(x_val, rozw(0, math.pi, k, m, n), marker='.')
plt.title('|C-I| dla k = {}, m = {}'.format(k, m))
plt.xlabel('n-iosc wezlow')
plt.ylabel('|C-I|')
#plt.legend()
plt.show()

