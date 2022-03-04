import matplotlib.pyplot as plt
import random
import numpy as np


def generator(x, a,c, m):
    tab = []
    x_t = x
    for i in range(n):
      x_t =(a*x_t+c)%m
      tab.append(x_t)
    return tab



x0 = 10
n = 10**4
m1 = 2**15
m2 = 2**32
a1 = 123
a2 = 69069
c = 1
t = generator(x0, a1, c, m1)
t1 = generator(x0, a2, c, m2)

plt.scatter([i/(m1+1) for i in t[1:]], [j/(m1+1) for j in t[:n-1]], s=0.7)
plt.title('rozklad xi od xi-1 dla  m = {}, a = {}'.format(m1, a1))
plt.show()
plt.scatter([i/(m1+1) for i in t1[1:]], [j/(m1+1) for j in t1[:n-1]], s=0.7)
plt.title('rozklad xi od xi-1 dla  m = {}, a = {}'.format(m2, a2))
plt.show()
# plt.style.use('ggplot')
plt.hist(t, bins=12, edgecolor = "blue")
plt.show()
# plt.style.use('ggplot')
plt.hist(t1, bins=12,edgecolor = "blue")
plt.show()

n = 10**3
delt = 3
mi = 4



def generatortrojkat(n, delt, mi):
    tab = []
    for i in range(n):
        e1 = random.random()
        e2 = random.random()
        tab.append(mi+(e1+e2-1)*delt)
    return tab


t3 = generatortrojkat(n, delt, mi)
#plt.style.use('ggplot')
#plt.hist(t3, bins=10, edgecolor = "blue")
#plt.show()

def F(x, u):
    if x <= u:
        return (-1/pow(delt,2)) * (-(pow(x,2)/2) + u*x) + (x/delt) - ((-1/pow(delt,2)) * (-pow((u-delt),2)/2 + u*(u-delt)) + (u-delt)/delt)
    else:
        return (-1/pow(delt,2)) * (pow(x,2)/2 - u*x) + x/delt - (-1/pow(delt,2) * (pow(u,2)/2 - pow(u,2)) + u/delt) + 1/2


a = mi - delt
b = mi + delt
h = (b - a) / 10
bins = np.zeros(10)
for number in t3:
  for j in range(10):
    if number < a + (j+1) * h:
      bins[j] += 1
      break

arr_val = [F(i+h, mi) - F(i, mi) for i in np.arange(a, b, h)]
plt.bar([i for i in np.arange(a, b, h)], bins/n, color = 'red', edgecolor = 'blue', width=0.5)
plt.plot([i for i in np.arange(a, b, h)], arr_val, color ='blue', label='$p_i$', marker = 'o')
plt.xlabel('X')
plt.ylabel('n_i/N')
plt.show()