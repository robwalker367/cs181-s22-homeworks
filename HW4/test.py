import numpy as np

class A(object):
  def assignment(self, nclusters=10):
    return str(nclusters)

class B(object):
  def assignment(self):
    return 'B'
  
methods = [A(), B()]
for method in methods:
  print(method.assignment())

K = 10
a1 = np.arange(10)
a2 = np.arange(10)[::-1]

N = a1.shape[0]
C = np.zeros((K, K))
for i in range(N):
  C[a1[i], a2[i]] += 1
print(C)