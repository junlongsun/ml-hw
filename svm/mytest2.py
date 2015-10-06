import numpy as np

a = np.array([[0,0]])
a = np.append(a, np.array([[1,2]]), axis=0)
a = np.append(a, np.array([[3,4]]), axis=0)
print type(a)
print a
print a[1]
