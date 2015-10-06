from numpy import array, zeros
import numpy as np

kINSP = array([(1, 8, +1),
               (7, 2, -1),
               (6, -1, -1),
               (-5, 0, +1),
               (-5, 1, -1),
               (-5, 2, +1),
               (6, 3, +1),
               (6, 1, -1),
               (5, 2, -1)])

kSEP = array([(-2, 2, +1),    # 0 - A
              (0, 4, +1),     # 1 - B
              (2, 1, +1),     # 2 - C
              (-2, -3, -1),   # 3 - D
              (0, -1, -1),    # 4 - E
              (2, -3, -1),    # 5 - F
              ])
sep_x = kSEP[:, 0:2]
sep_y = kSEP[:, 2]
insep_x = kINSP[:, 0:2]
insep_y = kINSP[:, 2]

alpha = zeros(len(sep_x))
alpha[4] = 0.34
alpha[0] = 0.12
alpha[2] = 0.22

x = insep_x
y = insep_y
w = array([-.25, .25])
b = -.25

support = set()
for i in range(len(y)):
    print y[i] * (sum(x[i,:]*w)+b)
    if abs(y[i] * (sum(x[i,:]*w)+b) - 1) < 1e-10:
        support.add(i)

print support
