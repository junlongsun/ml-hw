import numpy as np
from sklearn.neighbors import BallTree

#np.random.seed(0)
#X = np.random.random((10, 3))  # 10 points in 3 dimensions

X = np.array([[2, 0], [4, 1], [6, 0], [1, 4], [2, 4], [2, 5], [4, 4],
                [0, 2], [3, 2], [4, 2], [5, 2], [7, 3], [5, 5]])
print X
tree = BallTree(X, leaf_size=2)

dist, ind = tree.query(X[0], k=3)
print ind  # indices of 3 closest neighbors
print dist  # distances to 3 closest neighbors
