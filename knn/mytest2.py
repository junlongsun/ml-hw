from numpy import array
from sklearn.neighbors import BallTree
from knn import *

x = array([[2, 0], [4, 1], [6, 0], [1, 4], [2, 4], [2, 5], [4, 4],
                [0, 2], [3, 2], [4, 2], [5, 2], [7, 3], [5, 5]])
y = array([+1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1])
knn = {}
#for ii in [1, 2, 3]:
#    knn[ii] = Knearest(x, y, ii)
#    print knn[ii]

queries = array([[1, 5], [0, 3], [6, 1], [6, 4]])

kdtree = BallTree(x)
dist, ind = kdtree.query(x, k=3)
print dist
print ind
