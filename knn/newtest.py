
from numpy import array
from sklearn.neighbors import BallTree
import random
from numpy import median

train_x = array([[2, 0], [4, 1], [6, 0], [1, 4], [2, 4], [2, 5], [4, 4],
                [0, 2], [3, 2], [4, 2], [5, 2], [7, 3], [5, 5]])
train_y = array([+1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1])
test_x = array([[1, 5], [0, 3], [6, 1], [6, 4]])
test_y = array([[1,-1,1,-1]])

k = 3

kdtree = BallTree(train_x)
dist, ind = kdtree.query(test_x[0], k)
label = train_y[ind]

print dist
print ind
print label
print len(label)
print median(label)

print len(train_y)
randomNumber = random.randint(0, len(train_y))
# covert randomNumber to list with dimension of self._k
randomNumberList = list( randomNumber \
                          for x in xrange(k))
print randomNumber

print randomNumberList

print len(randomNumberList)
