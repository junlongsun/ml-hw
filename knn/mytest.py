<<<<<<< HEAD
import unittest

from numpy import array

from knn import *

class TestKnn(unittest.TestCase):
    def setUp(self):
        #Training data input
        self.x = array([[2, 0], [4, 1], [6, 0], [1, 4], [2, 4], [2, 5], [4, 4],
                        [0, 2], [3, 2], [4, 2], [5, 2], [7, 3], [5, 5]])
        #Training data output
        self.y = array([+1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1])
        self.knn = {}
        #The number of nearest points to consider in classification
        for ii in [1]:
            self.knn[ii] = Knearest(self.x, self.y, ii)
        #test data
        self.queries = array([[1, 5], [0, 3], [6, 1], [6, 4]])
        #corresponding test label: [1,-1,1,-1]

    def test1(self):
        print self.knn[1].classify(self.queries[0])
        #self.assertAlmostEqual(self.knn[1].classify(self.queries[0]), 1)
        #self.assertAlmostEqual(self.knn[1].classify(self.queries[1]), -1)
        #self.assertAlmostEqual(self.knn[1].classify(self.queries[2]), 1)
        #self.assertAlmostEqual(self.knn[1].classify(self.queries[3]), -1)

if __name__ == '__main__':
    unittest.main()
=======
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
>>>>>>> master
