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
