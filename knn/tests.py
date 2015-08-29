import unittest

from numpy import array

from knn import *

class TestKnn(unittest.TestCase):
    def setUp(self):
        self.x = array([[2, 0], [4, 1], [6, 0], [1, 4], [2, 4], [2, 5], [4, 4],
                        [0, 2], [3, 2], [4, 2], [5, 2], [7, 3], [5, 5]])
        self.y = array([+1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1]) # 1,-1 适用于assertAlmostEqual还是不是？，先不管，想按照聚类算法来玩，最后再看unittest.TestCase是怎么玩的
        self.knn = {}
        for ii in [1, 2, 3]:
            #选择聚类的个数k.
            self.knn[ii] = Knearest(self.x, self.y, ii)
            #print self.knn[ii]

        # example
        self.queries = array([[1, 5], [0, 3], [6, 1], [6, 4]])

    def test1(self):
        self.assertAlmostEqual(self.knn[1].classify(self.queries[0]), 1) # 1,-1 适用于assertAlmostEqual还是不是
        self.assertAlmostEqual(self.knn[1].classify(self.queries[1]), -1)
        self.assertAlmostEqual(self.knn[1].classify(self.queries[2]), 1)
        self.assertAlmostEqual(self.knn[1].classify(self.queries[3]), -1)

    def test2(self):
        self.assertAlmostEqual(self.knn[2].classify(self.queries[0]), 1)
        self.assertAlmostEqual(self.knn[2].classify(self.queries[1]), 0)
        self.assertAlmostEqual(self.knn[2].classify(self.queries[2]), 0)
        self.assertAlmostEqual(self.knn[2].classify(self.queries[3]), -1)

    def test3(self):
        self.assertAlmostEqual(self.knn[3].classify(self.queries[0]), 1)
        self.assertAlmostEqual(self.knn[3].classify(self.queries[1]), 1)
        self.assertAlmostEqual(self.knn[3].classify(self.queries[2]), 1)
        self.assertAlmostEqual(self.knn[3].classify(self.queries[3]), -1)

if __name__ == '__main__':
    unittest.main()
