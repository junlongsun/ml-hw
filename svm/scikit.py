from sklearn import svm
import argparse
import numpy as np
from numpy import zeros, shape

class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        f.close()

def select3and8(x,y):
    new_x = np.array([zeros(len(x[0]))])
    #print len(new_x)
    #print len(x[0])
    new_y = []
    for i in range(len(y)):
        if y[i] == 3 or y[i] == 8:
            new_x = np.append(new_x, np.array([x[i]]), axis=0)
            #new_y = np.append(new_y, np.array([y[i]]), axis=0)
            #new_x.append()
            #print "ok"
            if y[i] == 3:
                new_y.append(1)
            else:
                new_y.append(0)
    return new_x[1:], new_y

parser = argparse.ArgumentParser(description='KNN classifier options')
parser.add_argument('--k', type=int, default=3,
                        help="Number of nearest points to use")
parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
args = parser.parse_args()
data = Numbers("./mnist.pkl.gz")

#print len(data.train_x[0])
#print shape(data.train_y[0])
train_x, train_y = select3and8(data.train_x, data.train_y)
test_x, test_y = select3and8(data.test_x, data.test_y)

#print len(train_x)
#print len(train_y)
#print len(test_x)
#print len(test_y)

#print type(train_x[0])
#print shape(train_x)
#print train_x[0:2]
#print train_y[0:10]
'''
if args.limit > 0:
    print("Data limit: %i" % args.limit)
    knn = Knearest(data.train_x[:args.limit], data.train_y[:args.limit],
                       args.k)
else:
    knn = Knearest(data.train_x, data.train_y, args.k)
    print("Done loading data")
'''

#X = [[0, 0], [1, 1]]
#y = [0, 1]
clf = svm.SVC()
clf.fit(train_x, train_y)
print clf.predict(test_x)
#print clf.support_vectors_
#print clf.support_
#print clf.n_support_
