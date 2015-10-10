from sklearn import svm
import argparse
import numpy as np
from numpy import zeros, shape

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

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
                new_y.append(3)
            else:
                new_y.append(8)
    return new_x[1:], new_y

def calcScore(prediction, result):
    sum = 0
    for i in range(len(prediction)):
        if prediction[i] == result[i]:
            sum += 1
            #print sum
    #print len(prediction)
    return sum/float(len(prediction))

parser = argparse.ArgumentParser(description='KNN classifier options')
parser.add_argument('--k', type=int, default=3,
                        help="Number of nearest points to use")
parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
args = parser.parse_args()
data = Numbers("./mnist.pkl.gz")

#print len(data.train_x[0])
#print shape(data.train_y[0])
#train_x = data.train_x
train_x, train_y = select3and8(data.train_x, data.train_y)
#test_x, test_y = select3and8(data.test_x, data.test_y)

print train_y[0]
show(train_x[0].reshape(28, 28).T)
#show(train_x[0])

#print len(train_x)
#print len(train_y)
#print len(test_x)
#print len(test_y)

#print type(train_x[0])
#print shape(train_x)
#print train_x[0:2]
#print train_y[0:10]


#clf0 = svm.SVC(C=1.0, kernel="linear")
#clf0.fit(train_x, train_y)
#print "c: 10.0, k: linear"
#print calcScore(clf0.predict(test_x), test_y)

#print clf0.support_
#print clf0.support_vectors_

'''
clf0 = svm.SVC(C=10.0, kernel="linear")
clf0.fit(train_x, train_y)
print "c: 10.0, k: linear"
print calcScore(clf0.predict(test_x), test_y)

clf1 = svm.SVC(C=1.0, kernel="linear")
clf1.fit(train_x, train_y)
print "c: 1.0, k: linear"
print calcScore(clf1.predict(test_x), test_y)

clf2  = svm.SVC(C=0.1, kernel="linear")
clf2.fit(train_x, train_y)
print "c: 0.1, k: linear"
print calcScore(clf2.predict(test_x), test_y)

clf3  = svm.SVC(C=0.01, kernel="linear")
clf3.fit(train_x, train_y)
print "c: 0.01, k: linear"
print calcScore(clf3.predict(test_x), test_y)

clf4  = svm.SVC(C=0.001, kernel="linear")
clf4.fit(train_x, train_y)
print "c: 0.001, k: linear"
print calcScore(clf4.predict(test_x), test_y)

clf5  = svm.SVC(C=0.0001, kernel="linear")
clf5.fit(train_x, train_y)
print "c: 0.0001, k: linear"
print calcScore(clf5.predict(test_x), test_y)

clf6  = svm.SVC(C=10, kernel="rbf")
clf6.fit(train_x, train_y)
print "c: 10.0, k: rbf"
print calcScore(clf6.predict(test_x), test_y)

clf7  = svm.SVC(C=1, kernel="rbf")
clf7.fit(train_x, train_y)
print "c: 1.0, k: rbf"
print calcScore(clf7.predict(test_x), test_y)

clf8  = svm.SVC(C=0.1, kernel="rbf")
clf8.fit(train_x, train_y)
print "c: 0.1, k: rbf"
print calcScore(clf8.predict(test_x), test_y)

clf9  = svm.SVC(C=0.01, kernel="rbf")
clf9.fit(train_x, train_y)
print "c: .01, k: rbf"
print calcScore(clf9.predict(test_x), test_y)

clf10  = svm.SVC(C=0.001, kernel="rbf")
clf10.fit(train_x, train_y)
print "c: 0.001, k: rbf"
print calcScore(clf10.predict(test_x), test_y)

clf11  = svm.SVC(C=0.0001, kernel="rbf")
clf11.fit(train_x, train_y)
print "c: .0001, k: rbf"
print calcScore(clf11.predict(test_x), test_y)
'''
