import numpy as np
from sklearn.svm import SVC


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

X = np.array([[-1, -1], [-2, -1], [2, 1], [1, 2], [0,2]])
y = np.array([1, 1,  2, 2, 2])


train_x  = [[-0.8, -1]]
clf = SVC(C=1.0, kernel="linear")
clf.fit(X, y)
print(clf.predict(train_x))


sv = clf.support_vectors_
print sv
print clf.support_
#show(sv)
