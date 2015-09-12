from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
print X
print y
print X.shape

X_new = LinearSVC(C=0.01, penalty="l1", dual=False).fit_transform(X, y)
print X_new.shape
print X_new
