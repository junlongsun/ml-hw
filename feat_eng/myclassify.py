
from csv import DictReader, DictWriter

import numpy as np
from numpy import array, zeros, shape

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC

#from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'


class Featurizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(min_df=1)
        self.selector = SelectKBest(chi2, k=5000)

    def train_feature(self, examples, target):
        x_train = self.vectorizer.fit_transform(examples)
        doc_terms_train = x_train.toarray()
        x_train_new = self.selector.fit_transform(doc_terms_train, target)
        return x_train_new

    def test_feature(self, examples):
        doc_terms_test = self.vectorizer.transform(examples)
        x_test_new = self.selector.transform(doc_terms_test)
        return x_test_new

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-10:]
            bottom10 = np.argsort(classifier.coef_[0])[:10]
            print("Pos: %s" % " ".join(feature_names[top10]))
            print("Neg: %s" % " ".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))

if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))

    feat = Featurizer()

    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    y_train = array(list(labels.index(x[kTARGET_FIELD])
                         for x in train))

    x_train = feat.train_feature((x[kTEXT_FIELD] for x in train), y_train)
    x_test = feat.test_feature(x[kTEXT_FIELD] for x in test)

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    #feat.show_top10(lr, labels)
    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "spoiler"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'cat': labels[pp]}
        o.writerow(d)
