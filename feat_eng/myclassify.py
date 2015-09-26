
from csv import DictReader, DictWriter

import numpy as np
import re
import nltk

from numpy import array, zeros, shape, matrix, sum, max

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from nltk.tokenize import TweetTokenizer
from sklearn.decomposition import PCA
from nltk.stem.snowball import SnowballStemmer
from sklearn.pipeline import FeatureUnion

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'
kTrope = 'trope'

class Featurizer:
    def __init__(self):
        self.vectorizer1 = TfidfVectorizer(ngram_range=(1,1), analyzer="word", token_pattern=r'\b\w+\b', min_df=3)
        self.vectorizer2 = TfidfVectorizer(ngram_range=(2,3), analyzer="word", token_pattern=r'\b\w+\b', min_df=3)
        self.vectorizer3 = CountVectorizer(min_df=1)
        self.vectorizer4 = CountVectorizer(ngram_range=(1,1), analyzer="word", token_pattern=r'\b\w+\b', min_df=1)
        self.stemmer = SnowballStemmer("english")
        self.tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)

    def split_on_caps(self, str):
        rs = re.findall('[A-Z][^A-Z]*',str)
        fs = ""
        for word in rs:
            fs += " " + self.stemmer.stem(word)
        return fs

    def refineTropes(self, groupText):
        sen = []
        for x in groupText:
            temp = self.split_on_caps(x)
            sen.append(temp)
        return sen

    def split_on_Sen(self, text):
        fs = ""
        words = self.tknzr.tokenize(text)
        for i in range(len(words)):
            temp = self.stemmer.stem(words[i])
            fs += " " + temp
        return fs

    def refineSentence(self, groupText):
        sen = []
        for x in groupText:
            temp = self.split_on_Sen(x)
            #print temp
            sen.append(temp)
        return sen

    def train_feature(self, examples, target, trope):
        length = self.vectorizer3.fit_transform(examples)
        lengthTemp = length.toarray()
        self.maxlength = max(sum(lengthTemp, axis=1))

        #sent1 = self.refineSentence(examples)

        #print sent1
        x_train1 = self.vectorizer1.fit_transform(examples)
        dataX1 = x_train1.toarray()

        #x_train2 = self.vectorizer2.fit_transform(sent1)
        #dataX2 = x_train2.toarray()

        #sen2 = self.refineTropes(trope)
        x_train3 = self.vectorizer4.fit_transform(trope)
        dataX3 = x_train3.toarray()

        dataX = np.c_[dataX1, dataX3, sum(lengthTemp, axis=1)/self.maxlength ]

        pca = PCA(n_components=2000)
        selection = SelectKBest(k=4800)

        self.combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
        self.combined_features.fit(dataX, target)

        X_features = self.combined_features.transform(dataX)

        #x_train = self.vectorizer.fit_transform(examples)
        #doc_terms_train = x_train.toarray()
        #x_train_new = self.selector.fit_transform(doc_terms_train, target)
        return X_features

    def test_feature(self, examples, trope):
        length = self.vectorizer3.transform(examples)
        lengthTemp = length.toarray()

        #sent1 = self.refineSentence(examples)
        x_train1 = self.vectorizer1.transform(examples)
        dataX1 = x_train1.toarray()

        #x_train2 = self.vectorizer2.fit_transform(sent1)
        #dataX2 = x_train2.toarray()

        #sen2 = self.refineTropes(trope)
        x_train3 = self.vectorizer4.transform(trope)
        dataX3 = x_train3.toarray()

        dataXtest = np.c_[dataX1, dataX3, sum(lengthTemp, axis=1)/self.maxlength ]

        x_test_new = self.combined_features.transform(dataXtest)
        #doc_terms_test = self.vectorizer.transform(examples)
        #x_test_new = self.selector.transform(doc_terms_test)
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
    train = list(DictReader(open("train.csv", 'r')))
    test = list(DictReader(open("test.csv", 'r')))

    feat = Featurizer()

    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    y_train = array(list(labels.index(x[kTARGET_FIELD])
                         for x in train))


    sent1 = feat.refineSentence(x[kTEXT_FIELD] for x in train)
    trope1 = feat.refineTropes(x[kTrope] for x in train)
    sent2 = feat.refineSentence(x[kTEXT_FIELD] for x in test)
    trope2 = feat.refineTropes(x[kTrope] for x in test)

    x_train = feat.train_feature(sent1, y_train, trope1)
    x_test = feat.test_feature(sent2, trope2)

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    #feat.show_top10(lr, labels)
    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "spoiler"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'spoiler': labels[pp]}
        o.writerow(d)
