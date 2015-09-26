
from csv import DictReader, DictWriter
from numpy import array, zeros, shape, matrix, sum, max
import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA

import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfTransformer


from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from sklearn.decomposition import PCA

from nltk.tokenize import TweetTokenizer

def split_on_caps(str):

    rs = re.findall('[A-Z][^A-Z]*',str)
    fs = ""
    for word in rs:
        fs += " " + stemmer.stem(word)

    return fs

def refineTropes(groupText, index='trope'):
    sen = []
    #sentences = nltk.sent_tokenize(text)
    for x in groupText:
        temp = split_on_caps(x[index])
        #temp1 =  stemmer.stem(temp)
        #wnl.lemmatize(temp)
        sen.append(temp)
    return sen

def split_on_Sen(text):
    fs = ""
    words = tknzr.tokenize(text)
    for i in range(len(words)):
        temp = stemmer.stem(words[i])
        fs += " " + temp
    return fs

def refineSentence(groupText):
    sen = []
    #sentences = nltk.sent_tokenize(text)
    for x in groupText:
        temp = split_on_Sen(x)
        #temp1 =  stemmer.stem(temp)
        #wnl.lemmatize(temp)
        sen.append(temp)
    return sen

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'
kTrope = 'trope'

stemmer = SnowballStemmer("english")
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)

train = list(DictReader(open("train.csv", 'r')))
test = list(DictReader(open("test.csv", 'r')))

vectorizerNew = CountVectorizer(min_df=1)
#vectorizer = TfidfVectorizer(min_df=1)
#vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
#vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 5), analyzer="char", binary=False)
vectorizer1 = TfidfVectorizer(ngram_range=(1,1), analyzer="word", token_pattern=r'\b\w+\b', min_df=3)
vectorizer2 = TfidfVectorizer(ngram_range=(2,3), analyzer="word", token_pattern=r'\b\w+\b', min_df=3)
#vectorizer = TfidfVectorizer(ngram_range=(1, 1), analyzer="word", binary=False, min_df=1)
#x1 = vectorizer_char.transform(x[kTEXT_FIELD] for x in train)


labels = []
for line in train:
    if not line[kTARGET_FIELD] in labels:
        labels.append(line[kTARGET_FIELD])

y_train = matrix([list(labels.index(x[kTARGET_FIELD])
                     for x in train), range(len(train))])
y_train = y_train.transpose()

length = vectorizerNew.fit_transform(x[kTEXT_FIELD] for x in train )
lengthTemp = length.toarray()

sent1 = refineSentence(x[kTEXT_FIELD] for x in train )

x_train1 = vectorizer1.fit_transform(sent1)
dataX1 = x_train1.toarray()
#print vectorizer1.get_feature_names()

x_train2 = vectorizer2.fit_transform(sent1)
dataX2 = x_train2.toarray()

sen2 = refineTropes(train, kTrope)
vectorizerNew1 = CountVectorizer(ngram_range=(1,1), analyzer="word", token_pattern=r'\b\w+\b', min_df=1)
x_train3 = vectorizerNew1.fit_transform(sen2)
dataX3 = x_train3.toarray()
#print len(train)
#limit = len(train)
limit = 10
y_train = y_train[0:limit,:]
dataX1 = dataX1[0:limit,:]
dataX2 = dataX2[0:limit,:]
dataX3 = dataX3[0:limit,:]
lengthTemp = lengthTemp[0:limit]
#print shape(dataX3)
#dataX = np.c_[ dataX1, dataX2,  dataX3, sum(lengthTemp, axis=1)/max(sum(lengthTemp, axis=1)) ]
dataX = np.c_[ dataX1, dataX3, sum(lengthTemp, axis=1)/max(sum(lengthTemp, axis=1))  ]
#dataX = np.c_[ dataX1, dataX2]
print shape(dataX)

doc_terms_train, doc_terms_test, y_train, y_test= train_test_split(dataX, y_train, test_size = 0.2, random_state=42)#
#doc_terms_train, doc_terms_test, y_train, y_test= train_test_split(dataX, y_train, test_size = 0.2)

print shape(doc_terms_train)
pca = PCA(n_components=15)
selection = SelectKBest(k=24)
combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
#combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
combined_features.fit(doc_terms_train, y_train[:,0].A1)

X_features = combined_features.transform(doc_terms_train)
x_test_new = combined_features.transform(doc_terms_test)

lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)

lr.fit(X_features, y_train[:,0].A1)
predictions = lr.predict(x_test_new)

'''
print shape(X_features)
svm = SVC(kernel="linear")
pipeline = Pipeline([("features", combined_features), ("svm", svm)])
param_grid = dict(features__pca__n_components=[1, 2, 3],
                  features__univ_select__k=[1, 2],
                  svm__C=[0.1, 1, 10])
grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)


grid_search.fit(doc_terms_train, y_train[:,0].A1)
#x_test_new = grid_search.transform(x_test)
predictions = grid_search.predict(x_test)
'''
'''
#selector = SelectKBest(chi2, k=5000)
#selector = LinearSVC(C=1, penalty="l2", dual=False)
selector = SelectPercentile(score_func=f_classif, percentile=8)
#x_train_new = selector.fit_transform(doc_terms_train, y_train[:,0].A1)
#x_test_new = selector.transform(doc_terms_test)
#selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
x_train_new  = doc_terms_train
x_test_new = doc_terms_test
#print shape(x_test_new)


#feat.show_top10(lr, labels)


'''

count = 0
failedLabel = []
for i in range(len(predictions)):
    if predictions[i] == y_test[i,0] :
        count += 1
    else:
        failedLabel.append([y_test[i,1]])
print "total number is", len(predictions)
print "success", count

#print "failed label"
#print failedLabel
#print "all test label"
#print y_test[:,1].A1


'''

http://scikit-learn.org/stable/auto_examples/feature_stacker.html
http://stackoverflow.com/questions/15257674/scikit-learn-add-features-to-a-vectorized-set-of-documents
http://blog.christianperone.com/2011/10/machine-learning-text-feature-extraction-tf-idf-part-ii/
see original chinese post
'''
#selector = LinearSVC(C=1, penalty="l2", dual=False)

'''
select = SelectPercentile(score_func=chi2, percentile=16)
clf = LogisticRegression(tol=1e-8, penalty='l2', C=4)
countvect_char = TfidfVectorizer(ngram_range=(1, 5), analyzer="char", binary=False)
countvect_word = TfidfVectorizer(ngram_range=(1, 3), analyzer="word", binary=False, min_df=3)
badwords = BadWordCounter()
ft = FeatureStacker([("badwords", badwords), ("chars", countvect_char),("words", countvect_word)])
char_word_model = Pipeline([('vect', ft), ('select', select), ('logr', clf)])

clf = char_word_model
#clf.fit(comments, labels)
#X_train, y_train = comments[train], labels[train]
#X_test, y_test = comments[test], labels[test]
clf.fit(doc_terms_train, y_train)
probs = clf.predict_proba(doc_terms_test)
print("score: %f" % auc_score(y_test, probs[:, 1]))
#probs_common += probs
'''


#x_test = vectorizer.transform(x[kTEXT_FIELD] for x in test)
#test_x = x_test.toarray()
#print shape(test_x)
#x_test_new = ch2.transform(test_x)
#print shape(x_test_new)

#n = len(x_train)
#print n
#score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, n), total_word_count)


#word_fd = FreqDist()
#label_word_fd = ConditionalFreqDist()
#for word in movie_reviews.words(categories=['pos']):
#    word_fd.inc(word.lower())
#    label_word_fd['pos'].inc(word.lower())
