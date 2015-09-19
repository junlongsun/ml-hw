from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

from csv import DictReader, DictWriter

import numpy as np
from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC

from sklearn.datasets import load_iris

iris = load_iris()

X, y = iris.data, iris.target

print np.shape(X)
print np.shape(y)

'''
vectorizer = CountVectorizer(min_df=1)
vectorizerNew = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)

CountVectorizer(analyzer='word', binary=False,
        decode_error='strict',
        dtype='numpy.int64', encoding='utf-8', input='content',
        lowercase=True, max_df=1.0, max_features=None, min_df=1,
        ngram_range=(1, 1), preprocessor=None, stop_words=None,
        strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
        tokenizer=None, vocabulary=None)
corpus = [
     'This is the first document.',
     'This is the second second document.',
     'And the third one.',
     'Is this the first document?',
     'WeCanRuleTogether',
 ]

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'
kTrope = 'trope'

train = list(DictReader(open("train.csv", 'r')))
test = list(DictReader(open("test.csv", 'r')))

wnl = WordNetLemmatizer()
stemmer = SnowballStemmer("english")

def split_on_caps(str):

    rs = re.findall('[A-Z][^A-Z]*',str)
    fs = ""
    for word in rs:
        fs += " " + stemmer.stem(word)

    return fs

def refineSentence(groupText, index):
    sen = []
    #sentences = nltk.sent_tokenize(text)
    for x in groupText:
        temp = split_on_caps(x[index])
        #temp1 =  stemmer.stem(temp)
        #wnl.lemmatize(temp)
        sen.append(temp)
    return sen

#print wnl.lemmatize("runing")
#print wnl.lemmatize("playing")
#print wnl.lemmatize("I'm runing and playing")

sen = refineSentence(train, kTrope)
print sen
#print sen
#print split_on_caps(x[kTrope] for x in train)

#

#print wnl.lemmatize(sen)

vectorizerNew1 = CountVectorizer(ngram_range=(1,1), analyzer="word", token_pattern=r'\b\w+\b', min_df=1)
vectorizerNew1.fit_transform(sen)
#print vectorizerNew1.get_feature_names()
'''

'''
#print X

#analyze = vectorizer.build_analyzer()
#print (analyze("This is a text document to analyze.") == (
#     ['this', 'is', 'text', 'document', 'to', 'analyze']) )

#print (vectorizer.get_feature_names() == (
#     ['and', 'document', 'first', 'is', 'one',
#      'second', 'the', 'third', 'this']))

#print X.toarray()

#print vectorizer.vocabulary_.get('first')
#print vectorizer.transform(['Something completely new.']).toarray()
#print vectorizer.transform(['Something and new.']).toarray()


bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                     token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()
#print (analyze('Bi-grams are cool!') == (
#     ['bi', 'grams', 'are', 'cool', 'bi grams', 'grams are', 'are cool']))
X_2 = bigram_vectorizer.fit_transform(corpus).toarray()
feature_index = bigram_vectorizer.vocabulary_.get('is this')

transformer = TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=False,
                 use_idf=True)
counts = [[3, 0, 1],
           [2, 0, 0],
           [3, 0, 0],
           [4, 0, 0],
           [3, 2, 0],
           [3, 0, 2]]
tfidf = transformer.fit_transform(counts)
#print (tfidf.toarray())
#print (transformer.idf_)


vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus)
analyze = vectorizer.build_analyzer()
#print (analyze("This is a text document to analyze.") == (
#     ['this', 'is', 'text', 'document', 'to', 'analyze']))
#print X.toarray()

'''
