
from csv import DictReader, DictWriter
from numpy import array, zeros, shape

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier


kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'

train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
test = list(DictReader(open("../data/spoilers/test.csv", 'r')))

vectorizer = TfidfVectorizer(min_df=1)

labels = []
for line in train:
    if not line[kTARGET_FIELD] in labels:
        labels.append(line[kTARGET_FIELD])
y_train = array(list(labels.index(x[kTARGET_FIELD])
                     for x in train))
x_train = vectorizer.fit_transform(x[kTEXT_FIELD] for x in train)
dataX = x_train.toarray()

doc_terms_train, doc_terms_test, y_train, y_test= train_test_split(dataX, y_train, test_size = 0.2, random_state=42)
'''
selector = SelectKBest(chi2, k=5000)
'''
selector = LinearSVC(C=1, penalty="l2", dual=False)
x_train_new = selector.fit_transform(doc_terms_train, y_train)
x_test_new = selector.transform(doc_terms_test)
print shape(x_test_new)

lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
lr.fit(x_train_new, y_train)

#feat.show_top10(lr, labels)

predictions = lr.predict(x_test_new)
count = 0
for i in range(len(predictions)):
    if predictions[i] == y_test[i] :
        count += 1
print count


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
