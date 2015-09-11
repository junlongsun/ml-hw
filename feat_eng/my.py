from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = CountVectorizer(min_df=1)
vectorizer

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
 ]
X = vectorizer.fit_transform(corpus)

print X

analyze = vectorizer.build_analyzer()
print (analyze("This is a text document to analyze.") == (
     ['this', 'is', 'text', 'document', 'to', 'analyze']) )

print (vectorizer.get_feature_names() == (
     ['and', 'document', 'first', 'is', 'one',
      'second', 'the', 'third', 'this']))

print X.toarray()

print vectorizer.vocabulary_.get('first')
print vectorizer.transform(['Something completely new.']).toarray()
print vectorizer.transform(['Something and new.']).toarray()


bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                     token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()
print (analyze('Bi-grams are cool!') == (
     ['bi', 'grams', 'are', 'cool', 'bi grams', 'grams are', 'are cool']))
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
print (tfidf.toarray())
print (transformer.idf_)


vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus)
analyze = vectorizer.build_analyzer()
print (analyze("This is a text document to analyze.") == (
     ['this', 'is', 'text', 'document', 'to', 'analyze']))
print X.toarray()
