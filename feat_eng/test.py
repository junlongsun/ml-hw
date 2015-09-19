from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

a = ["I'm taking here,lol, ;-)  aren't you!",
    "here you going",
    ]

stemmer = SnowballStemmer("english")
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
#tokenizer1 = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
#tokenizer2 = RegexpTokenizer('\s+', gaps=True)
#print tknzr.tokenize(a)
#print tokenizer1.tokenize(a)
#print tokenizer2.tokenize(a)

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

print refineSentence(a)
