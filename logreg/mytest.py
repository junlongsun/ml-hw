import random
from numpy import zeros, sign, mat, shape, ones, array, argmax, argmin, log
from math import exp, log
from collections import defaultdict

import argparse


kSEED = 1701
kBIAS = "BIAS_CONSTANT"
random.seed(kSEED)

class Example:
    """
    Class to represent a logistic regression example
    """
    def __init__(self, label, words, vocab, df):
        """
        Create a new example

        :param label: The label (0 / 1) of the example
        :param words: The words in a list of "word:count" format
        :param vocab: The vocabulary to use as features (list)
        """
        self.nonzero = {}
        self.y = label
        self.x = zeros(len(vocab))
        for word, count in [x.split(":") for x in words]:
            if word in vocab:
                assert word != kBIAS, "Bias can't actually appear in document"
                self.x[vocab.index(word)] += float(count)
                self.nonzero[vocab.index(word)] = word
        self.x[0] = 1
        D = len(df)
        sumX = sum(self.x)
        self.tfidf = zeros(len(vocab))
        for i in range(D):
            idf = log(D/df[i])
            tf = self.x[i] / sumX
            self.tfidf[i] = tf * idf


def read_dataset(positive, negative, vocab, test_proportion=.1):
    """
    Reads in a text dataset with a given vocabulary

    :param positive: Positive examples
    :param negative: Negative examples
    :param vocab: A list of vocabulary words
    :param test_proprotion: How much of the data should be reserved for test
    """
    df = [float(x.split("\t")[1]) for x in open(vocab, 'r') if '\t' in x]
    vocab = [x.split("\t")[0] for x in open(vocab, 'r') if '\t' in x]
    assert vocab[0] == kBIAS, \
        "First vocab word must be bias term (was %s)" % vocab[0]

    train = []
    test = []
    for label, input in [(1, positive), (0, negative)]:
        for line in open(input):
            ex = Example(label, line.split(), vocab, df)
            if random.random() <= test_proportion:
                test.append(ex)
            else:
                train.append(ex)

    # Shuffle the data so that we don't have order effects
    random.shuffle(train)
    random.shuffle(test)

    return train, test, vocab, df

argparser = argparse.ArgumentParser()
argparser.add_argument("--mu", help="Weight of L2 regression",
                       type=float, default=0.0, required=False)
argparser.add_argument("--step", help="Initial SG step size",
                       type=float, default=0.1, required=False)
argparser.add_argument("--positive", help="Positive class",
                       type=str, default="../data/hockey_baseball/positive", required=False)
argparser.add_argument("--negative", help="Negative class",
                       type=str, default="../data/hockey_baseball/negative", required=False)
argparser.add_argument("--vocab", help="Vocabulary that can be features",
                       type=str, default="../data/hockey_baseball/vocab", required=False)
argparser.add_argument("--passes", help="Number of passes through train",
                       type=int, default=1, required=False)

args = argparser.parse_args()
train, test, vocab, df = read_dataset(args.positive, args.negative, args.vocab)
print df
print len(df)
