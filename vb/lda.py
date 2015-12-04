"""
VariationalBayes for Vanilla LDA
@author: Ke Zhai (zhaike@cs.umd.edu)
"""
from collections import defaultdict
import time

from numpy import log, exp, ones
import numpy

import scipy
import scipy.misc
from scipy.special import psi as digam


def parse_vocabulary(vocab):
    """
    Create a dictionary mapping integers to terms
    """

    type_to_index = {}
    index_to_type = {}
    for word in set(x.strip() for x in vocab):
        index_to_type[len(index_to_type)] = word
        type_to_index[word] = len(type_to_index)

    return type_to_index, index_to_type


def parse_data(corpus, vocab):
    """
    Read a dataset from a file (one document per line) and create it as two
    matrices of terms and counts.
    """

    doc_count = 0
    token_count = 0

    word_ids = []
    word_cts = []

    for document_line in corpus:
        #words = document_line.split()
        document_word_dict = defaultdict(int)
        for token in document_line.split():
            if token in vocab:
                document_word_dict[vocab[token]] += 1
                token_count += 1
            else:
                continue

        word_ids.append(numpy.array(document_word_dict.keys()))
        word_cts.append(numpy.array(document_word_dict.values()))

        doc_count += 1
        if doc_count % 10000 == 0:
            print "successfully import %d documents..." % doc_count

    print("Successfully imported %i documents with %i tokens." %
          (doc_count, token_count))

    return word_ids, word_cts


class VariationalBayes:
    """
    Class for learning a topic model using variational inference
    """

    def init(self, corpus, vocab, num_topics=5, alpha=0.1):
        """
        @param num_topics: the number of topics
        @param data: a defaultdict(dict) data type, first indexed by doc id,
        then indexed by term id take note: tokens are not types, they are
        repeatable and thus might be not unique
        """

        self._alpha = alpha
        self._type_to_index, self._index_to_type = \
            parse_vocabulary(vocab)
        self._num_topics = num_topics
        self._corpus = parse_data(corpus, self._type_to_index)
        self._num_types = len(self._type_to_index)

        # define the total number of document
        self._num_docs = len(self._corpus[0])

        # initialize a D-by-K matrix gamma, valued at N_d/K
        self._gamma = numpy.ones((self._num_docs, self._num_topics))

        # initialize a V-by-K matrix beta, valued at 1/V, subject to the sum
        # over every row is 1
        self._beta = numpy.random.gamma(100., 1. / 100.,
                                       (self._num_topics, self._num_types))
        #self._E_log_eta = compute_dirichlet_expectation(self._beta)
        self._iteration = 0

    @staticmethod
    def new_phi(gamma, beta, word, count):
        """
        Given gamma vector and complete beta, compute the phi for a word with a
        given count
        """
        #phi = numpy.zeros(numpy.size(gamma),numpy.size(word))
        phi = numpy.zeros(len(gamma))
        sumGamma = numpy.sum(gamma)
        #print beta.shape

        for i in range(len(gamma)):
            phi[i] = beta[i,word] * exp(digam(gamma[i]) - digam(sumGamma))
        sumPhi =  numpy.sum(phi)
        return count * phi/float(sumPhi)

    def e_step(self, local_parameter_iteration=50):
        """
        Run the e step of variational EM.  Compute new phi and gamma for all
        words and documents.
        """

        word_ids = self._corpus[0]
        word_cts = self._corpus[1]

        assert len(word_ids) == len(word_cts), "IDs and counts must match"

        number_of_documents = len(word_ids)

        # initialize a V-by-K matrix phi sufficient statistics
        topic_counts = numpy.zeros((self._num_topics, self._num_types))

        # initialize a D-by-K matrix gamma values
        gamma = numpy.ones((number_of_documents, self._num_topics)) * \
            (self._alpha + float(self._num_types) / float(self._num_topics))

        # iterate over all documents
        #for doc_id in xrange(number_of_documents):
        for doc_id in numpy.random.permutation(number_of_documents):
            # compute the total number of words
            term_ids = word_ids[doc_id]
            term_counts = word_cts[doc_id]
            total_word_count = numpy.sum(term_counts)

            # initialize gamma for this document
            gamma[doc_id, :].fill(self._alpha + float(total_word_count)
                                  / float(self._num_topics))

            # update phi and gamma until gamma converges
            for gamma_iteration in xrange(local_parameter_iteration):
                gamma_update = ones(self._num_topics)
                gamma_update.fill(self._alpha)

                for ww, cc in zip(term_ids, term_counts):
                    contrib = VariationalBayes.new_phi(gamma[doc_id, :],
                                                       self._beta, ww, cc)
                    gamma_update += contrib

                    # Save the last topic counts
                    if gamma_iteration == local_parameter_iteration - 1:
                        topic_counts[:, ww] += contrib

                gamma[doc_id, :] = gamma_update

            if (doc_id + 1) % 1000 == 0:
                print "Global iteration %i, doc %i" % \
                    (self._iteration, doc_id + 1)

        self._gamma = gamma
        return topic_counts

    def m_step(self, topic_counts):
        """
        Run the m step of variational inference, setting the beta parameter from
        the expected counts from the e step in the form of a matrix where each
        topic is a row.
        """
        phi = topic_counts
        new_beta = numpy.zeros((self._num_topics,self._num_types))
        for i in range(self._num_topics):
            for j in range(self._num_types):
                    new_beta[i,j] = phi[i,j]
        for i in range(self._num_topics):
            new_beta[i,:] = new_beta[i,:]/ sum(new_beta[i,:])
        return new_beta

    def update_alpha(self, current_alpha=None, gamma=None):
        """
        Update the scalar parameter alpha based on a gamma matrix.  If
        no gamma argument is supplied, use the current setting of
        gamma.
        """

        if current_alpha is None:
            current_alpha = self._alpha
        if gamma is None:
            gamma = self._gamma

        # Update below line
        #new_alpha = current_alpha

        alpha_sufficient_statistics = scipy.special.psi(gamma) - scipy.special.psi(numpy.sum(gamma, axis=1)[:, numpy.newaxis]);
        alpha_sufficient_statistics = numpy.sum(alpha_sufficient_statistics, axis=0);  # [numpy.newaxis, :];

        hyper_parameter_iteration=100
        hyper_parameter_decay_factor=0.9
        hyper_parameter_maximum_decay=10
        hyper_parameter_converge_threshold=1e-6

        alpha_update = current_alpha

        decay = 0;
        for alpha_iteration in xrange(hyper_parameter_iteration):
            alpha_sum = numpy.sum(current_alpha)
            alpha_gradient = self._num_docs * (digam(alpha_sum) - digam(current_alpha) + alpha_sufficient_statistics;
            alpha_hessian = -self._num_docs * scipy.special.polygamma(1, current_alpha);

            if numpy.any(numpy.isinf(alpha_gradient)) or numpy.any(numpy.isnan(alpha_gradient)):
                print "illegal alpha gradient vector", alpha_gradient

            sum_g_h = numpy.sum(alpha_gradient / alpha_hessian);
            sum_1_h = 1.0 / alpha_hessian;

            z = self._num_docs * scipy.special.polygamma(1, alpha_sum);
            c = sum_g_h / (1.0 / z + sum_1_h);

            # update the alpha vector
            while True:
                singular_hessian = False

                step_size = numpy.power(hyper_parameter_decay_factor, decay) * (alpha_gradient - c) / alpha_hessian;
                # print "step size is", step_size
                assert(current_alpha.shape == step_size.shape);

                if numpy.any(current_alpha <= step_size):
                    singular_hessian = True
                else:
                    alpha_update = current_alpha - step_size;

                if singular_hessian:
                    decay += 1;
                    if decay > hyper_parameter_maximum_decay:
                        break;
                else:
                    break;

            # compute the alpha sum
            # check the alpha converge criteria
            mean_change = numpy.mean(abs(alpha_update - current_alpha));
            new_alpha = alpha_update;
            if mean_change <= hyper_parameter_converge_threshold:
                break;

        return new_alpha

    def run_iteration(self, local_iter):
        """
        Run a complete iteration of an e step and an m step of variational
        inference.
        """

        self._iteration += 1

        clock_e_step = time.time()
        topic_counts = self.e_step(local_iter)
        clock_e_step = time.time() - clock_e_step

        clock_m_step = time.time()
        self._beta = self.m_step(topic_counts)
        self._alpha = self.update_alpha()

        clock_m_step = time.time() - clock_m_step

        print "Iteration %i\te_step %d sec, mstep %d sec" % \
            (self._iteration, clock_e_step, clock_m_step)

    def export_beta(self, exp_beta_path, top_display=10):
        """
        Write the most probable words in a topic to a text file.
        """

        output = open(exp_beta_path, 'w')

        for topic_index in xrange(self._num_topics):
            output.write("==========\t%d\t==========\n" % (topic_index))

            beta_probability = self._beta[topic_index, ]

            ii = 0
            for type_index in reversed(numpy.argsort(beta_probability)):
                ii += 1
                output.write("%s\t%g\n" % (self._index_to_type[type_index],
                                           beta_probability[type_index]))
                if top_display > 0 and ii >= top_display:
                    break

        output.close()

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--documents", help="Raw documents",
                           type=str, default="ap/train.dat", required=False)
    argparser.add_argument("--num_topics", help="Number of topics",
                           type=int, default=10, required=False)
    argparser.add_argument("--vocab", help="Vocabulary",
                           type=str, default="ap/voc.dat", required=False)
    argparser.add_argument("--alpha", help="Alpha hyperparameter",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--iterations", help="Number of outer iterations",
                           type=int, default=10, required=False)
    argparser.add_argument("--inner_iter", help="Number of inner iterations",
                           type=int, default=5, required=False)
    argparser.add_argument("--topics_out", help="Where we write topics",
                           type=str, default="topics.txt", required=False)

    flags = argparser.parse_args()

    vb = VariationalBayes()
    vb.init(open(flags.documents), open(flags.vocab),
            flags.num_topics, flags.alpha)

    for ii in xrange(flags.iterations):
        vb.run_iteration(flags.inner_iter)

    vb.export_beta(flags.topics_out)
