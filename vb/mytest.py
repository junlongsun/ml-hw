import unittest
import numpy
from numpy import array

from lda import VariationalBayes
from scipy.special import psi as digam
from math import exp

init_beta = array([[.26, .185, .185, .185, .185],
                        [.185, .185, .26, .185, .185],
                        [.185, .185, .185, .26, .185]])
#------------------------------------------------------------#
#test_single_phi
vb = VariationalBayes()
gamma = array([2.0, 2.0, 2.0])
beta = init_beta
#-----------------#
phi = vb.new_phi(gamma, beta, 0, 1)
#-----------------#
prop = 0.27711205238850234
normalizer = sum(x * prop for x in beta[:, 0])

print phi[0], beta[0][0] * prop / normalizer
print phi[1], beta[1][0] * prop / normalizer
print phi[2], beta[2][0] * prop / normalizer

sumGamma = numpy.sum(gamma)
print sumGamma, digam(sumGamma)
print exp(digam(gamma[0]) - digam(sumGamma))
print beta[0][0] * prop / normalizer
print (beta[0][0] * prop / normalizer) / (exp(digam(gamma[0]) - digam(sumGamma)))
#------------------------------------------------------------#
'''
#test_multiple_phi
vb = VariationalBayes()

gamma = array([2.0, 2.0, 2.0])
beta = init_beta
phi = vb.new_phi(gamma, beta, 0, 2)

prop = 0.27711205238850234
normalizer = sum(x * prop for x in beta[:, 0]) / 2.0
print phi[0], beta[0][0] * prop / normalizer
print phi[1], beta[1][0] * prop / normalizer
print phi[2], beta[2][0] * prop / normalizer
#------------------------------------------------------------#
#test_m
vb = VariationalBayes()
vb.init([], "stuck", 3)

topic_count = array([[5., 4., 3., 2., 1.],
                     [0., 2., 2., 4., 1.],
                     [1., 1., 1., 1., 1.]])

new_beta = vb.m_step(topic_count)
print new_beta[2][3], .2
print new_beta[0][0], .33333333
print new_beta[1][4], .11111111
print new_beta[0][3], .13333333
#------------------------------------------------------------#
'''
