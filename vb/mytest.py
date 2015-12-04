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
'''
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
'''
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
'''
#------------------------------------------------------------#
#test_m

vb = VariationalBayes()
vb.init([], "stuck", 3)
print vb._gamma
topic_count = array([[5., 4., 3., 2., 1.],
                     [0., 2., 2., 4., 1.],
                     [1., 1., 1., 1., 1.]])

new_beta = vb.m_step(topic_count)
print new_beta
#------------------------------------------------------------#
