from svm import weight_vector, find_support, find_slack, kINSP, kSEP
from numpy import array, zeros


sep_x = kSEP[:, 0:2]
sep_y = kSEP[:, 2]
insep_x = kINSP[:, 0:2]
insep_y = kINSP[:, 2]

'''
#test_wide_slack
w = array([-.25, .25])
b = -.25
find_slack(insep_x, insep_y, w, b), set([6, 4]))
'''

'''
#test_narrow_slack
w = array([0, 2])
b = -5

assertEqual(find_slack(insep_x, insep_y, w, b),
                 set([3, 5]))
'''

'''
#test_support
w = array([0.2, 0.8])
b = -0.2

print find_support(sep_x, sep_y, w, b)
#,                 set([0, 4, 2]))
'''
