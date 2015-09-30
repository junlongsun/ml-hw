import unittest

from rademacher import origin_plane_hypotheses, axis_aligned_hypotheses, \
    rademacher_estimate, kSIMPLE_DATA as rad_data, PlaneHypothesis, \
    constant_hypotheses
from rademacher import coin_tosses
from vc_sin import train_sin_classifier

def assign_exists(data, classifiers, pattern):
    """
    Given a dataset and set of classifiers, make sure that the classification
    pattern specified exists somewhere in the classifier set.
    """

    val = False
    assert len(data) == len(pattern), "Length mismatch between %s and %s" % \
        (str(data), str(pattern))
    for hh in classifiers:
        present = all(hh.classify(data[x]) == pattern[x] for
                      x in xrange(len(data)))
        # Uncomment for additional debugging code
        # if present:
        #    print("%s matches %s" % (str(hh), str(pattern)))
        val = val or present
    if not val:
        print("%s not found in:" % str(pattern))
        for hh in classifiers:
            print("\t%s %s" % (str(hh), [hh.classify(x) for x in data]))
    return val

var2d = {}
var2d[1] = [(3, 3)]
var2d[2] = [(3, 3), (3, 4)]
var2d[3] = [(3, 3), (3, 4), (4, 3)]
var2d[4] = rad_data

hypotheses = lambda x: [PlaneHypothesis(0, 0, 5),
                              PlaneHypothesis(0, 0, -5),
                              PlaneHypothesis(0, 1, 0),
                              PlaneHypothesis(0, -1, 0),
                              PlaneHypothesis(1, 0, 0),
                              PlaneHypothesis(-1, 0, 0)]
full_shatter = [(1, 1), (-1, -1)]
half_shatter = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

labels = [+1, +1, -1, +1]

data = [(1, False), (2, True), (3, False)]

classifier_pos = train_sin_classifier(data)

print len(data)

for xx, yy in data:
    #print xx
    print yy
    print classifier_pos.classify(xx)
'''
            self.assertEqual(True if yy == +1 else False,
                             classifier_pos.classify(xx))
'''
