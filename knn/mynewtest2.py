from collections import Counter, defaultdict

d = defaultdict(lambda: defaultdict(int))

d[0][0] += 1
d[1][1] += 2

print d
