import numpy as np
x = np.array([[0, 2], [1, 1], [2, 0]]).T
print np.mean(x[1, ])
print np.var(x[1, ])
print np.cov(x)

x = np.array([0, 1, 4, 5, 6, 8, 8, 9, 11, 10, 19, 19.2, 19.7, 3])
for i in [75, 90, 95, 99]: 
    print i, np.percentile(x, i)
