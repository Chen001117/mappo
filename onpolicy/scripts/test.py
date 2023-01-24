import numpy as np

a = np.ones([2,3,3])
a[0] += 1
b = np.ones([2,3,1])
b[0] += 2
print((a@b))