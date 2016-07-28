from __future__ import division
import os
import time
import numpy as np
import scipy.sparse
from psparse import pmultiply

n_trials = 10
N, M, P = 2000, 10000, 1000
RHO = 0.1

X = scipy.sparse.rand(N, M, RHO).tocsr()
X2 = scipy.sparse.rand(M, N, RHO).tocsr()
W = np.asfortranarray(np.random.randn(M, P))

assert np.all(pmultiply(X, W) == X.dot(W))
assert np.all(pmultiply(X2.T, W) == X2.T.dot(W))

print X.T.indptr

t0 = time.time()
for i in range(n_trials):
    A = pmultiply(X, W)

t1 = time.time()
for i in range(n_trials):
    B = X.dot(W)

t2 = time.time()

print 'This Code : %.5fs' % ((t1 - t0) / n_trials)
print 'Scipy     : %.5fs' % ((t2 - t1) / n_trials)
