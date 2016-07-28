from __future__ import division
import os
import time
import numpy as np
import scipy.sparse
from _psparse import pmultiply
import sparse_mk

n_trials = 10
N, M, P = 2000, 10000, 1000
RHO = 0.1

X = scipy.sparse.rand(N, N, RHO).tocsc()
W = np.asfortranarray(np.random.randn(N, P))

assert np.all(pmultiply(X, W) == X.dot(W))

print X.T.indptr

t0 = time.time()
for i in range(n_trials):
    A = pmultiply(X.T, W)

t1 = time.time()
for i in range(n_trials):
    B = X.T.dot(W)

t2 = time.time()

for i in range(n_trials):
    C = sparse_mk.SpMV_viaMKL(X.T.tocsr(), W)

t3 = time.time()

print np.allclose(A, B)
print np.allclose(A, C)

print 'This Code : %.5fs' % ((t1 - t0) / n_trials)
print 'Scipy     : %.5fs' % ((t2 - t1) / n_trials)
print 'Intel MKL : %.5fs' % ((t3 - t2) / n_trials)
