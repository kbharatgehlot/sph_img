"""
Utility functions
"""

import os
import sys
import time
import itertools
from multiprocessing import Pool

import tables

import numpy as np
from astropy import constants as const
import astropy.wcs as pywcs
import astropy.io.fits as pf
from scipy.special import sph_jn

from libwise import nputils, plotutils

import healpy as hp

from scipy.interpolate import RectBivariateSpline

import numexpr as ne
import weave

from scipy import signal
from scipy import interpolate
from scipy.special import j1 as bessel_j1

import Ylm


LOFAR_STAT_POS = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'statpos.data')

NUM_POOL = int(os.environ.get('OMP_NUM_THREADS', 2))

nputils.set_random_seed(10)

ne.set_num_threads(NUM_POOL)


def write_fits_gridded_visibilities(file, data, du, freq, dfreq):
    nx, ny = data.shape
    wcs = pywcs.WCS(naxis=4)

    # crpix is with origin 1:
    wcs.wcs.crpix = [nx / 2 + 1, ny / 2 + 1, 1, 1]
    wcs.wcs.crval = [0, 0, freq, 1]
    wcs.wcs.cdelt = [du, du, dfreq, 1]
    wcs.wcs.ctype = ['U---WAV', 'V---WAV', 'FREQ', 'STOCKES']

    hdu = pf.PrimaryHDU(data[np.newaxis, np.newaxis])
    hdu.data.flags.writeable = True
    header = wcs.to_header()
    hdu.header.update(header)

    hdulist = pf.HDUList([hdu])
    hdulist.writeto(file, clobber=True)


def write_gridded_visibilities(dirname, basename, V, umin, umax, du, freq, dfreq, sort_idx=None):
    g_real = os.path.join(dirname, basename + '_GR.fits')
    g_imag = os.path.join(dirname, basename + '_GI.fits')

    n = np.ceil(2 * umax / du)

    u = du * np.arange(-n / 2, n / 2)
    v = du * np.arange(-n / 2, n / 2)

    n = len(u)

    g_uu, g_vv = np.meshgrid(u, v)
    g_uu = g_uu.flatten()
    g_vv = g_vv.flatten()

    g_ru = np.sqrt(g_uu ** 2 + g_vv ** 2)

    idx = (g_ru > umin) & (g_ru < umax)
    # if sort_idx is None:
    #     sort_idx = slice(None)
    sort_idx = nputils.sort_index(g_ru[idx])

    flat_data = np.zeros_like(g_uu, dtype=np.complex)
    flat_nz_data = flat_data[idx]
    flat_nz_data[sort_idx] = V
    flat_data[idx] = flat_nz_data

    data = flat_data.reshape(n, n)

    write_fits_gridded_visibilities(g_real, data.real, du, freq, dfreq)
    write_fits_gridded_visibilities(g_imag, data.imag, du, freq, dfreq)

    return data


class Alm2VisTransMatrix(object):
    ''' Object that encapsulate all the necessary steps to create the independents
        transformation matrix that can be used to obtain independently the real
        and imaginary part of the visibilities. Also have method to split and recombine
        the alm.

        Example:
        trm = Alm2VisTransMatrix(ll, mm, ylm, jn)
        alm_r, alm_i = trm.split(alm)
        Vreal = np.dot(alm_r, trm.T_r.T)
        Vimag = np.dot(alm_i, trm.T_i.T)
        alm = trm.recombine(alm_r, alm_i)

        '''

    def __init__(self, ll, mm, ylm_set, lamb, order='C'):
        self.lm_size = len(ll)
        pi = np.pi

        t = time.time()
        ylm_lm_even, ylm_m0_l_even, ylm_lm_odd, ylm_m0_l_odd = ylm_set

        self.m0_l_even = get_lm_selection_index(ll, mm, ylm_m0_l_even.ll, ylm_m0_l_even.mm, keep_order=True)
        self.m0_l_odd = get_lm_selection_index(ll, mm, ylm_m0_l_odd.ll, ylm_m0_l_odd.mm, keep_order=True)
        self.lm_even = get_lm_selection_index(ll, mm, ylm_lm_even.ll, ylm_lm_even.mm, keep_order=True)
        self.lm_odd = get_lm_selection_index(ll, mm, ylm_lm_odd.ll, ylm_lm_odd.mm, keep_order=True)

        self.ll_r = np.hstack((ylm_m0_l_even.ll, ylm_lm_even.ll, ylm_lm_even.ll))
        self.mm_r = np.hstack((ylm_m0_l_even.mm, ylm_lm_even.mm, ylm_lm_even.mm))

        p_m0 = ((-1) ** (ylm_m0_l_even.ll / 2))[:, np.newaxis]
        p_mp = ((-1) ** (ylm_lm_even.ll / 2))[:, np.newaxis]

        i1 = len(ylm_m0_l_even.ll)
        i2 = len(ylm_lm_even.ll)

        # print "Time lm_selection %.2f s" % (time.time() - t)
        t = time.time()
        self.T_r = np.zeros((len(self.ll_r), ylm_m0_l_even.data.shape[1]), order=order)
        # print "Time T_r init %.2f s" % (time.time() - t)
        t = time.time()

        if len(ylm_m0_l_even.ll) > 0:
            jn = get_jn_fast_weave(ylm_m0_l_even.ll, ylm_m0_l_even.rb / lamb)
            r = ylm_m0_l_even.data.real
            ne.evaluate('4 * pi * p_m0 * r * jn', out=self.T_r[:i1, :])
            # print "Time m0 %.2f s" % (time.time() - t)

        t = time.time()
        r = ylm_lm_even.data.real
        i = ylm_lm_even.data.imag
        jn = get_jn_fast_weave(ylm_lm_even.ll, ylm_lm_even.rb / lamb)
        # print "Time jn %.2f s" % (time.time() - t)

        t = time.time()
        ne.evaluate('4 * pi * p_mp * 2 * r * jn', out=self.T_r[i1:i1 + i2, :])
        # print "Time ev1 %.2f s" % (time.time() - t)

        t = time.time()
        ne.evaluate('4 * pi * p_mp * -2 * i * jn', out=self.T_r[i1 + i2:i1 + i2 + i2, :])
        # print "Time ev2 %.2f s" % (time.time() - t)

        self.ll_i = np.hstack((ylm_m0_l_odd.ll, ylm_lm_odd.ll, ylm_lm_odd.ll))
        self.mm_i = np.hstack((ylm_m0_l_odd.mm, ylm_lm_odd.mm, ylm_lm_odd.mm))

        p_m0 = ((-1) ** (ylm_m0_l_odd.ll / 2))[:, np.newaxis]
        p_mp = ((-1) ** (ylm_lm_odd.ll / 2))[:, np.newaxis]

        i1 = len(ylm_m0_l_odd.ll)
        i2 = len(ylm_lm_odd.ll)

        self.T_i = np.zeros((len(self.ll_i), ylm_m0_l_even.data.shape[1]), order=order)

        if len(ylm_m0_l_odd.ll) > 0:
            r = ylm_m0_l_odd.data.real
            jn = get_jn_fast_weave(ylm_m0_l_odd.ll, ylm_m0_l_odd.rb / lamb)
            ne.evaluate('- 4 * pi * p_m0 * r * jn', out=self.T_i[:i1, :])

        if len(ylm_lm_odd.ll) > 0:
            r = ylm_lm_odd.data.real
            i = ylm_lm_odd.data.imag
            jn = get_jn_fast_weave(ylm_lm_odd.ll, ylm_lm_odd.rb / lamb)

            ne.evaluate('-4 * pi * p_mp * 2 * r * jn', out=self.T_i[i1:i1 + i2, :])
            ne.evaluate('-4 * pi * p_mp * -2 * i * jn', out=self.T_i[i1 + i2:i1 + i2 + i2, :])

    def split(self, alm):
        ''' Split the alm to be used to recover Re(V) and Im(V) independently'''
        alm_r = np.hstack((alm.real[self.m0_l_even], alm.real[self.lm_even], alm.imag[self.lm_even]))
        alm_i = np.hstack((alm.real[self.m0_l_odd], alm.real[self.lm_odd], alm.imag[self.lm_odd]))

        return (alm_r, alm_i)

    def recombine(self, alm_r, alm_i):
        ''' Recombine alm_r and alm_i to a full, complex alm'''
        split_r = (len(self.m0_l_even), len(self.m0_l_even) + len(self.lm_even))
        split_i = (len(self.m0_l_odd), len(self.m0_l_odd) + len(self.lm_odd))

        alm = np.zeros(self.lm_size, dtype=np.complex)
        alm[self.m0_l_even] = alm_r[:split_r[0]]
        alm[self.m0_l_odd] = alm_i[:split_i[0]]
        alm[self.lm_even] = alm_r[split_r[0]:split_r[1]] + 1j * alm_r[split_r[1]:]
        alm[self.lm_odd] = alm_i[split_i[0]:split_i[1]] + 1j * alm_i[split_i[1]:]

        return alm


class AbstractSimpleMatrix(object):

    def __init__(self, n_rows, n_cols, dtype=np.dtype(np.float64)):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.dtype = dtype
        self.init_matrix()

    def init_matrix(self):
        self.data = np.zeros((self.n_rows, self.n_cols), dtype=self.dtype)
        self.build_matrix(self.data)

    def build_matrix(self, array):
        pass

    def get(self):
        return self.data


class AbstractMatrix(object):

    def __init__(self, rows, cols, dtype=np.dtype(np.float64)):
        self.rows = np.unique(rows)
        self.cols = np.unique(cols)
        self.dtype = dtype
        self.init_matrix()

    def init_matrix(self):
        self.data = np.zeros((len(self.rows), len(self.cols)), dtype=self.dtype)
        self.build_matrix(self.data)

    def build_matrix(self, array):
        pass

    def get_full(self):
        return self.data

    def get(self, rows, cols):
        rows_uniq, rev_row_idx = np.unique(rows, return_inverse=True)
        idx_row = np.where(np.in1d(self.rows, rows_uniq))[0]

        cols_uniq, rov_col_idx = np.unique(cols, return_inverse=True)
        idx_col = np.where(np.in1d(self.cols, cols_uniq))[0]

        # PERF: this take quite some time.
        return self.data[idx_row, :][rev_row_idx, :][:, idx_col][:, rov_col_idx]


class AbstractIndexedMatrix(object):

    def __init__(self, rows, cols, idx_cols, dtype=np.dtype(np.float64)):
        self.idx_cols = idx_cols
        self.rows = rows
        self.cols = cols
        self.dtype = dtype
        self.init_matrix()

    def init_matrix(self):
        self.data = np.zeros((len(self.rows), len(self.cols)), dtype=self.dtype)
        self.build_matrix(self.data)

    def build_matrix(self, array):
        pass

    def get_chunk(self, min_idx_col, max_idx_col):
        max_idx_col = min(max_idx_col, max(self.idx_cols))
        left_col = np.nonzero(self.idx_cols >= min_idx_col)[0][0]
        right_col = np.nonzero(self.idx_cols <= max_idx_col)[0][-1]

        idx_col = slice(left_col, right_col + 1)
        return idx_col, self.data[:, idx_col]

    def get(self, rows, cols):
        idx_row = np.where(np.in1d(self.rows, rows))[0]
        idx_col = np.where(np.in1d(self.cols, cols))[0]

        # PERF: this take quite some time.
        return self.data[idx_row, :][:, idx_col]


class AbstractCachedMatrix(object):

    def __init__(self, name, cache_dir, force_build=False, keep_in_mem=False, compress=None):
        self.cache_dir = cache_dir
        self.force_build = force_build
        self.name = name
        self.keep_in_mem = keep_in_mem
        self.compress = compress
        self.h5_file = None

    def init_matrix(self):
        if len(self.rows) == 0 or len(self.cols) == 0:
            self.data = np.zeros((len(self.rows), len(self.cols)), dtype=self.dtype)
            return

        atom = tables.Atom.from_dtype(self.dtype)
        hid = get_hash_list_np_array([self.rows, self.cols])

        if not os.path.exists(self.cache_dir):
            print "\nCreating cache directory"
            os.mkdir(self.cache_dir)

        cache_file = os.path.join(self.cache_dir, '%s_%s.cache' % (self.name, hid))

        if not os.path.exists(cache_file) or self.force_build:
            print '\nBuilding matrix with size: %sx%s ...' % (len(self.rows), len(self.cols)),
            start = time.time()

            if self.compress is not None:
                filters = tables.Filters(complib=self.compress, complevel=5)
            else:
                filters = None

            cache_file_temp = cache_file + '.temp'

            with tables.open_file(cache_file_temp, 'w') as h5_file:
                array = h5_file.create_carray('/', 'data', shape=(len(self.rows), len(self.cols)),
                                              atom=atom, filters=filters)
                self.build_matrix(array)

            os.rename(cache_file_temp, cache_file)

            print 'Done in %.2f s' % (time.time() - start)
        else:
            print '\nUsing cached matrix from disk with size: %sx%s' % (len(self.rows), len(self.cols))

        self.h5_file = tables.open_file(cache_file, 'r')

        if self.keep_in_mem:
            start = time.time()
            print 'Mapping matrix to memory...',
            self.data = self.h5_file.root.data[:, :]
            print 'Done in %.2f s' % (time.time() - start)
        else:
            self.data = self.h5_file.root.data

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()


class YlmCachedMatrix(AbstractCachedMatrix, AbstractMatrix):

    def __init__(self, ll, mm, phis, thetas, cache_dir, dtype=np.dtype(np.complex128),
                 force_build=False, keep_in_mem=False, compress=None):
        self.ll = ll
        self.mm = mm
        self.phis = phis
        self.thetas = thetas
        rows = int_pairing(ll, mm)
        cols = real_pairing(phis, thetas)

        AbstractCachedMatrix.__init__(self, 'ylm', cache_dir, force_build=force_build,
                                      keep_in_mem=keep_in_mem, compress=compress)
        AbstractMatrix.__init__(self, rows, cols, dtype)

    def build_matrix(self, array):
        pool = Pool(processes=NUM_POOL)

        results_async = [pool.apply_async(Ylm.Ylm, (l, m, self.phis, self.thetas))
                         for l, m in zip(self.ll, self.mm)]
        for i, result in enumerate(results_async):
            array[i, :] = result.get()

        pool.close()

    def get(self, ll, mm, phis, thetas):
        rows = int_pairing(ll, mm)
        cols = real_pairing(phis, thetas)
        return AbstractMatrix.get(self, rows, cols)


def ft_phaser_fct(u, v, w, l, m):
    n = np.sqrt(1 - l ** 2 - m ** 2)
    return 1 / n * np.exp(-2 * np.pi * 1j * (u * l + v * m + w * n))


class FtMatrix(AbstractSimpleMatrix):

    def __init__(self, uu, vv, ww, l, m):
        self.uu = uu
        self.vv = vv
        self.ww = ww
        self.l = l
        self.m = m

        AbstractSimpleMatrix.__init__(self, len(self.uu), len(self.l), np.dtype(np.complex128))

    def build_matrix(self, array):
        pool = Pool(processes=NUM_POOL)

        # results_async = [pool.apply_async(ft_phaser_fct, (u, v, w, self.l, self.m))
        #                  for u, v, w in zip(self.uu, self.vv, self.ww)]
        results_async = [pool.apply_async(ft_phaser_fct, (self.uu, self.vv, self.ww, l, m))
                         for l, m in zip(self.l, self.m)]

        for i, result in enumerate(results_async):
            array[:, i] = result.get()

        pool.close()


class YlmChunck(object):

    def __init__(self, ll, mm, phis, thetas, rb, sort_idx_cols, data):
        self.ll = ll
        self.mm = mm
        self.phis = phis
        self.thetas = thetas
        self.rb = rb
        self.data = data
        self.sort_idx_cols = sort_idx_cols


class YlmIndexedCachedMatrix(AbstractCachedMatrix, AbstractIndexedMatrix):

    def __init__(self, ll, mm, phis, thetas, rb, cache_dir, dtype=np.dtype(np.complex128),
                 force_build=False, keep_in_mem=False, compress=None):
        self.sort_idx_cols = nputils.sort_index(rb)
        self.ll = ll
        self.mm = mm
        self.phis = phis[self.sort_idx_cols]
        self.thetas = thetas[self.sort_idx_cols]
        self.rb = rb[self.sort_idx_cols]
        rows = int_pairing(self.ll, self.mm)
        cols = real_pairing(self.phis, self.thetas)

        AbstractCachedMatrix.__init__(self, 'ylm', cache_dir, force_build=force_build,
                                      keep_in_mem=keep_in_mem, compress=compress)
        AbstractIndexedMatrix.__init__(self, rows, cols, self.rb, dtype=dtype)

    def build_matrix(self, array):
        pool = Pool(processes=NUM_POOL)

        results_async = [pool.apply_async(Ylm.Ylm, (l, m, self.phis, self.thetas))
                         for l, m in zip(self.ll, self.mm)]
        for i, result in enumerate(results_async):
            array[i, :] = result.get()

        pool.close()

    def get_chunk(self, bmin, bmax):
        idx_col, data = AbstractIndexedMatrix.get_chunk(self, bmin, bmax)

        return YlmChunck(self.ll, self.mm, self.phis[idx_col],
                         self.thetas[idx_col], self.rb[idx_col],
                         self.sort_idx_cols[idx_col], data)

    def get(self, ll, mm, phis, thetas):
        rows = int_pairing(ll, mm)
        cols = real_pairing(phis, thetas)
        return AbstractIndexedMatrix.get(self, rows, cols)


class MatrixSet(list):

    def get(self, rows, cols):
        return MatrixSet([m.get(rows, cols) for m in self])

    def get_chunk(self, min_idx_col, max_idx_col):
        return MatrixSet([m.get_chunk(min_idx_col, max_idx_col) for m in self])


class SplittedYlmMatrix(MatrixSet):

    def __init__(self, ll, mm, phis, thetas, rb, cache_dir, dtype=np.dtype(np.complex128),
                 force_build=False, keep_in_mem=False, compress=None):
        lm_even = ((mm != 0) & np.logical_not(is_odd(ll))).astype(bool)
        m0_l_even = ((mm == 0) & np.logical_not(is_odd(ll))).astype(bool)
        lm_odd = ((mm != 0) & (is_odd(ll))).astype(bool)
        m0_l_odd = ((mm == 0) & (is_odd(ll))).astype(bool)

        matrices = []
        for idx in [lm_even, m0_l_even, lm_odd, m0_l_odd]:
            sel_ll = ll[idx]
            sel_mm = mm[idx]
            ylm = YlmIndexedCachedMatrix(sel_ll, sel_mm, phis, thetas, rb, cache_dir, dtype=dtype,
                                         force_build=force_build, keep_in_mem=keep_in_mem, compress=compress)
            matrices.append(ylm)

        self.phis = matrices[0].phis
        self.thetas = matrices[0].thetas
        self.rb = matrices[0].rb

        MatrixSet.__init__(self, matrices)

    def close(self):
        for m in self:
            m.close()


class JnMatrix(AbstractMatrix):

    def __init__(self, ll, ru):
        self.full_ll = ll
        self.full_ru = ru
        self.ll = np.unique(ll)
        self.ru = np.unique(ru)
        AbstractMatrix.__init__(self, self.ll, self.ru, dtype=np.dtype(np.float64))

    def build_matrix(self, array):
        pool = Pool(processes=NUM_POOL)

        results_async = [pool.apply_async(sph_jn, (max(self.ll), 2 * np.pi * r)) for r in self.ru]

        for i, result in enumerate(results_async):
            array[:, i] = result.get()[0][self.ll]

        pool.close()

    def get_full(self):
        return self.get(self.full_ll, self.full_ru)


class SplittedJnMatrix(MatrixSet):

    def __init__(self, ylm_set, lamb):
        matrices = []
        for ylm in ylm_set:
            jn = JnMatrix(ylm.ll, ylm.rb / lamb)
            matrices.append(jn)
        MatrixSet.__init__(self, matrices)


def get_lm(lmax, lmin=0, dl=1, mmin=0, mmax=-1, dm=1, neg_m=False):
    ''' Create set of ll and mm'''
    if mmax == -1:
        mmax = lmax

    ll = []
    mm = []
    all_l = np.arange(lmin, lmax + 1, dl)
    all_m = np.arange(mmin, mmax + 1, dm)
    if neg_m:
        all_m = np.concatenate([-np.arange(max(1, mmin), mmax + 1, dm)[::-1], all_m])
    for m in all_m:
        m_l = all_l[all_l >= abs(m)]
        mm.extend([m] * len(m_l))
        if m < 0:
            ll.extend(m_l[::-1])
        else:
            ll.extend(m_l)
    return np.array(ll, dtype=int), np.array(mm, dtype=int)


def strip_mm(ll, mm, mmax_fct):
    ''' Strip the mm according to the mmax_fct.

        Ex: ll, mm = strip_mm(ll, mm, lambda l: 0.5 * l) '''
    ll2 = ll[mm < np.clip(mmax_fct(ll), 1, max(ll))].astype(int)
    mm2 = mm[mm < np.clip(mmax_fct(ll), 1, max(ll))].astype(int)

    return ll2, mm2


def get_sampled_lm(lmax, lmin=0, dl=1, mmin=0, mmax=-1, dm=1, neg_m=False, m_theta_max=None):
    ll, mm = get_lm(lmin=lmin, lmax=lmax, dl=dl, mmax=mmax, mmin=mmin)

    if m_theta_max is not None:
        ll, mm = strip_mm(ll, mm, lambda l: np.sin(m_theta_max) * l)

    return ll, mm


def get_lm_selection_index(ll1, mm1, ll2, mm2, keep_order=False):
    ''' Return the index of all modes (ll2, mm2) into (ll, mm).'''
    x = np.array([l + 1 / (m + 1.) for l, m in zip(ll1, mm1)])

    if keep_order:
        y = np.array([l + 1 / (m + 1.) for l, m in zip(ll2, mm2)])
        idx_x = np.argsort(x)
        sorted_x = x[idx_x]
        idx_y = np.searchsorted(sorted_x, y)
        idx = idx_x[idx_y]
    else:
        y = np.unique(np.array([l + 1 / (m + 1.) for l, m in zip(ll2, mm2)]))
        idx = np.where(np.in1d(x, y, assume_unique=False))[0]

    return idx


def merge_lm(lls, mms):
    # negative m not supported
    ll_concat = np.concatenate(lls)
    mm_concat = np.concatenate(mms)
    lmmax = np.max([ll_concat, mm_concat])

    full_ll, full_mm = get_lm(lmmax)

    idx = get_lm_selection_index(full_ll, full_mm, ll_concat, mm_concat)
    ll = full_ll[idx]
    mm = full_mm[idx]

    return ll, mm


def get_ylm(ll, mm, phis, thetas):
    return np.array([Ylm.Ylm(l, m, phis, thetas) for l, m in zip(ll, mm)])


def get_jn(ll, ru):
    uniq, idx = np.unique(ll, return_inverse=True)
    return np.array([sph_jn(max(uniq), 2 * np.pi * r)[0][uniq][idx] for r in ru]).T


def get_dct(n, nk, ni=None, nki=None, s=0, sk=0, fct=np.cos):
    if ni is None:
        ni = n
    if nki is None:
        nki = nk
    a = np.linspace(0, n - 1, ni)[:, np.newaxis] + s
    b = np.linspace(0, nk - 1, nki) + sk
    return np.sqrt(2. / n) * fct(np.pi / n * a * b)


def get_dct4(n, nk, ni=None, nki=None):
    return get_dct(n, nk, ni=ni, nki=nki, s=0.5, sk=0.5)


def get_dst4(n, nk, ni=None, nki=None):
    return get_dct(n, nk, ni=ni, nki=nki, s=0.5, sk=0.5, fct=np.sin)


def get_dct2(n, nk, ni=None, nki=None):
    dct = get_dct(n, nk, ni=ni, nki=nki, s=0.5, sk=0)
    # dct = np.sqrt(2. / n) * np.cos(np.pi / n * (np.arange(n)[:, np.newaxis] + 0.5) * (np.arange(nk)))
    dct[:, 0] = dct[:, 0] / np.sqrt(2)

    return dct


def get_dct3(n, nk, ni=None, nki=None):
    dct = get_dct(n, nk, ni=ni, nki=nki, s=0, sk=0.5)
    # dct = np.sqrt(2. / n) * np.cos(np.pi / n * (np.arange(n)[:, np.newaxis]) * (np.arange(nk) + 0.5))
    dct[0, :] = 1 / np.sqrt(n)

    return dct


def sparse_to_dense_weave_openmp(sparse, idx_x, idx_y):
    nx = len(idx_x)
    ny = len(idx_y)
    ny_sparse = sparse.shape[1]
    sparse = np.ascontiguousarray(sparse)
    res = np.zeros((nx, ny))

    code = '''
    long i;
    long j;
    #pragma omp parallel for private(i) private(j)
    for(i = 0; i < nx; i++)
        for(j = 0; j < ny; j++)
            res[i * ny + j] = sparse[idx_x[i] * ny_sparse + idx_y[j]];
    '''

    weave.inline(code, ['res', 'nx', 'ny', 'ny_sparse', 'idx_x', 'idx_y', 'sparse'],
                 extra_compile_args=['-march=native  -O3  -fopenmp '],
                 support_code=r"""
                    #include <stdio.h>
                    #include <omp.h>
                    #include <math.h>""",
                 libraries=['gomp'])

    return res


def sparse_to_dense_weave(sparse, idx_x, idx_y):
    nx = len(idx_x)
    ny = len(idx_y)
    ny_sparse = sparse.shape[1]
    sparse = np.ascontiguousarray(sparse)
    res = np.zeros((nx, ny))

    code = '''
    long i;
    long j;
    for(i = 0; i < nx; i++)
        for(j = 0; j < ny; j++)
            res[i * ny + j] = sparse[idx_x[i] * ny_sparse + idx_y[j]];
    '''

    weave.inline(code, ['res', 'nx', 'ny', 'ny_sparse', 'idx_x', 'idx_y', 'sparse'])

    return res


def get_jn_fast_weave(ll, ru, fct=sparse_to_dense_weave_openmp):
    uniq, idx = np.unique(ll, return_inverse=True)
    uniq_r, idx_r = np.unique(ru, return_inverse=True)
    sparse = np.array([sph_jn(max(uniq), 2 * np.pi * r)[0][uniq] for r in uniq_r]).T

    return fct(sparse, idx, idx_r)


def get_jn_fast(ll, ru):
    uniq, idx = np.unique(ll, return_inverse=True)
    uniq_r, idx_r = np.unique(ru, return_inverse=True)
    return np.array([sph_jn(max(uniq), 2 * np.pi * r)[0][uniq][idx] for r in uniq_r])[idx_r].T


def get_lm_map(alm, ll, mm):
    m_uniq = np.unique(mm)
    l_uniq = list(np.unique(ll))
    ma = np.zeros((len(m_uniq), len(l_uniq)), dtype=alm.dtype)
    for i, m in enumerate(m_uniq):
        v = alm[mm == m]
        idx = [l_uniq.index(l) for l in ll[mm == m]]
        ma[i, idx] = v
    return ma


def alm2map(alm, ll, mm, thetas, phis, ylm=None):
    if ylm is None:
        ylm = get_ylm(ll, mm, phis, thetas)
    a = np.dot(alm[mm == 0], ylm[mm == 0, :])
    b = 2 * (np.dot(alm.real[mm != 0], ylm.real[mm != 0, :]) - np.dot(alm.imag[mm != 0], ylm.imag[mm != 0, :]))
    return a + b


def get_full_alm(alm, ll, mm):
    full_ll, full_mm = get_lm(max(ll))
    full_alm = np.zeros_like(full_ll, dtype=alm.dtype)
    idx = get_lm_selection_index(full_ll, full_mm, ll, mm)
    full_alm[idx] = alm
    return full_alm


def fast_alm2map(alm, ll, mm, nside):
    if hp.Alm.getsize(max(ll)) != len(alm):
        alm = get_full_alm(alm, ll, mm)

    return hp.alm2map(alm, nside, verbose=False)


def map2alm(map, thetas, phis, ll, mm):
    # Note: hp.map2alm use jacobi iteration to improve the result
    ylm = get_ylm(ll, mm, phis, thetas)
    return 4 * np.pi / len(map) * np.dot(map, np.conj(ylm).T)


def is_odd(num):
    return num & 0x1


def alm_to_cartmap(alm, ll, mm, res, nx, ny, cache_dir='cache'):
    thxval = res * np.arange(-nx / 2., nx / 2.)
    thyval = res * np.arange(-ny / 2., ny / 2.)
    thx, thy = np.meshgrid(thxval, thyval)

    thz = np.sqrt(1 - thx ** 2 - thy ** 2)
    rs, phis, thetas = cart2sph(thx, thy, thz)

    # ylm = util.get_ylm(ll, mm, phis.flatten(), thetas.flatten())
    ylm_obj = YlmCachedMatrix(ll, mm, phis.flatten(), thetas.flatten(), cache_dir, keep_in_mem=True)
    ylm = ylm_obj.get_full()

    a = np.dot(alm[mm == 0], ylm[mm == 0, :])
    b = 2 * (np.dot(alm.real[mm != 0], ylm.real[mm != 0, :]) - np.dot(alm.imag[mm != 0], ylm.imag[mm != 0, :]))
    cart_map = a + b

    ylm_obj.close()
    del ylm

    return cart_map.reshape(nx, ny).real


def filter_cart_map(cart_map, res, umin, umax):
    m_u = 1 / res * np.linspace(-1 / 2., 1 / 2., cart_map.shape[0])
    m_v = 1 / res * np.linspace(-1 / 2., 1 / 2., cart_map.shape[1])
    m_uu, m_vv = np.meshgrid(m_u, m_v)

    m_ru = np.sqrt(m_uu ** 2 + m_vv ** 2)

    ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(cart_map)))
    ft[(m_ru <= umin) | (m_ru >= umax)] = 0

    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(ft)).real)


def sph2cart(theta, phi, r=None):
    """Convert spherical coordinates to 3D cartesian
    theta, phi, and r must be the same size and shape, if no r is provided
            then unit sphere coordinates are assumed (r=1)
    theta: colatitude/elevation angle, 0(north pole) =< theta =< pi (south pole)
    phi: azimuthial angle, 0 <= phi <= 2pi
    r: radius, 0 =< r < inf
    returns X, Y, Z arrays of the same shape as theta, phi, r
    see: http://mathworld.wolfram.com/SphericalCoordinates.html
    """
    if r is None:
        r = np.ones_like(theta)  # if no r, assume unit sphere radius

    # elevation is pi/2 - theta
    # azimuth is ranged (-pi, pi]
    X = r * np.cos((np.pi / 2.) - theta) * np.cos(phi - np.pi)
    Y = r * np.cos((np.pi / 2.) - theta) * np.sin(phi - np.pi)
    Z = r * np.sin((np.pi / 2.) - theta)

    return X, Y, Z


def cart2sph(X, Y, Z):
    """Convert 3D cartesian coordinates to spherical coordinates
    X, Y, Z: arrays of the same shape and size
    returns r: radius, 0 =< r < inf
            phi: azimuthial angle, 0 <= phi <= 2pi
            theta: colatitude/elevation angle, 0(north pole) =< theta =< pi (south pole)
    see: http://mathworld.wolfram.com/SphericalCoordinates.html
    """
    r = np.sqrt(X**2. + Y**2. + Z**2.)
    phi = np.arctan2(Y, X) + np.pi  # convert azimuth (-pi, pi] to phi (0, 2pi]
    theta = np.pi / 2. - np.arctan2(Z, np.sqrt(X**2. + Y**2.))  # convert elevation [pi/2, -pi/2] to theta [0, pi]

    return r, phi, theta


def cartmap2healpix(cart_map, res, nside):
    ''' res: resolution in radians
        nside: The healpix nside parameter, must be power of 2 and should match res '''

    nx, ny = cart_map.shape
    x = (np.arange(nx) - nx / 2) * res
    y = (np.arange(ny) - ny / 2) * res

    hp_map = np.zeros(hp.nside2npix(nside))
    thetas, phis = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    sph_x, sph_y, sph_z = sph2cart(thetas, phis)

    interp_fct = RectBivariateSpline(x, y, cart_map)

    idx = (thetas < min(nx, ny) / 2 * res)
    hp_map[idx] = interp_fct.ev(sph_x[idx], sph_y[idx])

    return hp_map


def real_pairing(a, b):
    return 2 ** a * 3 ** b


def int_pairing(a, b):
    ''' Cantor pairing function '''
    return 0.5 * (a + b) * (a + b + 1) + b


def gaussian_beam(thetas, fwhm):
    # fwhm in radians, centered at NP
    sigma = nputils.gaussian_fwhm_to_sigma(fwhm)

    gaussian_sph = np.exp(-(0.5 * (thetas / sigma) ** 2))

    return gaussian_sph


def sinc2_beam(thetas, fwhm, null_below_horizon=True, n_sidelibe=None):
    # fwhm in radians, centered at NP
    hwhm = fwhm / 2.
    sinc_sph = (np.sin((thetas * 1.4 / hwhm)) / (thetas * 1.4 / hwhm)) ** 2

    if null_below_horizon:
        sinc_sph[thetas > np.pi / 2.] = 0

    if n_sidelibe is not None:
        sinc_sph[thetas > hwhm / 1.4 * np.pi * (n_sidelibe + 1)] = 0

    return sinc_sph


def bessel_beam(thetas, fwhm):
    beam = (2 / np.pi * fwhm / thetas * bessel_j1(np.pi * thetas / fwhm)) ** 2
    beam[thetas == 0] = 1
    return beam


def tophat_beam(thetas, width):
    # width in radians, centered at NP

    return (thetas <= width / 2.)


def get_beam(thetas, beam_type, fwhm, n_sidelobe):
    if beam_type == 'gaussian':
        beam = gaussian_beam(thetas, fwhm)
    elif beam_type == 'sinc2':
        beam = sinc2_beam(thetas, fwhm, n_sidelibe=n_sidelobe)
    elif beam_type == 'bessel':
        beam = bessel_beam(thetas, fwhm)
    elif beam_type == 'tophat':
        beam = tophat_beam(thetas, fwhm)
    else:
        return None

    return beam


def get_cart_thetas(res, shape):
    nx, ny = shape
    thxval = res * np.arange(-nx / 2., nx / 2.)
    thyval = res * np.arange(-ny / 2., ny / 2.)
    thx, thy = np.meshgrid(thxval, thyval)

    return np.sqrt(thx ** 2 + thy ** 2)


def cart_uv(rumin, rumax, du, rnd_w=False, freqs_mhz=None):
    n = np.ceil(2 * rumax / du)
    u = du * np.arange(-n / 2, n / 2)
    v = du * np.arange(-n / 2, n / 2)
    uu, vv = np.meshgrid(u, v)
    uu = uu.flatten()
    vv = vv.flatten()

    ru = np.sqrt(uu ** 2 + vv ** 2)

    idx = (ru > rumin) & (ru < rumax)

    uu = uu[idx]
    vv = vv[idx]

    if rnd_w:
        uthetas = np.pi / 4. + nputils.get_random().rand(*uu.shape) * np.pi / 2.
        ww = np.sqrt(uu ** 2 + vv ** 2) * np.tan(uthetas - np.pi / 2.)
    else:
        ww = np.zeros_like(uu)
        uthetas = np.ones_like(uu) * np.pi / 2.

    if freqs_mhz is not None:
        uu = np.array([uu] * len(freqs_mhz))
        vv = np.array([vv] * len(freqs_mhz))
        ww = np.array([ww] * len(freqs_mhz))

    return uu, vv, ww


def polar_uv(rumin, rumax, nr, nphi, rnd_w=False, freqs_mhz=None, rnd_ru=False):
    if freqs_mhz is None:
        freqs_mhz = [150]
    all_ru = []
    all_uphis = []
    all_uthetas = []

    if rnd_w:
        uthetas = np.pi / 4. + nputils.get_random().rand(int(nr * nphi)) * np.pi / 2.
    else:
        uthetas = np.ones(nr * nphi) * np.pi / 2.

    for freq in freqs_mhz:
        uphis = 2 * np.pi * np.linspace(0, 1, num=nphi, endpoint=False)
        if rnd_ru:
            # r = nputils.get_random(int(freq)).uniform(rumin, rumax, nr)
            r = np.linspace(rumin, rumax, num=nr)
            r += nputils.get_random(int(freq)).randn(int(nr)) * 0.2 * (rumax - rumin) / float(nr)
        else:
            r = np.linspace(rumin, rumax, num=nr)
        uphis, ru = np.meshgrid(uphis, r)
        uphis = uphis.flatten()
        ru = ru.flatten()

        all_ru.append(ru)
        all_uphis.append(uphis)
        all_uthetas.append(uthetas)

    return all_ru, all_uphis, all_uthetas


def lofar_uv(freqs_mhz, dec_deg, hal, har, umin, umax, timeres, include_conj=True,
             min_max_is_baselines=False):
    m2a = lambda m: np.squeeze(np.asarray(m))

    lambs = const.c.value / (np.array(freqs_mhz) * 1e6)

    timev = np.arange(hal * 3600, har * 3600, timeres)

    statpos = np.loadtxt(LOFAR_STAT_POS)
    nstat = statpos.shape[0]

    # All combinations of nant to generate baselines
    stncom = itertools.combinations(np.arange(1, nstat), 2)
    b1, b2 = zip(*stncom)

    uu = []
    vv = []
    ww = []

    for lamb in lambs:
        lamb_u = []
        lamb_v = []
        lamb_w = []

        for tt in timev:
            HA = (tt / 3600.) * (15. / 180) * np.pi - (6.8689389 / 180) * np.pi
            dec = dec_deg * (np.pi / 180)
            RM = np.matrix([[np.sin(HA), np.cos(HA), 0.0],
                            [-np.sin(dec) * np.cos(HA), np.sin(dec) * np.sin(HA), np.cos(dec)],
                            [np.cos(dec) * np.cos(HA), - np.cos(dec) * np.sin(HA), np.sin(dec)]])
            statposuvw = np.dot(RM, statpos.T).T
            bu = m2a(statposuvw[b1, 0] - statposuvw[b2, 0])
            bv = m2a(statposuvw[b1, 1] - statposuvw[b2, 1])
            bw = m2a(statposuvw[b1, 2] - statposuvw[b2, 2])

            u = bu / lamb
            v = bv / lamb
            w = bw / lamb

            if min_max_is_baselines:
                ru = np.sqrt(bu ** 2 + bv ** 2 + bw ** 2)
            else:
                ru = np.sqrt(u ** 2 + v ** 2 + w ** 2)

            idx = (ru > umin) & (ru < umax)

            lamb_u.extend(u[idx])
            lamb_v.extend(v[idx])
            lamb_w.extend(w[idx])

            if include_conj:
                lamb_u.extend(- u[idx])
                lamb_v.extend(- v[idx])
                lamb_w.extend(w[idx])

        uu.append(np.array(lamb_u))
        vv.append(np.array(lamb_v))
        ww.append(np.array(lamb_w))

    return np.array(uu), np.array(vv), np.array(ww)


def griduvw_vis(u, v, w, vis, Nud, Nvd, Wslice, convxwn, convywn, beta):
    Wmin = min(w)
    Wmax = max(w)
    index = np.argsort(w)
    wa = [w[i] for i in index]  # sorted in ascending order
    print '\nMin w value = %s Max w value =%s\n' % (Wmin, Wmax)
    wr = np.linspace(Wmin, Wmax, Wslice)
    print '\nW ranges::\n'
    print wr

    ####################

    print '\nGriding starts::\n'
    base = 0
    Totalgridbase = 0
    nb = len(u)
    print '\nTotal no of baselines %s\n' % (nb)
    delUx = (2. * max(u)) / Nud
    delVy = (2. * max(v)) / Nvd
    window = signal.kaiser(np.round(convxwn), beta)
    xval = np.arange(0, np.round(convxwn), 1)
    tck = interpolate.splrep(xval, window, s=0)

    visamat = []
    umat = []
    vmat = []
    wmat = []

    for ws in range(Wslice):
        countw = 0
        vismat = np.zeros((Nud, Nvd), dtype=complex)
        ug = np.zeros((Nud, Nvd), dtype=np.float)
        vg = np.zeros((Nud, Nvd), dtype=np.float)
        wg = np.zeros((Nud, Nvd), dtype=np.float)
        countgwt = np.zeros((Nud, Nvd), dtype=np.float)
        countg = np.zeros((Nud, Nvd), dtype=np.int)
        # print 'Slice no is %s\n' %(ws)

        for ii in range(base, nb):
            if (wa[ii] >= wr[ws] and wa[ii] <= wr[ws + 1]):
                # print '\nBaselines are choosen within slice %s to %s\n' %(ws, ws+1)
                Uval = u[index[ii]]
                Vval = v[index[ii]]
                Wval = wa[ii]
                base = base + 1
                countw = countw + 1

                if(Uval > 0. and Vval > 0.):
                    uxg = int((Nud + 1) / 2 + np.round(Uval / delUx))
                    if uxg >= Nud:
                        uxg = Nud - 1
                    uyg = int((Nvd + 1) / 2 + np.round(Vval / delVy))
                    if uyg >= Nvd:
                        uyg = Nvd - 1
                if(Uval < 0. and Vval > 0.):
                    uxg = int((Nud + 1) / 2 - np.round(np.abs(Uval) / delUx))
                    if uxg <= 0:
                        uxg = 0
                    uyg = int((Nvd + 1) / 2 + np.round(Vval / delVy))
                    if uyg >= Nvd:
                        uyg = Nvd - 1
                if(Uval < 0. and Vval < 0.):
                    uxg = int((Nud + 1) / 2 - np.round(np.abs(Uval) / delUx))
                    if uxg <= 0:
                        uxg = 0
                    uyg = int((Nvd + 1) / 2 - np.round(np.abs(Vval) / delVy))
                    if uyg <= 0:
                        uyg = 0
                if(Uval > 0. and Vval < 0.):
                    uxg = int((Nud + 1) / 2 + np.round(Uval / delUx))
                    if uxg >= Nud:
                        uxg = Nud - 1
                    uyg = int((Nvd + 1) / 2 - np.round(np.abs(Vval) / delVy))
                    if uyg <= 0:
                        uyg = 0
                if(np.abs((uxg - Nud / 2) * delUx - Uval) <= 1. * convxwn and
                        np.abs((uyg - Nvd / 2) * delVy - Vval) <= 1. * convywn):

                    uvgdiffx = np.abs((uxg - Nud / 2) * delUx - Uval)
                    uvgdiffy = np.abs((uyg - Nvd / 2) * delVy - Vval)
                    wtx = interpolate.splev(np.round(uvgdiffx / delUx), tck, der=0)
                    wty = interpolate.splev(np.round(uvgdiffy / delVy), tck, der=0)
                    # print wtx, wty
                    wt = wtx * wty
                    if(wt < 0. or wt == 0):
                        print '\nWt is negative or zero\n'

                    vismat[uxg, uyg] = vismat[uxg, uyg] + wt * vis[index[ii]]
                    countgwt[uxg, uyg] = countgwt[uxg, uyg] + wt
                    countg[uxg, uyg] = countg[uxg, uyg] + 1
                    ug[uxg, uyg] = ug[uxg, uyg] + wt * Uval
                    vg[uxg, uyg] = vg[uxg, uyg] + wt * Vval
                    wg[uxg, uyg] = Wval

        count = 0
        for ii in range(Nud):
            for jj in range(Nvd):
                if(countgwt[ii, jj] > 0.):
                    count = count + 1

        # print '\nNumber of filled grid points %s\n' %(count)

        # Normalize
        vismatn = np.zeros((count + 1,), dtype=complex)
        ugn = np.zeros((count + 1,), dtype=np.float)
        vgn = np.zeros((count + 1,), dtype=np.float)
        wgn = np.zeros((count + 1,), dtype=np.float)

        count = 0
        for ii in range(Nud):
            for jj in range(Nvd):
                if(countgwt[ii, jj] > 0.):
                    count = count + 1
                    vismatn[count] = vismat[ii, jj] / countgwt[ii, jj]
                    ugn[count] = ug[ii, jj] / countgwt[ii, jj]
                    vgn[count] = vg[ii, jj] / countgwt[ii, jj]
                    wgn[count] = wg[ii, jj] / countg[ii, jj]  # only avg; not weighted avg

        Totalgridbase = Totalgridbase + countw

        visamat = np.append(visamat, vismatn)
        umat = np.append(umat, ugn)
        vmat = np.append(vmat, vgn)
        wmat = np.append(wmat, wgn)

    print '\nNumber of gridded baselines %s :: Base-GridBase = %s\n' % (Totalgridbase, nb - Totalgridbase)

    return umat, vmat, wmat, visamat


def vlm2alm(vlm, ll):
    return vlm / (4 * np.pi * (-1j) ** ll)


def alm2vlm(alm, ll):
    return 4 * np.pi * alm * (-1j) ** ll


def get_alm2vis_matrix(ll, mm, ylm_set, lamb, order='C'):
    return Alm2VisTransMatrix(ll, mm, ylm_set, lamb, order=order)


def get_hash_np_array(array):
    writeable = array.flags.writeable
    array.flags.writeable = False
    h = hash(array.data)
    array.flags.writeable = writeable

    return h


def get_hash_list_np_array(l):
    return hash(tuple([get_hash_np_array(array) for array in l]))


def progress_report(n):
    t = time.time()

    def report(i):
        print "\r",
        eta = ""
        if i > 0:
            remaining = (np.round((time.time() - t) / float(i) * (n - i)))
            eta = " (ETA: %s)" % time.strftime("%H:%M:%S", time.localtime(time.time() + remaining))
        print "Progress: %s / %s%s" % (i + 1, n, eta),
        sys.stdout.flush()
        if i == n - 1:
            print ""

    return report


def progress_tracker(n):
    t = time.time()

    def get_progress(i):
        eta = ""
        if i > 0:
            remaining = (np.round((time.time() - t) / float(i) * (n - i)))
            eta = " (ETA: %s)" % time.strftime("%H:%M:%S", time.localtime(time.time() + remaining))
        return "%s / %s%s" % (i + 1, n, eta)

    return get_progress


def partial_rev(func, *args, **keywords):
    def newfunc(*fargs, **fkeywords):
        newkeywords = keywords.copy()
        newkeywords.update(fkeywords)
        return func(*(fargs + args), **newkeywords)
    newfunc.func = func
    newfunc.args = args
    newfunc.keywords = keywords
    return newfunc


def test_alm2vis():
    ll, mm = get_lm(lmax=100, dl=3)
    alm = np.random.random(ll.size) + 1j * np.random.random(ll.size)
    alm[mm == 0] = alm[mm == 0].real

    vlm = alm2vlm(alm, ll)

    thetas = np.pi * np.random.rand(100)
    phis = 2 * np.pi * np.random.rand(100)
    rus = 8 * np.random.rand(100)
    ylm = get_ylm(ll, mm, phis, thetas)
    jn = get_jn(ll, rus)

    tr1 = Vlm2VisTransMatrix(ll, mm, ylm, jn)
    tr2 = Alm2VisTransMatrix(ll, mm, ylm, jn)

    vlm_r, vlm_i = tr1.split(vlm)
    vis1 = np.dot(vlm_r, tr1.T_r) + 1j * np.dot(vlm_i, tr1.T_i)

    alm_r, alm_i = tr2.split(alm)
    vis2 = np.dot(alm_r, tr2.T_r) + 1j * np.dot(alm_i, tr2.T_i)

    print np.allclose(vis1, vis2)


def test_uv_cov():
    import matplotlib.pyplot as plt

    # uu, vv, ww = cart_nu_uv(40, 4, [100, 200], rnd_w=True)
    # # print len(zip(uu, vv, ww)[0][0])
    # ru, uphis, uthetas = zip(*map(cart2sph, uu, vv, ww))

    # plt.figure()
    # colors = plotutils.ColorSelector()
    # # for u, v in zip(uu, vv):
    # #     plt.scatter(u, v, c=colors.get())
    # for r, phi in zip(ru, uphis):
    #     plt.scatter(r, phi, c=colors.get())
    # plt.xlabel('U')
    # plt.ylabel('V')
    # plt.show()

    # ru, uphis, uthetas = polar_uv(10, 50, 10, 50, rnd_w=False, freqs_mhz=[100, 150, 200], rnd_ru=True)
    # uu, vv, ww = zip(*map(sph2cart, uthetas, uphis, ru))

    # plt.figure()
    # colors = plotutils.ColorSelector()
    # for u, v in zip(uu, vv):
    #     plt.scatter(u, v, c=colors.get())
    # plt.xlabel('U')
    # plt.ylabel('V')
    # plt.show()

    # plt.figure()
    # colors = plotutils.ColorSelector()
    # for r, phi, theta in zip(ru, uphis, uthetas):
    #     plt.scatter(r, phi, c=colors.get())
    # plt.show()

    # plt.xlabel('ru')
    # plt.ylabel('uphis')
    # plt.show()

    # freqs = np.arange(110, 130, 5)
    freqs = [145.]
    uu, vv, ww = lofar_uv(freqs, 90, -6, 6, 45, 255, 200, min_max_is_baselines=False)
    plt.figure()
    colors = plotutils.ColorSelector()
    uphis = []
    uthetas = []
    for u, v, w in zip(uu, vv, ww):
        ru, uphi, utheta = cart2sph(u, v, w)
        # print ru
        # print np.unique(uphi), np.unique(utheta)
        print len(ru), min(ru), max(ru), len(np.unique(np.round(ru, decimals=2))), \
            len(np.unique(np.round(uphi, decimals=12))), len(np.unique(np.round(utheta, decimals=12)))
        uphis.append(np.round(uphi, decimals=12))
        uthetas.append(np.round(utheta, decimals=12))
        plt.scatter(u, v, c=colors.get(), marker='+', s=5)
        # plt.scatter(utheta, uphi, c=colors.get(), marker='+', )
    # print np.allclose(np.unique(uphis[0]), np.unique(uphis[1]))
    # print np.allclose(np.unique(uthetas[0]), np.unique(uthetas[1]))
    # print len(np.unique(uphis)), len(np.unique(uthetas))
    # print len(np.unique(zip(np.round(uphis, decimals=14), np.round(uthetas, decimals=14))))
    # print len(np.unique(np.round(uphis, decimals=14))), len(np.unique(np.round(uthetas, decimals=14)))
    # plt.figure()
    # plt.plot(np.unique(uphis), np.unique(uphis), ls='', marker='o')
    # plt.plot(sorted(uphi), sorted(uphi), ls='', marker='+')
    plt.show()


def row_func(row, cols):
    # return Ylm.Ylm(483, 234, cols, cols)
    return cols + 0.01 * row


def test_cached_matrix():

    class TestCachedMatrix(AbstractCachedMatrix):

        def build_matrix(self, array):
            for i, row in enumerate(self.rows):
                # print i, self.cols + 0.0001 * row
                array[i, :] = self.cols + 0.01 * row

    rows = np.arange(10)
    cols = np.arange(10)
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cache')
    array = TestCachedMatrix('test', rows, cols, dir, force_build=True, dtype=np.dtype(np.float64))
    print array.array[5, :]
    print array.array[9, :]
    print array.get([5, 9], cols)

    # print array.get(rows[0:2], cols[0:2])
    # print array.get([rows[0], rows[1]] * 3, [cols[0], cols[1]] * 3)


def test_cached_mp_matrix():

    # class TestCachedMatrix(CachedMatrixMultiProcess):
    #     pass

        # def build_matrix_row(self, row, cols):
        #     return np.random.random(len(cols)).tolist()

    rows = np.arange(10)
    cols = np.arange(10)
    array = GenericCachedMatrixMultiProcess('test_mp', rows, cols,
                                            os.path.dirname(os.path.realpath(__file__)), row_func)

    print array.array[5, :]
    print array.array[9, :]
    print array.get([5, 9], cols)


def test_cached_ylm():
    ll, mm = get_lm(100)
    # print ll
    # print mm
    ru, uphis, uthetas = polar_uv(50, 200, 20, 200, rnd_w=True)

    ylm_m = YlmCachedMatrix(ll, mm, uphis[0], uthetas[0], os.path.dirname(os.path.realpath(__file__)))

    print "Building simple"
    start = time.time()
    ylm2 = get_ylm(ll, mm, uphis[0], uthetas[0])
    print "Done:", time.time() - start

    print "Selecting"
    start = time.time()
    ylm = ylm_m.get(ll, mm, uphis[0], uthetas[0])
    print "Done:", time.time() - start

    # print ylm[:4, :4].real
    # print ylm2[:4, :4].real

    # print ylm[:4, :4]
    # print ll[:2], mm[:2], uphis[:2], uthetas[:2]
    # print get_ylm(ll[:2], mm[:2], uphis[:2], uthetas[:2])
    # print ylm_m.get(ll[:2], mm[:2], uphis[:2], uthetas[:2])

    print np.allclose(ylm, ylm2)
    # print ylm_m.array[:]
    # print ylm2

    ylm_m.close()


def test_index_matrix():

    class TestMatrix(AbstractIndexedMatrix):

        def build_matrix(self, array):
            for i, row in enumerate(self.rows):
                # print i, self.cols + 0.0001 * row
                array[i, :] = self.cols + 10000 * row

    rows = np.arange(5000)
    cols = np.arange(5000)
    array = TestMatrix(rows, cols, rows, cols, dtype=np.dtype(np.int64))
    t = time.time()
    print array.array[5, :]
    print time.time() - t
    t = time.time()
    print array.array[9, :]
    print time.time() - t
    t = time.time()
    print array.get(rows, cols)
    print time.time() - t
    t = time.time()
    print array.get_chunk(5, 900, 3, 600)
    print time.time() - t


def test_ylm_index_matrix():
    ll, mm = get_lm(200)
    # print ll
    # print mm
    ru, uphis, uthetas = polar_uv(50, 200, 20, 200, rnd_w=True)

    ylm_m = YlmIndexedCachedMatrix(ll, mm, uphis[0], uthetas[0], ll, ru[0],
                                   os.path.dirname(os.path.realpath(__file__)), keep_in_mem=True)
    ll = ylm_m.ll
    mm = ylm_m.mm
    uphis = ylm_m.phis
    uthetas = ylm_m.thetas
    ru = ru[0]

    print "Selecting chunk"
    start = time.time()
    idx_l, idx_r, ylm = ylm_m.get_chunk(10, 180, 40, 180)
    print "Done:", time.time() - start, ylm.shape

    print "Selecting"
    start = time.time()
    # idx_l = (ll >= 10) & (ll <= 20)
    # idx_r = (ru >= 70) & (ru <= 90)
    ylm2 = ylm_m.get(ll[idx_l], mm[idx_l], uphis[idx_r], uthetas[idx_r])
    print "Done:", time.time() - start, ylm2.shape

    print np.allclose(ylm, ylm2)

    # print "Building simple"
    # start = time.time()
    # ylm3 = get_ylm(ll[idx_l], mm[idx_l], uphis[idx_r], uthetas[idx_r])
    # print "Done:", time.time() - start

    # print np.allclose(ylm, ylm3)

    ylm_m.close()


def test_jn():
    ll, mm = get_lm(200, dl=2)
    ru, uphis, uthetas = polar_uv(50, 200, 20, 200, rnd_w=True)

    print "Building MP"
    start = time.time()
    jn_m = JnMatrix(ll, ru[0])
    print "Done:", time.time() - start

    print "Building simple"
    start = time.time()
    jn2 = get_jn(ll, ru[0])
    print jn2.shape
    print "Done:", time.time() - start

    print "Building simple 2"
    start = time.time()
    # jn3 = get_jn2(ll, ru[0])
    # print jn3.shape
    print "Done:", time.time() - start

    print "Selecting"
    start = time.time()
    jn = jn_m.get(ll, ru[0])
    print "Done:", time.time() - start

    print np.allclose(jn, jn2)
    # print np.allclose(jn2, jn3)


def test_lm_index():
    uu, vv, ww = lofar_uv([150], 90, -6, 6, 0, 100, 200, min_max_is_baselines=True)
    ru, uphis, uthetas = zip(*map(cart2sph, uu, vv, ww))
    uthetas = np.round(uthetas[0], decimals=10)
    uthetas_10d = np.array(np.unique(uthetas))
    print uthetas.shape, uthetas_10d.shape

    idx = np.where(np.in1d(uthetas_10d, uthetas, assume_unique=False))[0]
    print idx


def test_ylm_precision():
    ll, mm = get_lm(lmin=600, lmax=602)
    print len(ll)
    phi = 2 * np.pi * np.random.rand(len(ll))
    theta = np.pi * np.random.rand(len(ll))
    ylm1 = get_ylm(ll, mm, phi, theta)
    ylm2 = get_ylm(ll, mm, phi, theta + 1e-12)

    print np.allclose(ylm1, ylm2, atol=1e-10)
    print np.allclose(ylm1, ylm2, atol=1e-9)


def test_pairing():
    ll, mm = get_lm(40)
    print len(ll)
    print len(np.unique(int_pairing(mm, ll)))

    a = np.random.rand(10000)
    b = np.random.rand(10000)

    print len(np.unique(real_pairing(a, b)))


def test_ylm_compress():
    lmin = 1
    lmax = 100
    n = 100
    ll, mm = get_lm(lmin=lmin, lmax=lmax, mmax=lmax)
    uphi = np.linspace(0, 6, n)
    utheta = np.linspace(0.2, np.pi - 0.2, n)
    ur = np.linspace(1.2 * lmin / 6.28, 1 * lmax / 6.28, n)
    uphis, uthetas, urs = np.meshgrid(uphi, utheta, ur)
    urs = urs.flatten()
    uphis = uphis.flatten()
    uthetas = uthetas.flatten()

    t = time.time()
    ylm = YlmCachedMatrix(ll, mm, uphis, uthetas, 'cache', compress=None, force_build=True)
    print time.time() - t

    t = time.time()
    ylm.get(ll[mm == 5], mm[mm == 5], uphis, uthetas)
    print time.time() - t


def test_ylm_set():
    ll, mm = get_lm(100)
    ru, uphis, uthetas = polar_uv(50, 200, 20, 100, rnd_w=True)
    ru, uphis, uthetas = ru[0], uphis[0], uthetas[0]

    ylm_set = SplittedYlmMatrix(ll, mm, uphis, uthetas, ru,
                                os.path.dirname(os.path.realpath(__file__)), keep_in_mem=True)

    jn_set = SplittedJnMatrix(ylm_set, ru)

    lm_even = ((mm != 0) & np.logical_not(is_odd(ll))).astype(bool)
    m0_l_even = ((mm == 0) & np.logical_not(is_odd(ll))).astype(bool)
    lm_odd = ((mm != 0) & (is_odd(ll))).astype(bool)
    m0_l_odd = ((mm == 0) & (is_odd(ll))).astype(bool)

    for idx, ylm, jn in zip([lm_even, m0_l_even, lm_odd, m0_l_odd], ylm_set, jn_set):
        ylm1 = YlmCachedMatrix(ll[idx], mm[idx], uphis, uthetas,
                               os.path.dirname(os.path.realpath(__file__)), keep_in_mem=True)

        jn1 = JnMatrix(ll[idx], ru)
        jn2 = get_jn_fast(ylm.ll, ylm.rb)

        print np.allclose(ylm.data, ylm1.get(ylm.ll, ylm.mm, ylm.phis, ylm.thetas))
        print np.allclose(jn.data, jn1.get(jn.ll, jn.ru))
        print np.allclose(jn.full_ll, ylm.ll)
        print np.allclose(jn.full_ru, ylm.rb)
        print np.allclose(jn2, jn1.get(jn.full_ll, jn.full_ru))

        ylm1.close()

    ylm_set.close()


def test_gridding():
    import matplotlib.pyplot as plt

    uu, vv, ww = lofar_uv([150], 90, -6, 6, 30, 250, 200)
    uu = uu[0]
    vv = vv[0]
    ww = ww[0]
    vis = np.random.randn(len(uu))
    uug, vvg, wwg, visamat = griduvw_vis(uu, vv, ww, vis, 130, 130, 10, 1 / 0.07, 1 / 0.07, 8.6)

    print len(uu)
    print len(uug)
    plt.scatter(uug, vvg, marker='+', s=5)
    plt.show()


if __name__ == '__main__':
    # test_cached_matrix()
    # test_cached_mp_matrix()
    # test_cached_ylm()
    test_uv_cov()
    # test_lm_index()
    # test_ylm_precision()
    # test_pairing()
    # test_jn()
    # test_ylm_compress()
    # test_index_matrix()
    # test_ylm_index_matrix()
    # test_ylm_set()
    # test_gridding()
