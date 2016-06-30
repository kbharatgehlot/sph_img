"""
Utility functions
"""

import os
import time
import itertools
from multiprocessing import Pool

import tables

import numpy as np
from astropy import constants as const
from scipy.special import sph_jn
from libwise import nputils, plotutils

import healpy as hp

import Ylm

LOFAR_STAT_POS = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'statpos.data')

NUM_POOL = int(os.environ.get('OMP_NUM_THREADS', 2))

nputils.set_random_seed(10)


class Vlm2VisTransMatrix(object):
    ''' Object that encapsulate all the necessary steps to create the independents
        transformation matrix that can be used to obtain independently the real
        and imaginary part of the visibilities. Also have method to split and recombine
        the vlm.

        Example:
        trm = Vlm2VisTransMatrix(ll, mm, ylm, jn)
        vlm_r, vlm_i = trm.split(vlm)
        Vreal = np.dot(vlm_r, trm.T_r.T)
        Vimag = np.dot(vlm_i, trm.T_i.T)
        vlm = trm.recombine(vlm_r, vlm_i)

        '''

    def __init__(self, ll, mm, ylm, jn):
        self.lm_size = len(ll)

        self.lm_even = ((mm != 0) & np.logical_not(is_odd(ll))).astype(bool)
        self.m0_l_even = ((mm == 0) & np.logical_not(is_odd(ll))).astype(bool)
        self.ll_r = np.hstack((ll[self.m0_l_even], ll[self.lm_even], ll[self.lm_even]))
        self.mm_r = np.hstack((mm[self.m0_l_even], mm[self.lm_even], mm[self.lm_even]))
        self.T_r = np.vstack((ylm.real[self.m0_l_even, :] * jn[self.m0_l_even, :], 
                        2 * ylm.real[self.lm_even, :] * jn[self.lm_even, :], 
                        - 2 * ylm.imag[self.lm_even, :] * jn[self.lm_even, :]))

        self.lm_odd = ((mm != 0) & (is_odd(ll))).astype(bool)
        self.m0_l_odd = ((mm == 0) & (is_odd(ll))).astype(bool)
        self.ll_i = np.hstack((ll[self.m0_l_odd], ll[self.lm_odd], ll[self.lm_odd]))
        self.mm_i = np.hstack((mm[self.m0_l_odd], mm[self.lm_odd], mm[self.lm_odd]))
        self.T_i = np.vstack((ylm.real[self.m0_l_odd, :] * jn[self.m0_l_odd, :], 
                        2 * ylm.imag[self.lm_odd, :] * jn[self.lm_odd, :], 
                        2 * ylm.real[self.lm_odd, :] * jn[self.lm_odd, :]))

    def split(self, vlm):
        ''' Split the vlm to be used to recover Re(V) and Im(V) independently'''
        vlm_r = np.hstack((vlm.real[self.m0_l_even], vlm.real[self.lm_even], vlm.imag[self.lm_even]))
        vlm_i = np.hstack((vlm.imag[self.m0_l_odd], vlm.real[self.lm_odd], vlm.imag[self.lm_odd]))
        
        return (vlm_r, vlm_i)

    def recombine(self, vlm_r, vlm_i):
        ''' Recombine vlm_r and vlm_i to a full, complex vlm'''
        split_r = (self.m0_l_even.sum(), self.m0_l_even.sum() + self.lm_even.sum())
        split_i = (self.m0_l_odd.sum(), self.m0_l_odd.sum() + self.lm_odd.sum())

        vlm = np.zeros(self.lm_size, dtype=np.complex)
        vlm[self.m0_l_even] = vlm_r[:split_r[0]] 
        vlm[self.m0_l_odd] = 1j * vlm_i[:split_i[0]]
        vlm[self.lm_even] = vlm_r[split_r[0]:split_r[1]] + 1j * vlm_r[split_r[1]:]
        vlm[self.lm_odd] = vlm_i[split_i[0]:split_i[1]] + 1j * vlm_i[split_i[1]:]

        return vlm


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

    def __init__(self, ll, mm, ylm, jn):
        self.lm_size = len(ll)

        self.lm_even = ((mm != 0) & np.logical_not(is_odd(ll))).astype(bool)
        self.m0_l_even = ((mm == 0) & np.logical_not(is_odd(ll))).astype(bool)
        self.ll_r = np.hstack((ll[self.m0_l_even], ll[self.lm_even], ll[self.lm_even]))
        self.mm_r = np.hstack((mm[self.m0_l_even], mm[self.lm_even], mm[self.lm_even]))

        p_m0 = ((-1) ** (ll[self.m0_l_even] / 2))[:, np.newaxis]
        p_mp = ((-1) ** (ll[self.lm_even] / 2))[:, np.newaxis]
        self.T_r = 4 * np.pi * np.vstack((p_m0 * ylm.real[self.m0_l_even, :] * jn[self.m0_l_even, :], 
                        p_mp * 2 * ylm.real[self.lm_even, :] * jn[self.lm_even, :], 
                        p_mp * -2 * ylm.imag[self.lm_even, :] * jn[self.lm_even, :]))

        self.lm_odd = ((mm != 0) & (is_odd(ll))).astype(bool)
        self.m0_l_odd = ((mm == 0) & (is_odd(ll))).astype(bool)
        self.ll_i = np.hstack((ll[self.m0_l_odd], ll[self.lm_odd], ll[self.lm_odd]))
        self.mm_i = np.hstack((mm[self.m0_l_odd], mm[self.lm_odd], mm[self.lm_odd]))

        p_m0 = ((-1) ** (ll[self.m0_l_odd] / 2))[:, np.newaxis]
        p_mp = ((-1) ** (ll[self.lm_odd] / 2))[:, np.newaxis]
        self.T_i = - 4 * np.pi * np.vstack((p_m0 * ylm.real[self.m0_l_odd, :] * jn[self.m0_l_odd, :], 
                        p_mp * 2 * ylm.real[self.lm_odd, :] * jn[self.lm_odd, :], 
                        p_mp * -2 * ylm.imag[self.lm_odd, :] * jn[self.lm_odd, :]))

    def split(self, alm):
        ''' Split the alm to be used to recover Re(V) and Im(V) independently'''
        alm_r = np.hstack((alm.real[self.m0_l_even], alm.real[self.lm_even], alm.imag[self.lm_even]))
        alm_i = np.hstack((alm.real[self.m0_l_odd], alm.real[self.lm_odd], alm.imag[self.lm_odd]))
        
        return (alm_r, alm_i)

    def recombine(self, alm_r, alm_i):
        ''' Recombine alm_r and alm_i to a full, complex alm'''
        split_r = (self.m0_l_even.sum(), self.m0_l_even.sum() + self.lm_even.sum())
        split_i = (self.m0_l_odd.sum(), self.m0_l_odd.sum() + self.lm_odd.sum())

        alm = np.zeros(self.lm_size, dtype=np.complex)
        alm[self.m0_l_even] = alm_r[:split_r[0]] 
        alm[self.m0_l_odd] = alm_i[:split_i[0]]
        alm[self.lm_even] = alm_r[split_r[0]:split_r[1]] + 1j * alm_r[split_r[1]:]
        alm[self.lm_odd] = alm_i[split_i[0]:split_i[1]] + 1j * alm_i[split_i[1]:]

        return alm


class AbstractMatrix(object):

    def __init__(self, rows, cols, dtype=np.dtype(np.float64)):
        self.rows = np.unique(rows)
        self.cols = np.unique(cols)
        self.dtype = dtype
        self.init_matrix()

    def init_matrix(self):
        self.array = np.zeros((len(self.rows), len(self.cols)), dtype=self.dtype)
        self.build_matrix(self.array)

    def build_matrix(self, array):
        pass

    def get(self, rows, cols):
        rows_uniq, rev_row_idx = np.unique(rows, return_inverse=True)
        idx_row = np.where(np.in1d(self.rows, rows_uniq))[0]

        cols_uniq, rov_col_idx = np.unique(cols, return_inverse=True)
        idx_col = np.where(np.in1d(self.cols, cols_uniq))[0]

        return self.array[idx_row, :][rev_row_idx, :][:, idx_col][:, rov_col_idx]


class AbstractCachedMatrix(AbstractMatrix):

    def __init__(self, name, rows, cols, cache_dir, dtype=np.dtype(np.float64), 
                 force_build=False, keep_in_mem=False):
        self.cache_dir = cache_dir
        self.force_build = force_build
        self.name = name
        self.keep_in_mem = keep_in_mem
        AbstractMatrix.__init__(self, rows, cols, dtype)

    def init_matrix(self):
        atom = tables.Atom.from_dtype(self.dtype)
        hid = get_hash_list_np_array([self.rows, self.cols])

        if not os.path.exists(self.cache_dir):
            print "\nCreating cache directory"
            os.mkdir(self.cache_dir)

        cache_file = os.path.join(self.cache_dir, '%s_%s.cache' % (self.name, hid))

        if not os.path.exists(cache_file) or self.force_build:
            print '\nBuilding matrix with size: %sx%s ...' % (len(self.rows), len(self.cols))
            start = time.time()

            cache_file_temp = cache_file + '.temp'
            
            with tables.open_file(cache_file_temp, 'w') as h5_file:
                array = h5_file.create_array('/', 'data', shape=(len(self.rows), len(self.cols)), 
                                             atom=atom)
                self.build_matrix(array)

            os.rename(cache_file_temp, cache_file)

            print 'Done in %.2f s' % (time.time() - start)

        self.h5_file = tables.open_file(cache_file, 'r')

        if self.keep_in_mem:
            self.array = self.h5_file.root.data[:, :]
        else:
            self.array = self.h5_file.root.data

    def build_matrix(self, array):
        pass

    def close(self):
        self.h5_file.close()


class GenericCachedMatrixMultiProcess(AbstractCachedMatrix):

    def __init__(self, name, rows, cols, cache_dir, row_func, dtype=np.dtype(np.float64), 
                 force_build=False):
        self.row_func = row_func
        AbstractCachedMatrix.__init__(self, name, rows, cols, cache_dir, dtype=dtype, force_build=force_build)

    def build_matrix(self, array):
        pool = Pool(processes=NUM_POOL)

        results_async = [pool.apply_async(self.row_func, (row, self.cols)) for row in self.rows]
        for i, result in enumerate(results_async):
            array[i, :] = result.get(timeout=2)


class YlmCachedMatrix(AbstractCachedMatrix):

    def __init__(self, ll, mm, phis, thetas, cache_dir, dtype=np.dtype(np.complex128),
                 force_build=False, keep_in_mem=False):
        rows, row_idx = np.unique(int_pairing(ll, mm), return_index=True)
        cols, col_idx = np.unique(real_pairing(phis, thetas), return_index=True)
        self.ll = ll[row_idx]
        self.mm = mm[row_idx]
        self.phis = phis[col_idx]
        self.thetas = thetas[col_idx]

        AbstractCachedMatrix.__init__(self, 'ylm', rows, cols, cache_dir, 
                                 dtype=dtype, force_build=force_build, keep_in_mem=keep_in_mem)

    def build_matrix(self, array):
        pool = Pool(processes=NUM_POOL)

        results_async = [pool.apply_async(Ylm.Ylm, (l, m, self.phis, self.thetas)) \
                            for l, m in zip(self.ll, self.mm)]
        for i, result in enumerate(results_async):
            array[i, :] = result.get(timeout=2)

        pool.close()

    def get(self, ll, mm, phis, thetas):
        rows = int_pairing(ll, mm)
        cols = real_pairing(phis, thetas)
        return AbstractCachedMatrix.get(self, rows, cols)


class JnMatrix(AbstractMatrix):

    def __init__(self, ll, ru):
        self.ll = np.unique(ll)
        self.ru = np.unique(ru)
        AbstractMatrix.__init__(self, self.ll, self.ru, dtype=np.dtype(np.float64))

    def build_matrix(self, array):
        pool = Pool(processes=NUM_POOL)
    
        results_async = [pool.apply_async(sph_jn, (max(self.ll), 2 * np.pi * r)) for r in self.ru]
            # return np.array([sph_jn(max(uniq), 2 * np.pi * r)[0][uniq][idx] for r in ru]).T

        for i, result in enumerate(results_async):
            array[:, i] = result.get(timeout=2)[0][self.ll]
        
        pool.close()


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
        if m <0:
            ll.extend(m_l[::-1])
        else:
            ll.extend(m_l)
    return np.array(ll), np.array(mm)


def strip_mm(ll, mm, mmax_fct):
    ''' Strip the mm according to the mmax_fct.

        Ex: ll, mm = strip_mm(ll, mm, lambda l: 0.5 * l) '''
    ll2 = ll[mm <= [mmax_fct(l) for l in ll]].astype(int)
    mm2 = mm[mm <= [mmax_fct(l) for l in ll]].astype(int)

    return ll2, mm2


def get_lm_selection_index(ll1, mm1, ll2, mm2):
    ''' Return the index of all modes (ll2, mm2) into (ll, mm).'''
    x = np.array([l + 1 / (m + 1.) for l, m in zip(ll1, mm1)])
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


def alm2map(alm, ll, mm, thetas, phis):
    ylm = get_ylm(ll, mm, phis, thetas)
    return np.dot(alm[mm == 0], ylm[mm == 0, :]) + 2 * (np.dot(alm.real[mm != 0], ylm.real[mm != 0, :]) - np.dot(alm.imag[mm != 0], ylm.imag[mm != 0, :]))


def vlm2vis(vlm, ll, mm, uthetas, uphis, rus):
    ylm = get_ylm(ll, mm, uphis, uthetas)
    jn = get_jn(ll, rus)
    trm = Vlm2VisTransMatrix(ll, mm, ylm, jn)
    vlm_r, vlm_i = trm.split(vlm)

    return np.dot(vlm_r, trm.T_r) + 1j * np.dot(vlm_i, trm.T_i)


def fast_alm2map(alm, ll, mm, nside):
    if hp.Alm.getsize(max(ll)) != len(alm):
        full_ll, full_mm = get_lm(max(ll))
        full_alm = np.zeros_like(full_ll, dtype=alm.dtype)
        idx = get_lm_selection_index(full_ll, full_mm, ll, mm)
        full_alm[idx] = alm
        alm = full_alm

    return hp.alm2map(alm, nside, verbose=False)


def map2alm(map, thetas, phis, ll, mm):
    # Note: hp.map2alm use jacobi iteration to improve the result
    ylm = get_ylm(ll, mm, phis, thetas)
    return 4 * np.pi / len(map) * np.dot(map, np.conj(ylm).T)


def is_odd(num):
    return num & 0x1


def get_power_spectra(alm, ll, mm):
    l_uniq = np.unique(ll)
    # return np.array([np.sum(np.abs(alm[ll == l]) ** 2) / (2 * np.sum(ll == l)) for l in l_uniq])
    return np.array([np.sum(np.abs(alm[ll == l]) ** 2) / (2 * l) for l in l_uniq])


def sph2cart(theta, phi, r=None):
    """Convert spherical coordinates to 3D cartesian
    theta, phi, and r must be the same size and shape, if no r is provided then unit sphere coordinates are assumed (r=1)
    theta: colatitude/elevation angle, 0(north pole) =< theta =< pi (south pole)
    phi: azimuthial angle, 0 <= phi <= 2pi
    r: radius, 0 =< r < inf
    returns X, Y, Z arrays of the same shape as theta, phi, r
    see: http://mathworld.wolfram.com/SphericalCoordinates.html
    """
    if r is None: r = np.ones_like(theta) #if no r, assume unit sphere radius

    #elevation is pi/2 - theta
    #azimuth is ranged (-pi, pi]
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
    phi = np.arctan2(Y, X) + np.pi #convert azimuth (-pi, pi] to phi (0, 2pi]
    theta = np.pi / 2. - np.arctan2(Z, np.sqrt(X**2. + Y**2.)) #convert elevation [pi/2, -pi/2] to theta [0, pi]

    return r, phi, theta


def real_pairing(a, b):
    return 2 ** a * 3 ** b


def int_pairing(a, b):
    ''' Cantor pairing function '''
    return 0.5 * (a + b) * (a + b + 1) + b


def gaussian_beam(thetas, fwhm):
    # fwhm in radians, centered at NP
    sigma = nputils.gaussian_fwhm_to_sigma(fwhm)

    gaussian_sph = np.exp(-(0.5  * (thetas / sigma) ** 2))

    return gaussian_sph


def sinc2_beam(thetas, fwhm, null_below_horizon=True):
    # fwhm in radians, centered at NP    
    hwhm = fwhm / 2.
    sinc_sph = (np.sin((thetas * 1.4 / hwhm)) / (thetas * 1.4 / hwhm)) ** 2

    if null_below_horizon:
        sinc_sph[thetas > np.pi / 2.] = 0

    return sinc_sph


def tophat_beam(thetas, width):
    # width in radians, centered at NP

    return (thetas <= width / 2.)


def cart_uv(umax, n, rnd_w=False, freqs_mhz=None):
    u = np.linspace(-umax, umax, n)
    v = np.linspace(-umax, umax, n)
    uu, vv = np.meshgrid(u, v)

    if rnd_w:
        uthetas = np.pi / 4. + nputils.get_random().rand(*uu.shape) * np.pi / 2.
        ww = np.sqrt(uu ** 2 + vv ** 2) * np.tan(uthetas - np.pi / 2.)
    else:
        ww = np.ones_like(uu) * 0.
        uthetas = np.arctan(np.sqrt(uu ** 2 + vv ** 2) / ww)

    uu = uu.flatten()
    vv = vv.flatten()
    ww = ww.flatten()

    if freqs_mhz is not None:
        uu = [uu] * len(freqs_mhz)
        vv = [vv] * len(freqs_mhz)
        ww = [ww] * len(freqs_mhz)

    return uu, vv, ww


def cart_nu_uv(bmax, n, freqs_mhz, rnd_w=False):
    lambs = const.c.value / (np.array(freqs_mhz) * 1e6)

    uu, vv, ww = cart_uv(bmax, n, rnd_w=rnd_w, freqs_mhz=freqs_mhz)

    uu = [uu_s / lamb for uu_s, lamb in zip(uu, lambs)]
    vv = [vv_s / lamb for vv_s, lamb in zip(vv, lambs)]
    ww = [ww_s / lamb for ww_s, lamb in zip(ww, lambs)]

    return uu, vv, ww


def polar_uv(rumin, rumax, nr, nphi, rnd_w=False, freqs_mhz=None, rnd_ru=False):
    if freqs_mhz is None:
        freqs_mhz = [150]
    all_ru = []
    all_uphis = []
    all_uthetas = []

    if rnd_w:
        uthetas = np.pi / 4. + nputils.get_random().rand(nr * nphi) * np.pi / 2.
    else:
        uthetas = np.ones(nr * nphi) * np.pi / 2.

    for freq in freqs_mhz:
        uphis = np.arange(0, 2 * np.pi, step=2 * np.pi / float(nphi))
        if rnd_ru:
            # r = nputils.get_random(int(freq)).uniform(rumin, rumax, nr)
            r = np.linspace(rumin, rumax, num=nr)
            r += nputils.get_random(int(freq)).randn(nr) * 0.2 * (rumax - rumin) / float(nr)
        else:
            r = np.linspace(rumin, rumax, num=nr)
        uphis, ru = np.meshgrid(uphis, r)
        uphis = uphis.flatten()
        ru = ru.flatten()

        all_ru.append(ru)
        all_uphis.append(uphis)
        all_uthetas.append(uthetas)

    return all_ru, all_uphis, all_uthetas


def polar_nu_uv(bmin, bmax, nr, nphi, freqs_mhz, rnd_w=False, rnd_ru=False):
    lambs = const.c.value / (np.array(freqs_mhz) * 1e6)

    ru, uphis, uthetas = polar_uv(bmin, bmax, nr, nphi, rnd_w=rnd_w, 
                        freqs_mhz=freqs_mhz, rnd_ru=rnd_ru)

    ru = [ru_s / lamb for ru_s, lamb in zip(ru, lambs)]

    return ru, uphis, uthetas


def lofar_uv(freqs_mhz, dec_deg, hal, har, umin, umax, timeres, include_conj=True, 
                min_max_is_baselines=False):
    m2a = lambda m: np.squeeze(np.asarray(m))

    lambs = const.c.value / (np.array(freqs_mhz) * 1e6)
    k = 2 * np.pi / lambs

    timev = np.arange(hal * 3600, har * 3600, timeres)

    statpos = np.loadtxt(LOFAR_STAT_POS)
    nstat = statpos.shape[0]

    #All combinations of nant to generate baselines
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
            HA  =  (tt / 3600.) * (15. / 180) * np.pi - (6.8689389 / 180) * np.pi
            dec =  dec_deg * (np.pi / 180)
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
                ru = np.sqrt(bu ** 2 + bv ** 2 + bw **2)
            else:
                ru = np.sqrt(u ** 2 + v ** 2 + w **2)
            
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

    return uu, vv, ww


def vlm2alm(vlm, ll):
    return vlm / (4 * np.pi * (-1j) ** ll)


def alm2vlm(alm, ll):
    return 4 * np.pi * alm * (-1j) ** ll


def get_vlm2vis_matrix(ll, mm, ylm, jn):
    return Vlm2VisTransMatrix(ll, mm, ylm, jn)


def get_alm2vis_matrix(ll, mm, ylm, jn):
    return Alm2VisTransMatrix(ll, mm, ylm, jn)


def get_hash_np_array(array):
    writeable = array.flags.writeable
    array.flags.writeable = False
    h = hash(array.data)
    array.flags.writeable = writeable

    return h

def get_hash_list_np_array(l):
    return hash(tuple([get_hash_np_array(array) for array in l]))


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

    uu, vv, ww = lofar_uv([35, 40, 45], 45, -6, 6, 0, 30, 600, min_max_is_baselines=False)
    plt.figure()
    colors = plotutils.ColorSelector()
    uphis = []
    uthetas = []
    for u, v, w in zip(uu, vv, ww):
        ru, uphi, utheta = cart2sph(u, v, w)
        # print ru
        # print len(np.unique(uphi)), len(np.unique(utheta))
        print len(ru), len(np.unique(np.round(ru, decimals=2))), len(np.unique(np.round(uphi, decimals=12))), len(np.unique(np.round(utheta, decimals=12)))
        uphis.extend(uphi)
        uthetas.extend(utheta)
        plt.scatter(u, v, c=colors.get(), marker='+')
        # plt.scatter(ru, uphi, c=colors.get())
    # print len(np.unique(uphis)), len(np.unique(uthetas))
    # print len(np.unique(zip(np.round(uphis, decimals=14), np.round(uthetas, decimals=14))))
    # print len(np.unique(np.round(uphis, decimals=14))), len(np.unique(np.round(uthetas, decimals=14)))
    plt.figure()
    plt.plot(np.unique(uphis), np.unique(uphis), ls='', marker='o')
    plt.plot(sorted(uphi), sorted(uphi), ls='', marker='+')
    plt.show()


def row_func(row, cols):
    # return Ylm.Ylm(483, 234, cols, cols)
    return cols + 0.01 * row


def test_cached_matrix():
    
    class TestCachedMatrix(CachedMatrix):

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
    array = GenericCachedMatrixMultiProcess('test_mp', rows, cols, os.path.dirname(os.path.realpath(__file__)), row_func)

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
    jn3 = get_jn2(ll, ru[0])
    print jn3.shape
    print "Done:", time.time() - start

    print "Selecting"
    start = time.time()
    jn = jn_m.get(ll, ru[0])
    print "Done:", time.time() - start

    print np.allclose(jn, jn2)
    print np.allclose(jn2, jn3)


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
    print len(np.unique(int_pairing(mm,ll)))

    a = np.random.rand(10000)
    b = np.random.rand(10000)

    print len(np.unique(real_pairing(a, b)))

if __name__ == '__main__':
    # test_cached_matrix()
    # test_cached_mp_matrix()
    # test_cached_ylm()
    # test_uv_cov()
    # test_lm_index()
    # test_ylm_precision()
    # test_pairing()
    test_jn()
