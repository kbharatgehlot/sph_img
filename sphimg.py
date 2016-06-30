import os
import imp
import glob
import time
import warnings

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from libwise import plotutils, nputils
from libwise import scriptshelper as sh

from scipy.sparse.linalg import cg

import healpy as hp

import util


def plot_sky(alm, ll, mm, nside, title='', savefile=None):
    map = util.fast_alm2map(alm, ll, mm, nside)
    hp.orthview(map, rot=(180, 90), title=title)
    hp.graticule(verbose=False)

    if savefile is not None:
        plt.gcf().savefig(savefile)
        plt.close()


def plot_sky_cart(alm, ll, mm, nside, title='', theta_max=0.35, savefile=None):
    map = util.fast_alm2map(alm, ll, mm, nside)
    theta_max = np.degrees(theta_max)
    hp.cartview(map, rot=(180, 90), 
            lonra=[-theta_max, theta_max], latra=[-theta_max, theta_max], title=title)
    hp.graticule(verbose=False)

    if savefile is not None:
        plt.gcf().savefig(savefile)
        plt.close()


def plot_sky_cart_diff(alm1, alm2, ll1, mm1, ll2, mm2, nside, theta_max=0.35, savefile=None):
    cbs = plotutils.ColorbarSetting(plotutils.ColorbarInnerPosition(location=2, height="80%", pad=1))
    latra = np.degrees(theta_max)

    map1 = util.fast_alm2map(alm1, ll1, mm1, nside)
    map2 = util.fast_alm2map(alm2, ll2, mm2, nside)
    diff = map1 - map2

    fig = plt.figure(1, figsize=(14, 5))
    hp.cartview(map1, latra=(-latra, latra), lonra=(-latra, latra), rot=(180, 90),
                           xsize=200, fig=fig, sub=131, cbar=False, title='Input sky')
    hp.graticule(dpar=5, verbose=False)
    cbs.add_colorbar(fig.axes[0].get_children()[-2], fig.axes[0])

    hp.cartview(map2, latra=(-latra, latra), lonra=(-latra, latra), rot=(180, 90), 
                           xsize=200, fig=fig, sub=132, cbar=False, title='Output sky')
    hp.graticule(dpar=5, verbose=False)
    cbs.add_colorbar(fig.axes[2].get_children()[-2], fig.axes[2])

    hp.cartview(diff, latra=(-latra, latra), lonra=(-latra, latra), rot=(180, 90), 
                           xsize=200, fig=fig, sub=133, cbar=False, title='Diff')
    hp.graticule(dpar=5, verbose=False)
    cbs.add_colorbar(fig.axes[4].get_children()[-2], fig.axes[4])

    fig.axes[4].text(0.95, 0.05, 'std diff: %.3f\nmax diff: %.3f' % (diff.std(), diff.max()), 
                     ha='right', va='center', transform=fig.axes[4].transAxes)

    if savefile is not None:
        fig.set_size_inches(14, 5)
        fig.savefig(savefile)
        plt.close(fig)


def plot_uv_cov(uu, vv, ww, title, savefile=None):
    fig, (ax1, ax2)  = plt.subplots(ncols=2, figsize=(12, 5))
    ax1.scatter(uu, vv)
    ax1.set_xlabel('U')
    ax1.set_ylabel('V')

    ax2.scatter(uu, ww)
    ax2.set_xlabel('U')
    ax2.set_ylabel('W')
    
    fig.suptitle(title)

    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)


def plot_visibilities(uu, vv, ww, V, savefile=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(uu, vv, ww, c=abs(V), cmap='jet', norm=plotutils.LogNorm(), lw=0)
    ax.set_xlabel('U')
    ax.set_ylabel('V')
    ax.set_zlabel('W')

    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)


def plot_sampling(ll, mm, sel_ll, sel_mm, savefile=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(ll, mm, s=2, c=plotutils.black, marker='x', label='full')
    ax.scatter(sel_ll, sel_mm, s=10, c=plotutils.orange, marker='+', label='sample')
    ax.set_ylabel('m')
    ax.set_xlabel('l')
    ax.legend(loc=2)
    
    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)


def plot_vlm_vs_vlm_rec(sel_ll, sel_mm, sel_vlm, vlm_rec, savefile=None):
    fig, (ax1, ax2)  = plt.subplots(ncols=2, figsize=(12, 5))

    ax1.scatter(sel_mm, abs(sel_vlm), c='blue', marker='o', label='Input')
    ax1.scatter(sel_mm, abs(vlm_rec), c='orange', marker='+', label='Recovered')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-5, 1)
    ax1.set_xlim(0, max(sel_mm))
    ax1.set_xlabel('m')
    ax1.set_ylabel('abs(vlm)')
    ax1.legend()

    ax2.scatter(sel_ll, abs(sel_vlm), c='blue', marker='o', label='Input')
    ax2.scatter(sel_ll, abs(vlm_rec), c='orange', marker='+', label='Recovered')
    ax2.set_yscale('log')
    ax2.set_ylim(1e-5, 5)
    ax2.set_xlim(0, max(sel_ll))
    ax2.set_xlabel('l')
    ax2.set_ylabel('abs(vlm)')
    ax2.legend()

    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)


def plot_vlm_vs_vlm_rec_map(sel_ll, sel_mm, sel_vlm, vlm_rec, fisher_error, savefile=None):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))

    cbs = plotutils.ColorbarSetting(plotutils.ColorbarInnerPosition(location=2, height="80%", pad=1))

    lm_map = util.get_lm_map(abs(sel_vlm), sel_ll, sel_mm)
    lm_map_rec = util.get_lm_map(abs(vlm_rec), sel_ll, sel_mm)
    lm_map_fisher_error = util.get_lm_map(abs(fisher_error), sel_ll, sel_mm)

    extent = (min(sel_ll), max(sel_ll), min(sel_mm), max(sel_mm))

    im_mappable = ax1.imshow(abs(lm_map), norm=plotutils.LogNorm(), vmin=1e-6, vmax=1e2, extent=extent, aspect='auto')
    ax1.set_title('Input vlm')
    cbs.add_colorbar(im_mappable, ax1)

    im_mappable = ax2.imshow(abs(lm_map_rec), norm=plotutils.LogNorm(), vmin=1e-6, vmax=1e2, extent=extent, aspect='auto')
    ax2.set_title('Recovered vlm')
    cbs.add_colorbar(im_mappable, ax2)

    im_mappable = ax3.imshow(abs(lm_map - lm_map_rec), norm=plotutils.LogNorm(), vmin=1e-6, vmax=1e2, extent=extent, aspect='auto')
    ax3.set_title('Diff')
    cbs.add_colorbar(im_mappable, ax3)

    im_mappable = ax4.imshow(lm_map_fisher_error, norm=plotutils.LogNorm(), vmin=1e-6, vmax=1e2, extent=extent, aspect='auto')
    ax4.set_title('Fisher error')
    cbs.add_colorbar(im_mappable, ax4)

    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xlabel('l')
        ax.set_ylabel('m')
        
    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)


def plot_power_sepctra(ll, mm, alm, sel_ll, sel_mm, alm_rec, config, savefile=None):
    #TODO: check the normalization here!
    l_sampled = np.arange(max(sel_ll) + 1)[np.bincount(sel_ll) > (config.out_mmax_full_sample + 1)]
    idx = np.where(np.in1d(sel_ll, l_sampled))[0]
    # print np.bincount(sel_ll)
    # print l_sampled

    ps_rec = util.get_power_spectra(alm_rec[idx], sel_ll[idx], sel_mm[idx]) #* len(sel_ll) / float(len(ll))
    ps = util.get_power_spectra(alm, ll, mm)
    
    fig = plt.figure()
    fig, (ax1, ax2)  = plt.subplots(ncols=2, figsize=(12, 5))
    ax1.plot(np.unique(ll), ps, label='Beam modulated input power spectra')
    ax1.plot(l_sampled, ps_rec, label='Recovered power spectra', marker='+')
    # ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('l')
    ax1.set_ylabel('cl')

    diff = ps_rec - ps[np.in1d(np.unique(ll), l_sampled)]
    # print nputils.stat(diff)
    ax2.plot(l_sampled, diff, ls='', marker='+')
    ax2.set_xlabel('l')
    ax2.set_ylabel('diff')
    ax2.text(0.95, 0.05, 'std diff: %s\nmax diff: %s' % (diff.std(), diff.max()), 
                     ha='right', va='center', transform=ax2.transAxes)

    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)


def plot_mf_power_spectra(ll, mm, alms, freqs, config, savefile=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    l_sampled = np.arange(max(ll) + 1)[np.bincount(ll) > (config.out_mmax_full_sample + 1)]
    idx = np.where(np.in1d(ll, l_sampled))[0]

    for freq, alm in zip(freqs, alms):
        ps = util.get_power_spectra(alm[idx], ll[idx], mm[idx])
        ax.plot(l_sampled, ps, label='%s Mhz', marker='+')

    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('l')
    ax.set_ylabel('cl')

    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)


def plot_vlm_diff(sel_ll, sel_mm, sel_vlm, vlm_rec, savefile=None):
    fig, (ax1, ax2)  = plt.subplots(ncols=2, figsize=(12, 5))

    ax1.scatter(sel_mm, abs(vlm_rec.real - sel_vlm.real), c='green', marker='+', label='Input')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-5, 1e-1)
    ax1.set_xlim(0, max(sel_mm))
    ax1.set_xlabel('m')
    ax1.set_ylabel('abs(vlm.real - vlm_rec.real)')

    ax2.scatter(sel_ll, abs(vlm_rec.real - sel_vlm.real), c='green', marker='+', label='Input')
    ax2.set_yscale('log')
    ax2.set_ylim(1e-5, 1e-1)
    ax2.set_xlim(0, max(sel_ll))
    ax2.set_xlabel('l')
    ax2.set_ylabel('abs(vlm.real - vlm_rec.real)')

    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)


def plot_vis_diff(ru, V, Vobs, Vrec, savefile=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ru * (2 * np.pi), abs(V - Vrec), ls='', marker='+', c='orange', label='V - Vrec')
    ax.plot(ru * (2 * np.pi), abs(V - Vobs), ls='', marker='+', c='green', label='V - Vobs')
    ax.set_yscale('log')
    ax.set_ylabel('abs(vis diff)')
    ax.set_xlabel('2 pi u')
    ax.legend(loc='best')

    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)


def simulate_sky(config):
    ll, mm = util.get_lm(config.lmax)
    nside = config.nside
    lmax = config.lmax
    fwhm = config.fwhm

    print 'Simulating full sky with healpix synfast'
    np.random.seed(config.synfast_rnd_seed)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full_map, alm = hp.synfast(config.cl, config.nside, alm=True, lmax=lmax, verbose=False)
    full_map = full_map + abs(full_map.min())

    # full_map = np.zeros_like(full_map)
    # full_map[hp.ang2pix(nside, 0.05, 0.05)] = 10000

    full_alm = hp.map2alm(full_map, lmax=lmax)

    thetas, phis = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))

    if config.beam_type == 'gaussian':
        print 'Using a gaussian beam with fwhm = %.3f rad' % fwhm
        beam = util.gaussian_beam(thetas, fwhm)

    elif config.beam_type == 'sinc2':
        print 'Using a sinc2 beam with fwhm = %.3f rad' % fwhm
        beam = util.sinc2_beam(thetas, fwhm)

    elif config.beam_type == 'tophat':
        print 'Using a tophat beam with width = %.3f rad' % fwhm
        beam = util.tophat_beam(thetas, width)

    else:
        print 'Not using any beam'
        beam = np.ones_like(thetas)

    map = full_map * beam
    alm = hp.map2alm(map, lmax)

    alms = [alm * n for n in config.cl_freq]

    return ll, mm, alms


def sample_input_alm(config, alms, ll, mm):
    ll_inp, mm_inp = util.get_lm(lmin=config.inp_lmin, lmax=config.inp_lmax, 
                             dl=config.inp_dl, dm=config.inp_dm, 
                             mmax=config.inp_mmax, mmin=config.inp_mmin)
    theta_max = config.out_theta_max

    if config.inp_mmax_strip:
        ll2 = ll_inp[mm_inp < np.clip(theta_max * ll_inp, 1, config.inp_lmax)].astype(int)
        mm2 = mm_inp[mm_inp < np.clip(theta_max * ll_inp, 1, config.inp_lmax)].astype(int)
        ll_inp = ll2
        mm_inp = mm2

    idx = util.get_lm_selection_index(ll, mm, ll_inp, mm_inp)
    alms = [alm[idx] for alm in alms]

    return alms, ll_inp, mm_inp


def simulate_uv_cov(config):
    freqs_mhz = config.freqs_mhz
    if config.uv_type == 'cart':
        print 'Using cartesian uv coverage, fixed umax'
        uu, vv, ww = util.cart_uv(config.cart_umax, config.cart_n, 
                                  rnd_w=config.cart_rnd_w, freqs_mhz=freqs_mhz)
        ru, uphis, uthetas = [list(k) for k in zip(*map(util.cart2sph, uu, vv, ww))]
        title = 'n=%s, umax=%s, random w:%s' % (config.cart_n, config.cart_umax, config.cart_rnd_w)
    
    if config.uv_type == 'cart_nu':
        print 'Using cartesian uv coverage, fixed bmax'
        uu, vv, ww = util.cart_nu_uv(config.cart_bmax, config.cart_n, 
                                  rnd_w=config.cart_rnd_w)
        ru, uphis, uthetas = [list(k) for k in zip(*map(util.cart2sph, uu, vv, ww))]
        title = 'n=%s, bmax=%s, random w:%s' % (config.cart_n, config.cart_bmax, config.cart_rnd_w)
    
    elif config.uv_type == 'polar':
        print 'Using polar uv coverage, fixed umax'
        ru, uphis, uthetas = util.polar_uv(config.polar_rumax, config.polar_rumin, 
                    config.polar_nr, config.polar_nphi, rnd_w=config.polar_rnd_w, 
                    freqs_mhz=freqs_mhz, rnd_ru=config.polar_rnd_ru)
        uu, vv, ww = [list(k) for k in zip(*map(util.sph2cart, uthetas, uphis, ru))]

        title = 'nr=%s, nphi=%s, rumax=%s, rumin=%s, random w=%s' % (config.polar_nr, config.polar_nphi, 
                            config.polar_rumax, config.polar_rumin, config.cart_rnd_w)

    elif config.uv_type == 'polar_nu':
        print 'Using polar uv coverage, fixed bmax'
        ru, uphis, uthetas = util.polar_nu_uv(config.polar_rbmax, config.polar_rbmin, 
                    config.polar_nr, config.polar_nphi, freqs_mhz, 
                    rnd_w=config.polar_rnd_w, rnd_ru=config.polar_rnd_ru)
        uu, vv, ww = [list(k) for k in zip(*map(util.sph2cart, uthetas, uphis, ru))]

        title = 'nr=%s, nphi=%s, rbmax=%s, rbmin=%s, random w=%s' % (config.polar_nr, config.polar_nphi, 
                            config.polar_rbmax, config.polar_rbmin, config.cart_rnd_w)

    elif config.uv_type == 'lofar':
        print 'Using LOFAR uv coverage'
        uu, vv, ww = util.lofar_uv(freqs_mhz, config.lofar_dec_deg, 
                    config.lofar_hal, config.lofar_har, config.lofar_umin, 
                    config.lofar_umax, config.lofar_timeres, include_conj=config.lofar_include_conj,
                    min_max_is_baselines=config.lofar_min_max_is_baselines)
        ru, uphis, uthetas = [list(k) for k in zip(*map(util.cart2sph, uu, vv, ww))]

        if config.lofar_min_max_is_baselines:
            title = 'Lofar with bmin=%s, bmax=%s' % (config.lofar_umin, config.lofar_umax)
        else:
            title = 'Lofar with umin=%s, umax=%s' % (config.lofar_umin, config.lofar_umax)
    else:
        print 'Configuration value uv_type invalid'
        sh.usage(True)

    # round to similar tolerance:
    for k in [uu, vv, ww, ru, uphis, uthetas]:
        for i, s in enumerate(k):
            k[i] = np.round(s, decimals=config.n_decimal_tol)

    return uu, vv, ww, ru, uphis, uthetas, title


def get_out_lm_sampling(ll, mm, config):
    lmax = config.out_lmax
    mmax = config.out_mmax
    mmin = config.out_mmin
    lmin = config.out_lmin
    dl = config.out_dl
    dm = config.out_dm
    theta_max = config.out_theta_max

    if config.out_four_consective:
        dl = 4 * dl
        ll2a, mm2a = util.get_lm(lmin=lmin, lmax=lmax, dl=dl, mmax=mmax, mmin=mmin)
        ll2b, mm2b = util.get_lm(lmin=lmin+1, lmax=lmax, dl=dl, mmax=mmax, mmin=mmin)
        ll2c, mm2c = util.get_lm(lmin=lmin+2, lmax=lmax, dl=dl, mmax=mmax, mmin=mmin)
        ll2d, mm2d = util.get_lm(lmin=lmin+3, lmax=lmax, dl=dl, mmax=mmax, mmin=mmin)
        ll, mm = util.merge_lm([ll2a, ll2b, ll2c, ll2d], [mm2a, mm2b, mm2c, mm2d])
    else:
        ll, mm = util.get_lm(lmin=lmin, lmax=lmax, dl=dl, mmax=mmax, mmin=mmin)

    if config.out_mmax_full_sample >= 0:
        llm0, mmm0 = util.get_lm(lmin=lmin, lmax=lmax, mmax=config.out_mmax_full_sample)
        ll, mm = util.merge_lm([ll, llm0], [mm, mmm0])

    if config.out_mmax_strip:
        ll2 = ll[mm < np.clip(theta_max * ll, 1, lmax)].astype(int)
        mm2 = mm[mm < np.clip(theta_max * ll, 1, lmax)].astype(int)
        ll = ll2
        mm = mm2

    return ll, mm


def compute_visibilities(alm, ll, mm, uphis, uthetas, ru, global_ylm):
    ylm = global_ylm.get(ll, mm, uphis, uthetas)
    # jn = util.get_jn(ll, ru)
    jn = util.JnMatrix(ll, ru).get(ll, ru)

    trm = util.get_alm2vis_matrix(ll, mm, ylm, jn)

    alm_r, alm_i = trm.split(alm)
    V = np.dot(alm_r, trm.T_r) + 1j * np.dot(alm_i, trm.T_i)

    return V


def alm_ml_inversion(ll, mm, Vobs, uphis, uthetas, ru, global_ylm, config, simulate=False):
    print "Building transformation matrix..."
    ylm = global_ylm.get(ll, mm, uphis, uthetas)
    # jn = util.get_jn(ll, ru)
    jn = util.JnMatrix(ll, ru).get(ll, ru)

    trm = util.get_alm2vis_matrix(ll, mm, ylm, jn)

    #Free some memory
    ylm = None
    jn = None

    norm = 1 #len(ll) / float(len(ll))

    print '\nSize of transformation matrix: T_r: %s T_i: %s' % (trm.T_r.shape, trm.T_i.shape)

    C_Dinv = np.diag([1 / (config.noiserms ** 2)] * len(Vobs))

    print '\nComputing LHS and RHS matrix ...'
    lhs_r = np.dot(np.dot(trm.T_r, C_Dinv), trm.T_r.T) + np.eye(len(trm.ll_r)) * config.reg_lambda
    rhs_r = np.dot(np.dot(trm.T_r, C_Dinv), norm * Vobs.real)

    lhs_i = np.dot(np.dot(trm.T_i, C_Dinv), trm.T_i.T) + np.eye(len(trm.ll_i)) * config.reg_lambda
    rhs_i = np.dot(np.dot(trm.T_i, C_Dinv), norm * Vobs.imag)

    print "Building fisher matrix ..."
    fisher_error_i = np.sqrt(np.diag(np.linalg.inv(lhs_i)))
    fisher_error_r = np.sqrt(np.diag(np.linalg.inv(lhs_r)))
    fisher_error = np.abs(trm.recombine(fisher_error_r, fisher_error_i))

    if simulate:
        return fisher_error

    print '\nStarting CG inversion for the real visibilities ...'
    start = time.time()
    alm_rec_r, info = cg(lhs_r, rhs_r, tol=config.cg_tol, maxiter=config.cg_maxiter)
    res = np.linalg.norm(rhs_r - np.dot(lhs_r, alm_rec_r)) / np.linalg.norm(rhs_r)
    print 'Done in %.2f s. Status: %s, Residual: %s' % (time.time() - start, info, res)

    print '\nStarting CG inversion for the imaginary visibilities ...'
    start = time.time()
    alm_rec_i, info = cg(lhs_i, rhs_i, tol=config.cg_tol, maxiter=config.cg_maxiter)
    res = np.linalg.norm(rhs_i - np.dot(lhs_i, alm_rec_i)) / np.linalg.norm(rhs_i)
    print 'Done in %.2f s. Status: %s, Residual: %s' % (time.time() - start, info, res)

    alm_rec = trm.recombine(alm_rec_r, alm_rec_i)
    Vrec = np.dot(alm_rec_r, trm.T_r) + 1j * np.dot(alm_rec_i, trm.T_i)

    return alm_rec, Vrec, fisher_error


def get_config(dirname):
    return imp.load_source('config', os.path.join(dirname, 'config.py'))


def save_alm(dirname, ll, mm, alm, alm_rec, fisher_error):
    filename = os.path.join(dirname, 'alm.dat')
    print "Saving alm result to:", filename

    np.savetxt(filename, np.array([ll, mm, alm.real, alm.imag, alm_rec.real, alm_rec.imag, fisher_error.real, fisher_error.imag]).T)


def load_alm(dirname):
    filename = os.path.join(dirname, 'alm.dat')
    print "Loading alm result from:", filename

    ll, mm, alm_real, alm_imag, alm_rec_real, alm_rec_imag, fisher_error_real, fisher_error_imag = np.loadtxt(filename).T

    return ll, mm, alm_real + 1j * alm_imag, alm_rec_real + 1j * alm_rec_imag, fisher_error_real + 1j * fisher_error_imag


def save_visibilities(dirname, ru, uphis, uthetas, Vobs, Vrec):
    filename = os.path.join(dirname, 'visibilities.dat')
    print "Saving visibilities result to:", filename

    np.savetxt(filename, np.array([ru, uphis, uthetas, Vobs.real, Vobs.imag, Vrec.real, Vrec.imag]).T)


def load_visibilities(dirname):
    filename = os.path.join(dirname, 'visibilities.dat')
    print "Loading visibilities result feom:", filename

    ru, uphis, uthetas, Vobs_real, Vobs_imag, Vrec_real, Vrec_imag = np.loadtxt(filename).T

    return ru, uphis, uthetas, Vobs_real + 1j * Vobs_imag, Vrec_real + 1j * Vrec_imag


def load_results(dirname):
    if not os.path.exists(dirname):
        print "Path does not exists:", dirname
        return

    print "loading result from %s ..." % dirname
    freq_res = []
    for freq_dir in sorted(glob.glob(os.path.join(dirname, 'freq_*'))):
        ll, mm, alm, alm_rec, fisher_error = load_alm(freq_dir)
        ru, uphis, uthetas, Vobs, Vrec = load_visibilities(freq_dir)
        freq_res.append([alm, alm_rec, fisher_error, ru, uphis, uthetas, Vobs, Vrec])

    alm, alm_rec, fisher_error, ru, uphis, uthetas, Vobs, Vrec = zip(*freq_res)

    return ll, mm, alm, alm_rec, fisher_error, ru, uphis, uthetas, Vobs, Vrec


def do_inversion(config, result_dir):
    full_ll, full_mm, full_alms = simulate_sky(config)
    plot_sky_cart(full_alms[0], full_ll, full_mm, config.nside, theta_max=config.fwhm,
        title='Input sky, beam=%.1f deg, lmax=%s' % (np.degrees(config.fwhm), config.lmax), 
        savefile=os.path.join(result_dir, 'input_sky.pdf'))

    inp_alms, inp_ll, inp_mm = sample_input_alm(config, full_alms, full_ll, full_mm)

    uu, vv, ww, ru, uphis, uthetas, uv_params_str = simulate_uv_cov(config)

    vlms = [util.alm2vlm(alm, inp_ll) for alm in inp_alms]

    sel_ll, sel_mm = get_out_lm_sampling(inp_ll, inp_mm, config)

    # plotting the sampling
    plot_sampling(inp_ll, inp_mm, sel_ll, sel_mm, os.path.join(result_dir, 'lm_sampling.pdf'))

    print '\nBuilding the global YLM matrix...'
    all_phis = np.concatenate(uphis)
    all_thetas = np.concatenate(uthetas)
    uniq, idx_uniq = np.unique(util.real_pairing(all_thetas, all_phis), return_index=True)

    global_ylm = util.YlmCachedMatrix(inp_ll, inp_mm, all_phis[idx_uniq], 
                      all_thetas[idx_uniq], config.cache_dir, keep_in_mem=config.keep_in_mem)

    alms_rec = []

    for i, freq in enumerate(config.freqs_mhz):
        print "\nProcessing frequency %s MHz" % freq
        
        result_freq_dir = os.path.join(result_dir, 'freq_%s' % i)
        os.mkdir(result_freq_dir)

        plot_uv_cov(uu[i], vv[i], ww[i], uv_params_str, os.path.join(result_freq_dir, 'uv_cov.pdf'))

        # computing the visibilities
        alm = inp_alms[i]
        print "Building visibilities..."
        V = compute_visibilities(alm, inp_ll, inp_mm, uphis[i], uthetas[i], ru[i], global_ylm)

        Vobs = V + config.noiserms * np.random.randn(len(V)) + 1j * config.noiserms * np.random.randn(len(V))

        # plotting the visibilities
        plot_visibilities(uu[i], vv[i], ww[i], V, os.path.join(result_freq_dir, 'vis_from_vlm.pdf'))

        idx = util.get_lm_selection_index(inp_ll, inp_mm, sel_ll, sel_mm)

        sel_alm = alm[idx]
        sel_vlm = util.alm2vlm(sel_alm, sel_ll)

        alm_rec, Vrec, fisher_error = alm_ml_inversion(sel_ll, sel_mm, Vobs, uphis[i], uthetas[i], 
                                         ru[i], global_ylm, config)

        alms_rec.append(alm_rec)

        save_alm(result_freq_dir, sel_ll, sel_mm, sel_alm, alm_rec, fisher_error)
        save_visibilities(result_freq_dir, ru[i], uphis[i], uthetas[i], Vobs, Vrec)

        vlm_rec = util.alm2vlm(alm_rec, sel_ll)

        # normalization
        norm_alm_rec = alm_rec.copy()
        norm_alm_rec[sel_mm > config.out_mmax_full_sample] = norm_alm_rec[sel_mm > config.out_mmax_full_sample] * config.out_dl

        norm_alm_rec2 = alm_rec.copy()
        norm_alm_rec2[sel_mm > config.out_mmax_full_sample] = norm_alm_rec2[sel_mm > config.out_mmax_full_sample] / float(config.out_dl)

        print "Plotting result"
        # plot vlm vs vlm_rec
        plot_vlm_vs_vlm_rec(sel_ll, sel_mm, sel_vlm, vlm_rec, os.path.join(result_freq_dir, 'vlm_vs_vlm_rec.pdf'))

        # plot vlm vs vlm_rec in a map
        plot_vlm_vs_vlm_rec_map(sel_ll, sel_mm, sel_vlm, vlm_rec, fisher_error, 
            os.path.join(result_freq_dir, 'lm_maps_imag.pdf'))

        # plot power spectra
        plot_power_sepctra(inp_ll, inp_mm, alm, sel_ll, sel_mm, alm_rec, config,
            os.path.join(result_freq_dir, 'angular_power_spectra.pdf'))

        plot_power_sepctra(inp_ll, inp_mm, alm, sel_ll, sel_mm, norm_alm_rec2, config,
            os.path.join(result_freq_dir, 'angular_power_spectra_norm.pdf'))

        # plot vlm diff
        plot_vlm_diff(sel_ll, sel_mm, sel_vlm, vlm_rec, 
            os.path.join(result_freq_dir, 'vlm_minus_vlm_rec.pdf'))

        # plot visibilities diff
        plot_vis_diff(ru[i], V, Vobs, Vrec, os.path.join(result_freq_dir, 'vis_minus_vis_rec.pdf'))

        # plot output sky
        # plot_sky_cart(alm_rec, sel_ll, sel_mm, config.nside, theta_max=config.fwhm,
        #     title='Output sky, beam=%.1f deg, lmax=%s' % (np.degrees(config.fwhm), config.lmax), 
        #     savefile=os.path.join(result_freq_dir, 'output_sky.pdf'))

        plot_sky_cart_diff(alm, norm_alm_rec, inp_ll, inp_mm, sel_ll, sel_mm, config.nside, theta_max=config.fwhm, 
            savefile=os.path.join(result_freq_dir, 'output_sky.pdf'))

    global_ylm.close()

    if len(alms_rec) > 1:
        plot_mf_power_spectra(sel_ll, sel_mm, alms_rec, config.freqs_mhz, config,
            os.path.join(result_dir, 'mf_power_spectra.pdf'))

    print '\nAll done!'

if __name__ == '__main__':
    main()
