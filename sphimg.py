import os
import imp
import glob
import time
import warnings
import multiprocessing

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.image import AxesImage

from libwise import plotutils, nputils
from libwise import scriptshelper as sh

from scipy.sparse.linalg import cg
from scipy.sparse import block_diag, diags
from scipy.fftpack import dct, idct
from psparse import pmultiply

import astropy.constants as const
import astropy.io.fits as pyfits
import astropy.wcs as pywcs

import healpy as hp


import util


def plot_sky(alm, ll, mm, nside, title='', savefile=None):
    dl = np.mean(np.diff(np.unique(ll)))

    map = dl * util.fast_alm2map(alm, ll, mm, nside)
    hp.orthview(map, rot=(180, 90), title=title)
    hp.graticule(verbose=False)

    if savefile is not None:
        plt.gcf().savefig(savefile)
        plt.close()


def plot_sky_cart(alm, ll, mm, nside, title='', theta_max=0.35, savefile=None):
    dl = np.mean(np.diff(np.unique(ll)))

    map = dl * util.fast_alm2map(alm, ll, mm, nside)
    theta_max = np.degrees(theta_max)
    hp.cartview(map, rot=(180, 90),
                lonra=[-theta_max, theta_max], latra=[-theta_max, theta_max], title=title)
    hp.graticule(verbose=False, dpar=theta_max / 4., dmer=theta_max / 4.)

    thetas, phis = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))

    if savefile is not None:
        plt.gcf().savefig(savefile)
        plt.close()

    return thetas, phis, map


def plot_sky_cart_diff(alm1, alm2, ll1, mm1, ll2, mm2, nside, theta_max=0.35, savefile=None):
    cbs = plotutils.ColorbarSetting(plotutils.ColorbarInnerPosition(location=2, height="80%", pad=1))
    latra = np.degrees(theta_max)

    dl_m1 = np.mean(np.diff(np.unique(ll1)))
    dl_m2 = np.mean(np.diff(np.unique(ll2)))

    map1 = dl_m1 * util.fast_alm2map(alm1, ll1, mm1, nside)
    map2 = dl_m2 * util.fast_alm2map(alm2, ll2, mm2, nside)
    diff = map1 - map2

    fig = plt.figure(figsize=(14, 5))
    hp.cartview(map1, latra=(-latra, latra), lonra=(-latra, latra), rot=(180, 90),
                xsize=200, fig=fig, sub=131, cbar=False, title='Input sky')
    hp.graticule(dpar=5, verbose=False)

    hp.cartview(map2, latra=(-latra, latra), lonra=(-latra, latra), rot=(180, 90),
                xsize=200, fig=fig, sub=132, cbar=False, title='Output sky')
    hp.graticule(dpar=5, verbose=False)

    hp.cartview(diff, latra=(-latra, latra), lonra=(-latra, latra), rot=(180, 90),
                xsize=200, fig=fig, sub=133, cbar=False, title='Diff')
    hp.graticule(dpar=5, verbose=False)

    lastax = None

    thetas, phis = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))

    for ax in fig.axes:
        if isinstance(ax, hp.projaxes.HpxCartesianAxes):
            lastax = ax
            for art in ax.get_children():
                if isinstance(art, AxesImage):
                    cbs.add_colorbar(art, ax)

    if lastax is not None:
        diff = diff[thetas <= theta_max]
        lastax.text(0.92, 0.05, 'rms residual: %.3e\nmax residual: %.3e' % (diff.std(), diff.max()),
                    ha='right', va='center', transform=lastax.transAxes)

    if savefile is not None:
        fig.set_size_inches(14, 5)
        fig.savefig(savefile)
        plt.close(fig)


def plot_uv_cov(uu, vv, ww, config, title, savefile=None):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
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
    fig = plt.figure(figsize=(12, 5))
    # ax = fig.add_subplot(121, projection='3d')
    # ax.scatter3D(uu, vv, ww, c=abs(V), cmap='jet', norm=plotutils.LogNorm(), lw=0)
    # ax.set_xlabel('U')
    # ax.set_ylabel('V')
    # ax.set_zlabel('W')

    ru, uphis, uthetas = util.cart2sph(uu, vv, ww)
    ax = fig.add_subplot(111)
    ax.scatter(ru, abs(V), marker='+')
    ax.set_yscale('log')
    ax.set_xlabel('ru')
    ax.set_ylabel('abs(V)')
    ax.set_ylim(min(abs(V)), max(abs(V)))

    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)


def plot_2d_visibilities(uu, vv, V, savefile=None):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

    cbs = plotutils.ColorbarSetting(plotutils.ColorbarOutterPosition())

    mappable = ax1.scatter(uu, vv, c=V.real, lw=0)
    ax1.set_xlabel('U')
    ax1.set_xlabel('V')
    ax1.set_title('Real')

    cbs.add_colorbar(mappable, ax1)

    mappable = ax2.scatter(uu, vv, c=V.imag, lw=0)
    ax2.set_xlabel('U')
    ax2.set_xlabel('V')
    ax2.set_title('Imaginary')

    cbs.add_colorbar(mappable, ax1)

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


def plot_vlm_vs_vlm_rec(sel_ll, sel_mm, sel_vlm, vlm_rec, savefile=None, name='vlm'):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

    ax1.scatter(sel_mm, abs(sel_vlm), c='blue', marker='o', label='Input')
    ax1.scatter(sel_mm, abs(vlm_rec), c='orange', marker='+', label='Recovered')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-5, 1)
    ax1.set_xlim(0, max(sel_mm))
    ax1.set_xlabel('m')
    ax1.set_ylabel('abs(%s)' % name)
    ax1.legend()

    ax2.scatter(sel_ll, abs(sel_vlm), c='blue', marker='o', label='Input')
    ax2.scatter(sel_ll, abs(vlm_rec), c='orange', marker='+', label='Recovered')
    ax2.set_yscale('log')
    ax2.set_ylim(1e-5, 5)
    ax2.set_xlim(0, max(sel_ll))
    ax2.set_xlabel('l')
    ax2.set_ylabel('abs(%s)' % name)
    ax2.legend()

    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)


def plot_vlm(ll, mm, vlm_rec, savefile=None, name='vlm'):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

    # ax1.autoscale(axis='y')
    ax1.plot(mm, abs(vlm_rec), c='orange', marker='+', label='Recovered', ls='')
    ax1.set_yscale('log')
    ax1.set_xlim(0, max(mm))
    ax1.set_xlabel('m')
    ax1.set_ylabel('abs(%s)' % name)
    ax1.legend()

    # ax2.autoscale(axis='y')
    ax2.plot(ll, abs(vlm_rec), c='orange', marker='+', label='Recovered', ls='')
    ax2.set_yscale('log')
    ax2.set_xlim(0, max(ll))
    ax2.set_xlabel('l')
    ax2.set_ylabel('abs(%s)' % name)
    ax2.legend()

    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)


def plot_vlm_vs_vlm_rec_map(sel_ll, sel_mm, sel_vlm, vlm_rec, cov_error,
                            savefile=None, vmin=1e-6, vmax=1e2, name='vlm'):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))

    cbs = plotutils.ColorbarSetting(plotutils.ColorbarInnerPosition(location=2, height="80%", pad=1))

    lm_map = util.get_lm_map(abs(sel_vlm), sel_ll, sel_mm)
    lm_map_rec = util.get_lm_map(abs(vlm_rec), sel_ll, sel_mm)
    lm_map_cov_error = util.get_lm_map(abs(cov_error), sel_ll, sel_mm)

    extent = (min(sel_ll), max(sel_ll), min(sel_mm), max(sel_mm))

    im_mappable = ax1.imshow(abs(lm_map), norm=plotutils.LogNorm(), vmin=vmin,
                             vmax=vmax, extent=extent, aspect='auto')
    ax1.set_title('Input %s' % name)
    cbs.add_colorbar(im_mappable, ax1)

    im_mappable = ax2.imshow(abs(lm_map_rec), norm=plotutils.LogNorm(), vmin=vmin,
                             vmax=vmax, extent=extent, aspect='auto')
    ax2.set_title('Recovered %s' % name)
    cbs.add_colorbar(im_mappable, ax2)

    im_mappable = ax3.imshow(abs(lm_map - lm_map_rec), norm=plotutils.LogNorm(), vmin=vmin,
                             vmax=vmax, extent=extent, aspect='auto')
    ax3.set_title('Diff')
    cbs.add_colorbar(im_mappable, ax3)

    im_mappable = ax4.imshow(lm_map_cov_error, norm=plotutils.LogNorm(), vmin=vmin,
                             vmax=vmax, extent=extent, aspect='auto')
    ax4.set_title('Noise covariance matrix')
    cbs.add_colorbar(im_mappable, ax4)

    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xlabel('l')
        ax.set_ylabel('m')

    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)


def plot_vlm_rec_map(sel_ll, sel_mm, vlm_rec, cov_error, savefile=None, vmin=1e-6, vmax=1e2, name='vlm'):
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))

    cbs = plotutils.ColorbarSetting(plotutils.ColorbarInnerPosition(location=2, height="80%", pad=1))

    lm_map_rec = util.get_lm_map(abs(vlm_rec), sel_ll, sel_mm)
    lm_map_cov_error = util.get_lm_map(abs(cov_error), sel_ll, sel_mm)

    extent = (min(sel_ll), max(sel_ll), min(sel_mm), max(sel_mm))

    im_mappable = ax1.imshow(abs(lm_map_rec), norm=plotutils.LogNorm(), vmin=vmin,
                             vmax=vmax, extent=extent, aspect='auto')
    ax1.set_title('Input %s' % name)
    cbs.add_colorbar(im_mappable, ax1)

    im_mappable = ax2.imshow(abs(lm_map_cov_error), norm=plotutils.LogNorm(), vmin=vmin,
                             vmax=vmax, extent=extent, aspect='auto')
    ax2.set_title('Recovered %s' % name)
    cbs.add_colorbar(im_mappable, ax2)

    for ax in (ax1, ax2):
        ax.set_xlabel('l')
        ax.set_ylabel('m')

    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)


def plot_2d_power_spectra(ll, mm, alms, freqs, config, savefile=None, vmin=1e-14, vmax=1e-10, ft=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ps = util.get_2d_power_spectra(alms, ll, mm, freqs, ft=ft)

    cbs = plotutils.ColorbarSetting(plotutils.ColorbarOutterPosition())
    extent = (min(ll), max(ll), min(freqs), max(freqs))

    im_mappable = ax.imshow(np.array(ps), aspect='auto', norm=plotutils.LogNorm(),
                            vmin=vmin, vmax=vmax, extent=extent)
    cbs.add_colorbar(im_mappable, ax)
    ax.set_ylabel("Frequency")
    ax.set_xlabel('l')
    ax.set_title("Power spectra")

    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)

    return np.array(ps), ax


def plot_rec_power_sepctra(ll, mm, alm, savefile=None):
    ps = util.get_power_spectra(alm, ll, mm)

    fig = plt.figure()
    fig, ax1 = plt.subplots()
    ax1.plot(np.unique(ll), ps, label='Beam modulated power spectra')
    ax1.set_yscale('log')
    ax1.set_xlabel('l')
    ax1.set_ylabel('cl')

    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)


def plot_power_spectra(ll, mm, alm, alm_rec, config, savefile=None):
    el = np.unique(ll)

    if config.beam_type == 'gaussian':
        omega = np.pi * nputils.gaussian_fwhm_to_sigma(config.fwhm) ** 2.
    else:
        omega = config.fwhm ** 2

    pb_corr = 4 * np.pi / omega

    ps_rec = util.get_power_spectra(alm_rec, ll, mm) * pb_corr
    ps = util.get_power_spectra(alm, ll, mm) * pb_corr

    fig = plt.figure()
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    ax1.plot(np.arange(config.lmax + 1), config.cl, label='Input power spectra')
    ax1.plot(el, ps, label='Beam modulated input power spectra')
    ax1.plot(el, ps_rec, label='Recovered power spectra', marker='+')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('l')
    ax1.set_ylabel('cl')
    ax1.set_xlim(min(el), max(el))
    ax1.legend()

    diff = ps_rec - ps
    # print nputils.stat(diff)
    ax2.plot(el, diff, ls='', marker='+')
    ax2.set_xlabel('l')
    ax2.set_ylabel('diff')
    ax2.text(0.95, 0.05, 'std diff: %.3e\nmax diff: %.3e' % (diff.std(), diff.max()),
             ha='right', va='center', transform=ax2.transAxes)

    if savefile is not None:
        fig.tight_layout()
        fig.savefig(savefile)
        plt.close(fig)


def plot_mf_power_spectra(ll, mm, alms, freqs, config, savefile=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    l_sampled = np.arange(max(ll) + 1)[np.bincount(ll) > 0]
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


def plot_mf_power_spectr_diff(ll, mm, alms, alms_rec, freqs, savefile=None, vmin=1e-14, vmax=1e-10):
    alm_cube = np.array(alms)
    alm_rec_cube = np.array(alms_rec)
    ps_rec = []
    ps = []
    for i in range(alm_cube.shape[0]):
        ps.append(util.get_power_spectra(alm_cube[i], ll, mm))
        ps_rec.append(util.get_power_spectra(alm_rec_cube[i], ll, mm))

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(6, 12))
    cbs = plotutils.ColorbarSetting(plotutils.ColorbarOutterPosition())
    extent = (min(ll), max(ll), min(freqs), max(freqs))

    im_mappable = ax1.imshow(np.array(ps), aspect='auto', norm=plotutils.LogNorm(),
                             vmin=vmin, vmax=vmax, extent=extent)
    cbs.add_colorbar(im_mappable, ax1)
    ax1.set_ylabel("Frequency")
    ax1.set_xlabel('l')
    ax1.set_title("Original power spectra")

    im_mappable = ax2.imshow(np.array(ps_rec), aspect='auto', norm=plotutils.LogNorm(),
                             vmin=vmin, vmax=vmax, extent=extent)
    cbs.add_colorbar(im_mappable, ax2)
    ax2.set_ylabel("Frequency")
    ax2.set_xlabel('l')
    ax2.set_title("Recovered power spectra")

    im_mappable = ax3.imshow(np.array(ps_rec) - np.array(ps), aspect='auto',
                             extent=extent)
    cbs.add_colorbar(im_mappable, ax3)
    ax3.set_ylabel("Frequency")
    ax3.set_xlabel('l')
    ax3.set_title("Difference")

    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)

    return np.array(ps), np.array(ps_rec)


def plot_vlm_diff(sel_ll, sel_mm, sel_vlm, vlm_rec, cov_error, savefile=None, name='vlm'):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

    ax1.scatter(sel_mm, abs(vlm_rec.real - sel_vlm.real), c='green', marker='+', label='Input')
    ax1.scatter(sel_mm, cov_error.real, c='red', marker='+', label='cov error')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-5, 1e-1)
    ax1.set_xlim(0, max(sel_mm))
    ax1.set_xlabel('m')
    ax1.set_ylabel('abs(%s.real - %s_rec.real)' % (name, name))

    diff = vlm_rec.real - sel_vlm.real
    ax2.scatter(sel_ll, abs(diff), c='green', marker='+', label='Input')
    ax2.scatter(sel_ll, cov_error.real, c='red', marker='+', label='cov error')
    ax2.set_yscale('log')
    ax2.set_ylim(1e-5, 1e-1)
    ax2.set_xlim(0, max(sel_ll))
    ax2.set_xlabel('l')
    ax2.set_ylabel('abs(%s.real - %s_rec.real)' % (name, name))
    ax2.text(0.02, 0.95, 'std diff: %.3e\nmax diff: %.3e' % (diff.std(), abs(diff).max()),
             ha='left', va='center', transform=ax2.transAxes)

    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)


def plot_vis_simu_diff(ru, V, Vobs, Vrec, savefile=None):
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


def plot_vis_vs_vis_rec(ru, Vobs, Vrec, savefile=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(ru * (2 * np.pi), abs(Vobs), c='blue', marker='o', label='Input')
    ax.scatter(ru * (2 * np.pi), abs(Vrec), c='orange', marker='+', label='Recovered')
    ax.set_yscale('log')
    ax.set_ylabel('abs(vis)')
    ax.set_xlabel('2 pi u')
    ax.set_ylim(1e-2, 5)
    ax.legend()

    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)


def plot_vis_diff(ru, Vobs, Vrec, savefile=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ru * (2 * np.pi), abs(Vobs - Vrec), ls='', marker='+', c='green', label='Vobs - Vrec')
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

    thetas, phis = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))

    if config.beam_type == 'gaussian':
        print 'Using a gaussian beam with fwhm = %.3f rad' % fwhm
        beam = util.gaussian_beam(thetas, fwhm)

    elif config.beam_type == 'sinc2':
        print 'Using a sinc2 beam with fwhm = %.3f rad' % fwhm
        beam = util.sinc2_beam(thetas, fwhm)

    elif config.beam_type == 'tophat':
        print 'Using a tophat beam with width = %.3f rad' % fwhm
        beam = util.tophat_beam(thetas, fwhm)

    else:
        print 'Not using any beam'
        beam = np.ones_like(thetas)

    if config.add_fg is True:
        print 'Using simulated sky from fits file'
        fg_fits = pyfits.open(config.fg_file)
        fg_slice = slice(config.fg_freq_start, config.fg_freq_stop, config.fg_freq_step)
        # Extract slices and convert them to Jy/sr
        fg_maps_cart = fg_fits[0].data[fg_slice]
        if not len(fg_maps_cart) == len(config.freqs_mhz):
            raise Exception('Number of selected slices does not match the number of frequencies')

        fg_maps = [util.cartmap2healpix(fg_map_cart, config.fg_res, config.nside) * beam
                   for fg_map_cart in fg_maps_cart]
        fg_alms = [hp.map2alm(m, lmax) for m in fg_maps]
    else:
        print 'Simulating full sky with healpix synfast'
        np.random.seed(config.synfast_rnd_seed)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full_map, alm = hp.synfast(config.cl, config.nside, alm=True, lmax=lmax, verbose=False)

        full_map = full_map + abs(full_map.min())

        full_maps = [full_map * a for a in config.cl_freq]

        fg_maps = [m * beam for m in full_maps]
        fg_alms = [hp.map2alm(m, lmax) for m in fg_maps]

    if config.add_eor is True:
        print 'Using simulated EoR signal from fits file'
        eor_fits = pyfits.open(config.eor_file)
        eor_slice = slice(config.eor_freq_start, config.eor_freq_stop, config.eor_freq_step)
        eor_maps_cart = eor_fits[0].data[eor_slice]
        if not len(eor_maps_cart) == len(config.freqs_mhz):
            raise Exception('Number of selected slices does not match the number of frequencies')

        eor_maps = [util.cartmap2healpix(eor_map_cart, config.eor_res, config.nside) * beam
                    for eor_map_cart in eor_maps_cart]
        eor_alms = [hp.map2alm(m, lmax) for m in eor_maps]
    else:
        print 'No EoR signal added'
        eor_alms = [np.zeros_like(m) for m in fg_alms]

    alms = [m1 + m2 for m1, m2, in zip(fg_alms, eor_alms)]

    beam_lm = hp.map2alm(beam, lmax)

    return ll, mm, alms, fg_alms, eor_alms, beam_lm


def sample_input_alm(config, alms, ll, mm):
    ll_inp, mm_inp = util.get_lm(lmin=config.inp_lmin, lmax=config.inp_lmax,
                                 dl=config.inp_dl, dm=config.inp_dm,
                                 mmax=config.inp_mmax, mmin=config.inp_mmin)
    theta_max = config.out_theta_max

    if config.inp_mmax_strip:
        ll_inp, mm_inp = util.strip_mm(ll_inp, mm_inp, lambda l: np.sin(theta_max) * l)

    if config.inp_lm_even_only:
        idx = np.logical_not(util.is_odd(ll_inp + mm_inp)).astype(bool)
        ll_inp = ll_inp[idx]
        mm_inp = mm_inp[idx]

    idx = util.get_lm_selection_index(ll, mm, ll_inp, mm_inp)
    alms = [alm[idx] for alm in alms]

    return alms, ll_inp, mm_inp


def simulate_uv_cov(config):
    freqs_mhz = config.freqs_mhz
    lambs = const.c.value / (np.array(freqs_mhz) * 1e6)
    bmin = config.uv_rumin * min(lambs)
    bmax = config.uv_rumax * max(lambs)
    freq = const.c.value * 1e-6
    if config.uv_type == 'polar':
        print 'Using polar uv coverage'
        rb, uphis, uthetas = util.polar_uv(bmin, bmax, config.polar_nr, config.polar_nphi,
                                           rnd_w=config.polar_rnd_w,
                                           freqs_mhz=[freq], rnd_ru=config.polar_rnd_ru)

    elif config.uv_type == 'cartesian':
        print 'Using cartesian uv coverage'
        uu, vv, ww = util.cart_uv(bmin, bmax, config.cart_du * min(lambs), rnd_w=config.cart_rnd_w,
                                  freqs_mhz=[freq])
        rb, uphis, uthetas = util.cart2sph(uu, vv, ww)

    elif config.uv_type == 'lofar':
        print 'Using LOFAR uv coverage'
        uu, vv, ww = util.lofar_uv([freq], config.lofar_dec_deg,
                                   config.lofar_hal, config.lofar_har, bmin,
                                   bmax, config.lofar_timeres,
                                   include_conj=config.lofar_include_conj)
        rb, uphis, uthetas = util.cart2sph(uu, vv, ww)
    elif config.uv_type == 'gridded':
        print 'Using gridded visibilities configuration'
        uu, vv, weights = read_gridded_config(config.gridded_weights, config)
        ww = np.zeros_like(uu)
        ru, uphis, uthetas = util.cart2sph(uu, vv, ww)
        rb = np.array([ru * min(lambs)])
        uphis = np.array([uphis])
        uthetas = np.array([uthetas])
        config.noiserms = config.noiserms / np.sqrt(weights)
        config.weights = weights
    else:
        print 'Configuration value uv_type invalid'
        sh.usage(True)

    return rb[0], uphis[0], uthetas[0]


def get_out_lm_sampling(config):
    lmax = config.out_lmax
    mmax = config.out_mmax
    mmin = config.out_mmin
    lmin = config.out_lmin
    dl = config.out_dl
    theta_max = config.out_theta_max

    ll, mm = util.get_lm(lmin=lmin, lmax=lmax, dl=dl, mmax=mmax, mmin=mmin)

    if config.out_mmax_strip:
        ll, mm = util.strip_mm(ll, mm, lambda l: np.sin(theta_max) * l)

    if config.out_lm_even_only:
        idx = np.logical_not(util.is_odd(ll + mm)).astype(bool)
        ll = ll[idx]
        mm = mm[idx]

    return ll, mm


def interpolate_lm_odd(alm, ll, mm, config):
    prev = config.out_lm_even_only
    config.out_lm_even_only = False
    ll2, mm2 = get_out_lm_sampling(config)
    config.out_lm_even_only = prev

    alm2 = np.zeros_like(ll2, dtype=np.complex)

    for m in np.unique(mm):
        x2 = ll2[mm2 == m]
        yr = alm[mm == m].real
        yi = alm[mm == m].imag
        y2r = idct(dct(yr), n=len(x2)) / len(x2)
        y2i = idct(dct(yi), n=len(x2)) / len(x2)
        norm = np.sum(mm == m) / float(np.sum(mm2 == m))
        alm2[mm2 == m] = (y2r + 1j * y2i) * norm

    return alm2, ll2, mm2


def l_smoothing(alm, ll, mm):
    alm_smooth = np.zeros_like(alm, dtype=np.complex)

    for i, m in enumerate(np.unique(mm)):
        x = ll[mm == m]
        n = len(x)
        y_rec = alm[mm == m]
        tf_dct = dct(y_rec, norm='ortho')
        tf_dct[n / 2:] = 0
        y_rec2 = idct(tf_dct, norm='ortho')

        alm_smooth[mm == m] = y_rec2

    return alm_smooth


def l_sampling(ll, mm, dl, dl_start):
    el = np.unique(ll)
    ll2, mm2 = util.get_lm(ll.max(), lmin=ll.min() + dl_start, dl=int(dl), mmax=mm.max())
    mm2 = mm2[ll2 >= min(ll)]
    ll2 = ll2[ll2 >= min(ll)]
    mmax = np.zeros(ll.max() + 1)
    mmax[el] = np.array([mm[ll == l_].max() for l_ in el])
    ll2, mm2 = util.strip_mm(ll2, mm2, lambda l: mmax[l])
    idx = util.get_lm_selection_index(ll, mm, ll2, mm2)

    return ll2, mm2, idx


def compute_visibilities(alm, ll, mm, uphis, uthetas, i, trm):
    alm_r, alm_i = trm.split(alm)
    V = np.dot(alm_r, trm.T_r) + 1j * np.dot(alm_i, trm.T_i)

    return V


def alm_ml_inversion(ll, mm, Vobs, uphis, uthetas, i, trm, config):
    if config.use_dct:
        dct_blocks = []
        if isinstance(config.dct_dl, (list, np.ndarray)):
            dl = float(config.dct_dl[i])
        else:
            dl = float(config.dct_dl)

        print "Using DCT with DL=%s" % dl
        print "Building DCT Matrix ..."
        for sel_block in [trm.m0_l_even, trm.lm_even, trm.lm_even]:
            for m in np.unique(mm[sel_block]):
                n = len(ll[sel_block][mm[sel_block] == m])
                nk = int(np.ceil(n / dl))
                dct_blocks.append(util.get_dct2(n, nk))
        dct_real = block_diag(dct_blocks).tocsr()

        dct_blocks = []

        for sel_block in [trm.m0_l_odd, trm.lm_odd, trm.lm_odd]:
            for m in np.unique(mm[sel_block]):
                n = len(ll[sel_block][mm[sel_block] == m])
                nk = int(np.ceil(n / dl))
                dct_blocks.append(util.get_dct2(n, nk))

        dct_imag = block_diag(dct_blocks).tocsr()

        print "Computing dot products of T and DCT ..."
        t = time.time()
        if config.use_psparse:
            X_r = pmultiply(dct_real.T, np.asfortranarray(trm.T_r)).T
            X_i = pmultiply(dct_imag.T, np.asfortranarray(trm.T_i)).T
        else:
            X_r = (dct_real.T.dot(trm.T_r)).T
            X_i = (dct_imag.T.dot(trm.T_i)).T
        print "Done in %.2f s" % (time.time() - t)

    else:
        X_r = trm.T_r.T
        X_i = trm.T_i.T

    print '\nSize of transformation matrix: X_r: %s X_i: %s' % (X_r.shape, X_i.shape)

    t = time.time()

    if isinstance(config.noiserms, np.ndarray):
        C_Dinv = diags(1 / (config.noiserms ** 2))
    else:
        C_Dinv = diags([1 / (config.noiserms ** 2)] * len(Vobs))

    print '\nComputing LHS and RHS matrix ...',
    X_r_dot_C_Dinv = C_Dinv.T.dot(X_r).T
    lhs_r = np.dot(X_r_dot_C_Dinv, X_r) + np.eye(X_r.shape[1]) * config.reg_lambda
    rhs_r = np.dot(X_r_dot_C_Dinv, Vobs.real)

    X_i_dot_C_Dinv = C_Dinv.T.dot(X_i).T
    lhs_i = np.dot(X_i_dot_C_Dinv, X_i) + np.eye(X_i.shape[1]) * config.reg_lambda
    rhs_i = np.dot(X_i_dot_C_Dinv, Vobs.imag)
    print "Done in %.2f s" % (time.time() - t)

    print "Building covariance matrix ...",
    t = time.time()

    if config.reg_lambda > 0:
        lhs_r_err_1 = np.linalg.inv(lhs_r)
        lhs_r_err_2 = np.dot(X_r_dot_C_Dinv, X_r)
        cov_error_r_tild = np.sqrt(np.diag(np.dot(np.dot(lhs_r_err_2, lhs_r_err_1), lhs_r_err_1)))

        lhs_i_err_1 = np.linalg.inv(lhs_i)
        lhs_i_err_2 = np.dot(X_i_dot_C_Dinv, X_i)
        cov_error_i_tild = np.sqrt(np.diag(np.dot(np.dot(lhs_i_err_2, lhs_i_err_1), lhs_i_err_1)))
    else:
        cov_error_r_tild = np.sqrt(np.diag(np.linalg.inv(lhs_r)))
        cov_error_i_tild = np.sqrt(np.diag(np.linalg.inv(lhs_i)))

    if config.use_dct:
        cov_error_r = []
        j = 0
        for sel_block in [trm.m0_l_even, trm.lm_even, trm.lm_even]:
            for m in np.unique(mm[sel_block]):
                n = len(ll[sel_block][mm[sel_block] == m])
                nk = int(np.ceil(n / dl))
                dct_C_Dinv = np.diag(np.repeat([1 / cov_error_r_tild[j:j + nk] ** 2], np.ceil(n / float(nk)))[:n])
                dct_C_Dinv *= (n / float(nk)) ** 2
                dct_mat = util.get_dct2(n, n)
                dct_lhs = np.dot(dct_C_Dinv.T.dot(dct_mat.T).T, dct_mat.T)
                cov_error_r.extend(np.sqrt(np.linalg.inv(dct_lhs).diagonal()))
                j += nk

        cov_error_i = []
        j = 0
        for sel_block in [trm.m0_l_odd, trm.lm_odd, trm.lm_odd]:
            for m in np.unique(mm[sel_block]):
                n = len(ll[sel_block][mm[sel_block] == m])
                nk = int(np.ceil(n / dl))
                dct_C_Dinv = np.diag(np.repeat([1 / cov_error_i_tild[j:j + nk] ** 2], np.ceil(n / float(nk)))[:n])
                dct_C_Dinv *= (n / float(nk)) ** 2
                dct_mat = util.get_dct2(n, n)
                dct_lhs = np.dot(dct_C_Dinv.T.dot(dct_mat.T).T, dct_mat.T)
                cov_error_i.extend(np.sqrt(np.linalg.inv(dct_lhs).diagonal()))
                j += nk

        cov_error_r = np.array(cov_error_r)
        cov_error_i = np.array(cov_error_i)
    else:
        cov_error_r = cov_error_r_tild
        cov_error_i = cov_error_i_tild

    cov_error = np.abs(trm.recombine(cov_error_r, cov_error_i))

    print "Done in %.2f s" % (time.time() - t)

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

    if config.use_dct:
        alm_rec_r = (dct_real.dot(alm_rec_r.T)).T  # np.dot(alm_rec_r, dct_real.T)
        alm_rec_i = (dct_imag.dot(alm_rec_i.T)).T  # np.dot(alm_rec_i, dct_imag.T)

    alm_rec = trm.recombine(alm_rec_r, alm_rec_i)
    Vrec = np.dot(alm_rec_r, trm.T_r) + 1j * np.dot(alm_rec_i, trm.T_i)

    if config.compute_alm_noise:
        print '\nComputing alm noise ...',
        start = time.time()
        rhs_noise_r = np.dot(X_r_dot_C_Dinv, config.noiserms * np.random.randn(len(Vobs)))
        rhs_noise_i = np.dot(X_i_dot_C_Dinv, config.noiserms * np.random.randn(len(Vobs)))

        alm_rec_noise_r, info = cg(lhs_r, rhs_noise_r, tol=config.cg_tol, maxiter=config.cg_maxiter)
        alm_rec_noise_i, info = cg(lhs_i, rhs_noise_i, tol=config.cg_tol, maxiter=config.cg_maxiter)

        if config.use_dct:
            alm_rec_noise_r = (dct_real.dot(alm_rec_noise_r.T)).T  # np.dot(alm_rec_r, dct_real.T)
            alm_rec_noise_i = (dct_imag.dot(alm_rec_noise_i.T)).T  # np.dot(alm_rec_i, dct_imag.T)
        alm_rec_noise = trm.recombine(alm_rec_noise_r, alm_rec_noise_i)
        print 'Done in %.2f s.' % (time.time() - start)
    else:
        alm_rec_noise = np.zeros_like(alm_rec)

    return alm_rec, alm_rec_noise, Vrec, cov_error


def alm_post_processing(alm, ll, mm, config):
    ''' Post processing + conversion from Jy/sr to K'''
    if config.do_lm_interp and config.out_lm_even_only:
        alm, ll, mm = interpolate_lm_odd(alm, ll, mm, config)
    if config.do_l_smoothing and not config.out_lm_even_only:
        alm = l_smoothing(alm, ll, mm)
    if config.do_l_sampling:
        ll, mm, idx = l_sampling(ll, mm, config.l_sampling_dl, config.l_sampling_dl_start)
        alm = alm[idx]

    return alm, ll, mm


def get_config(dirname):
    return imp.load_source('config', os.path.join(dirname, 'config.py'))


def save_data(filename, data, columns_name):
    df = pd.DataFrame(dict(zip(columns_name, data)))
    df.to_csv(filename)


def check_file_compressed(filename, extensions=['.gz', '.xz']):
    for ext in [''] + extensions:
        if os.path.exists(filename + ext):
            return filename + ext


def load_data(filename, compelx_columns):
    def conv_fct(data):
        try:
            return np.complex(data)
        except:
            return np.nan
    conv = dict(zip(compelx_columns, [conv_fct] * len(compelx_columns)))
    return pd.read_csv(check_file_compressed(filename), converters=conv)


def save_alm_simu_eor(dirname, ll, mm, alm, alm_fg, alm_eor):
    columns = ['ll', 'mm', 'alm', 'alm_fg', 'alm_eor']
    filename = os.path.join(dirname, 'alm_simu_eor.dat')
    save_data(filename, [ll, mm, alm, alm_fg, alm_eor], columns)


def save_alm_simu(dirname, ll, mm, alm):
    columns = ['ll', 'mm', 'alm']
    filename = os.path.join(dirname, 'alm_simu.dat')
    save_data(filename, [ll, mm, alm], columns)


def save_alm_rec(dirname, ll, mm, alm_rec, alm_rec_noise, cov_error):
    columns = ['ll', 'mm', 'alm_rec', 'alm_rec_noise', 'cov_error']
    filename = os.path.join(dirname, 'alm_rec.dat')
    save_data(filename, [ll, mm, alm_rec, alm_rec_noise, cov_error], columns)


def load_alm_simu_eor(dirname):
    filename = os.path.join(dirname, 'alm_simu_eor.dat')
    df = load_data(filename, ['alm', 'alm_fg', 'alm_eor'])

    return df.ll, df.mm, df.alm, df.alm_fg, df.alm_eor


def load_alm_simu(dirname):
    filename = os.path.join(dirname, 'alm_simu_eor.dat')
    df = load_data(filename, ['alm'])

    return df.ll, df.mm, df.alm


def load_alm_rec(dirname):
    filename = os.path.join(dirname, 'alm_rec.dat')
    df = load_data(filename, ['alm_rec', 'alm_rec_noise', 'cov_error'])

    return df.ll, df.mm, df.alm_rec, df.alm_rec_noise, df.cov_error


def save_alm(dirname, ll, mm, alm, alm_fg, alm_eor, alm_rec, alm_rec_noise, cov_error):
    filename = os.path.join(dirname, 'alm.dat')
    print "Saving alm result to:", filename

    np.savetxt(filename, np.array([ll, mm, alm.real, alm.imag, alm_fg.real, alm_fg.imag,
                                   alm_eor.real, alm_eor.imag, alm_rec.real, alm_rec.imag,
                                   alm_rec_noise.real, alm_rec_noise.imag, cov_error.real,
                                   cov_error.imag]).T)


def load_alm(dirname):
    filename = os.path.join(dirname, 'alm.dat')
    # print "Loading alm result from:", filename

    # a = np.loadtxt(filename)
    a = pd.read_csv(check_file_compressed(filename), delimiter=" ", header=None).values

    if a.shape[1] == 8:
        ll, mm, alm_real, alm_imag, alm_rec_real, alm_rec_imag, \
            cov_error_real, cov_error_imag = a.T

        return ll, mm, alm_real + 1j * alm_imag, alm_rec_real + 1j * alm_rec_imag, \
            cov_error_real + 1j * cov_error_imag
    elif a.shape[1] == 12:
        ll, mm, alm_real, alm_imag, alm_fg_real, alm_fg_imag, alm_eor_real, alm_eor_imag, \
            alm_rec_real, alm_rec_imag, cov_error_real, cov_error_imag = a.T

        return ll, mm, alm_real + 1j * alm_imag, alm_fg_real + 1j * alm_fg_imag, \
            alm_eor_real + 1j * alm_eor_imag, alm_rec_real + 1j * alm_rec_imag, \
            cov_error_real + 1j * cov_error_imag
    elif a.shape[1] == 14:
        ll, mm, alm_real, alm_imag, alm_fg_real, alm_fg_imag, alm_eor_real, alm_eor_imag, \
            alm_rec_real, alm_rec_imag, alm_rec_noise_real, alm_rec_noise_imag, \
            cov_error_real, cov_error_imag = a.T

        return ll, mm, alm_real + 1j * alm_imag, alm_fg_real + 1j * alm_fg_imag, \
            alm_eor_real + 1j * alm_eor_imag, alm_rec_real + 1j * alm_rec_imag, \
            alm_rec_noise_real + 1j * alm_rec_noise_imag, cov_error_real + 1j * cov_error_imag
    else:
        print "Format not understood"


def save_visibilities_simu(dirname, ru, uphis, uthetas, V):
    columns = ['ru', 'uphis', 'uthetas', 'V']
    filename = os.path.join(dirname, 'visibilities_simu.dat')
    save_data(filename, [ru, uphis, uthetas, V], columns)


def save_visibilities_rec(dirname, ru, uphis, uthetas, Vobs, Vrec):
    columns = ['ru', 'uphis', 'uthetas', 'Vobs', 'Vrec']
    filename = os.path.join(dirname, 'visibilities_rec.dat')
    save_data(filename, [ru, uphis, uthetas, Vobs, Vrec], columns)


def save_visibilities(dirname, ru, uphis, uthetas, V, Vobs, Vrec):
    filename = os.path.join(dirname, 'visibilities.dat')
    print "Saving visibilities result to:", filename

    np.savetxt(filename, np.array([ru, uphis, uthetas, V.real, V.imag,
                                   Vobs.real, Vobs.imag, Vrec.real, Vrec.imag]).T)


def load_visibilities_rec(dirname):
    filename = os.path.join(dirname, 'visibilities_rec.dat')
    df = load_data(filename, ['Vobs', 'Vrec'])

    return df.ru, df.uphis, df.uthetas, df.Vobs, df.Vrec


def load_visibilities_simu(dirname):
    filename = os.path.join(dirname, 'visibilities_simu.dat')
    df = load_data(filename, ['V'])

    return df.ru, df.uphis, df.uthetas, df.V


def load_visibilities(dirname):
    filename = os.path.join(dirname, 'visibilities.dat')
    # print "Loading visibilities result from:", filename

    # a = np.loadtxt(filename)
    a = pd.read_csv(filename, delimiter=" ", header=None).values

    if a.shape[1] == 7:
        ru, uphis, uthetas, Vobs_real, Vobs_imag, Vrec_real, Vrec_imag = a.T

        return ru, uphis, uthetas, Vobs_real + 1j * Vobs_imag, Vrec_real + 1j * Vrec_imag
    elif a.shape[1] == 9:
        ru, uphis, uthetas, V_real, V_imag, Vobs_real, Vobs_imag, Vrec_real, Vrec_imag = a.T

        return ru, uphis, uthetas, V_real + 1j * V_imag, Vobs_real + 1j * Vobs_imag, Vrec_real + 1j * Vrec_imag
    else:
        print "Format not understood"


def load_results(dirname):
    if not os.path.exists(dirname):
        print "Path does not exists:", dirname
        return

    print "loading result from %s ..." % dirname
    key_fct = lambda a: int(a.split('_')[-1])
    freq_res = []
    for freq_dir in sorted(glob.glob(os.path.join(dirname, 'freq_*')), key=key_fct):
        ll, mm, alm, alm_rec, cov_error = load_alm(freq_dir)
        ru, uphis, uthetas, Vobs, Vrec = load_visibilities(freq_dir)
        freq_res.append([alm, alm_rec, cov_error, ru, uphis, uthetas, Vobs, Vrec])

    alm, alm_rec, cov_error, ru, uphis, uthetas, Vobs, Vrec = zip(*freq_res)

    return ll.astype(int), mm.astype(int), alm, alm_rec, cov_error, ru, uphis, uthetas, Vobs, Vrec


def load_results_v2(dirname):
    if not os.path.exists(dirname):
        print "Path does not exists:", dirname
        return

    print "loading result from %s ..." % dirname
    key_fct = lambda a: int(a.split('_')[-1])
    freq_res = []
    for freq_dir in sorted(glob.glob(os.path.join(dirname, 'freq_*')), key=key_fct):
        ll, mm, alm, alm_fg, alm_eor, alm_rec, cov_error = load_alm(freq_dir)
        ru, uphis, uthetas, V, Vobs, Vrec = load_visibilities(freq_dir)
        freq_res.append([alm, alm_fg, alm_eor, alm_rec, cov_error, ru, uphis, uthetas, V, Vobs, Vrec])

    alm, alm_fg, alm_eor, alm_rec, cov_error, ru, uphis, uthetas, V, Vobs, Vrec = zip(*freq_res)

    return ll.astype(int), mm.astype(int), alm, alm_fg, alm_eor, alm_rec, cov_error, ru, uphis, uthetas, V, Vobs, Vrec


def load_results_v3(dirname):
    if not os.path.exists(dirname):
        print "Path does not exists:", dirname
        return

    print "loading result from %s ..." % dirname
    key_fct = lambda a: int(a.split('_')[-1])
    freq_res = []
    for freq_dir in sorted(glob.glob(os.path.join(dirname, 'freq_*')), key=key_fct):
        ll, mm, alm, alm_fg, alm_eor, alm_rec, alm_rec_noise, cov_error = load_alm(freq_dir)
        ru, uphis, uthetas, V, Vobs, Vrec = load_visibilities(freq_dir)
        freq_res.append([alm, alm_fg, alm_eor, alm_rec, alm_rec_noise, cov_error, ru, uphis, uthetas, V, Vobs, Vrec])

    alm, alm_fg, alm_eor, alm_rec, alm_rec_noise, cov_error, ru, uphis, uthetas, V, Vobs, Vrec = zip(*freq_res)

    return ll.astype(int), mm.astype(int), alm, alm_fg, alm_eor, alm_rec, alm_rec_noise, cov_error, \
        ru, uphis, uthetas, V, Vobs, Vrec


def read_gridded_config(weighting_file, config):
    f_w = pyfits.open(weighting_file)
    du = f_w[0].header['CDELT1']
    dv = f_w[0].header['CDELT2']
    nu = f_w[0].header['NAXIS1']
    nv = f_w[0].header['NAXIS2']

    u = du * np.arange(-nu / 2, nu / 2)
    v = dv * np.arange(-nv / 2, nv / 2)

    uu, vv = np.meshgrid(u, v)

    weights = f_w[0].data[0][0]

    idx_u, idx_v = np.nonzero(weights)

    weights = weights[idx_u, idx_v].flatten()
    uu = uu[idx_u, idx_v].flatten()
    vv = vv[idx_u, idx_v].flatten()

    ru = np.sqrt(uu ** 2 + vv ** 2)

    idx = (ru >= config.uv_rumin) & (ru <= config.uv_rumax)

    weights = weights[idx]
    # weights = np.ones_like(weights) * np.mean(weights)  # Uniform weights
    uu = uu[idx]
    vv = vv[idx]

    return uu, vv, weights


def get_gridded_visibilities(config, V, uu, vv):
    f_w = pyfits.open(config.gridded_weights)
    du = f_w[0].header['CDELT1']
    n = f_w[0].header['NAXIS1']

    u = du * np.arange(-n / 2, n / 2)
    v = du * np.arange(-n / 2, n / 2)

    n = len(u)

    g_uu, g_vv = np.meshgrid(u, v)

    x = (np.round(g_uu, decimals=4) + 1e-6 * np.round(g_vv, decimals=4)).flatten()
    y = np.round(uu, decimals=4) + 1e-6 * np.round(vv, decimals=4)

    idx = np.where(np.in1d(x, y, assume_unique=False))[0]

    idx_x = np.argsort(x)
    sorted_x = x[idx_x]
    idx_y = np.searchsorted(sorted_x, y)
    idx = idx_x[idx_y]

    g_Vobs = np.zeros_like(g_uu, dtype=np.complex).flatten()
    g_Vobs[idx] = V
    g_Vobs = g_Vobs.reshape(*g_uu.shape)

    return g_Vobs


def read_gridded_visbilities(filename, config):
    dirname = os.path.dirname(filename)
    basename = '_'.join(os.path.basename(filename).split('_')[:-1])
    if config.stokes == 'I':
        print 'Reading %s Stokes I...' % basename
        g_real = os.path.join(dirname, basename + '_GR.fits')
        g_imag = os.path.join(dirname, basename + '_GI.fits')
    else:
        print 'Reading %s Stokes %s...' % (basename, config.stokes)
        g_real = os.path.join(dirname, basename + '_GR%s.fits' % config.stokes)
        g_imag = os.path.join(dirname, basename + '_GI%s.fits' % config.stokes)
    weighting = os.path.join(dirname, basename + '_W.fits')

    assert os.path.exists(g_real) and os.path.exists(g_imag)

    f_g_real = pyfits.open(g_real)
    f_g_imag = pyfits.open(g_imag)
    freq = f_g_real[0].header['CRVAL3']
    du = f_g_real[0].header['CDELT1']
    dv = f_g_real[0].header['CDELT2']
    nu = f_g_real[0].header['NAXIS1']
    nv = f_g_real[0].header['NAXIS2']
    delnu = f_g_real[0].header['CDELT3']

    u = du * np.arange(-nu / 2, nu / 2)
    v = dv * np.arange(-nv / 2, nv / 2)

    uu, vv = np.meshgrid(u, v)

    vis = f_g_real[0].data[0][0] + 1j * f_g_imag[0].data[0][0]

    idx_u, idx_v = np.nonzero(vis)

    vis = vis[idx_u, idx_v].flatten()
    uu = uu[idx_u, idx_v].flatten()
    vv = vv[idx_u, idx_v].flatten()

    ru = np.sqrt(uu ** 2 + vv ** 2)

    idx = (ru >= config.uv_rumin) & (ru <= config.uv_rumax)

    vis = vis[idx]
    uu = uu[idx]
    vv = vv[idx]

    if os.path.exists(weighting):
        w = pyfits.open(weighting)[0].data[0][0]
        w = w[idx_u, idx_v].flatten()[idx]
        noiserms = (config.SEFD / np.sqrt(2. * delnu * config.Int_time)) / np.sqrt(w)

    return freq, uu, vv, vis, noiserms


def write_fits_gridded_visibilities(file, data, du, freq, dfreq):
    nx, ny = data.shape
    wcs = pywcs.WCS(naxis=4)

    # crpix is with origin 1:
    wcs.wcs.crpix = [nx / 2 + 1, ny / 2 + 1, 1, 1]
    wcs.wcs.crval = [0, 0, freq, 1]
    wcs.wcs.cdelt = [du, du, dfreq, 1]
    wcs.wcs.ctype = ['U---WAV', 'V---WAV', 'FREQ', 'STOCKES']

    hdu = pyfits.PrimaryHDU(data[np.newaxis, np.newaxis])
    hdu.data.flags.writeable = True
    header = wcs.to_header()
    hdu.header.update(header)

    hdulist = pyfits.HDUList([hdu])
    hdulist.writeto(file, clobber=True)


def write_gridded_visibilities(dirname, basename, V, config, freq, dfreq):
    g_real = os.path.join(dirname, basename + '_GR.fits')
    g_imag = os.path.join(dirname, basename + '_GI.fits')

    du = config.cart_du
    n = np.ceil(2 * config.uv_rumax / du)

    u = du * np.arange(-n / 2, n / 2)
    v = du * np.arange(-n / 2, n / 2)

    n = len(u)

    g_uu, g_vv = np.meshgrid(u, v)
    g_uu = g_uu.flatten()
    g_vv = g_vv.flatten()

    g_ru = np.sqrt(g_uu ** 2 + g_vv ** 2)

    idx = (g_ru > config.uv_rumin) & (g_ru < config.uv_rumax)
    sort_idx = nputils.sort_index(g_ru[idx])

    flat_data = np.zeros_like(g_uu, dtype=np.complex)
    flat_nz_data = flat_data[idx]
    flat_nz_data[sort_idx] = V
    flat_data[idx] = flat_nz_data

    data = flat_data.reshape(n, n)

    write_fits_gridded_visibilities(g_real, data.real, du, freq, dfreq)
    write_fits_gridded_visibilities(g_imag, data.imag, du, freq, dfreq)


def do_inversion(config, result_dir):
    nfreqs = len(config.freqs_mhz)

    assert len(config.freqs_mhz) == len(config.cl_freq)

    if config.use_dct and isinstance(config.dct_dl, (list, np.ndarray)):
        assert len(config.dct_dl) == nfreqs, "Lenght of dct_dl should be the same as the number of frequencies"

    full_ll, full_mm, full_alms, fg_alms, eor_alms, beam_alm = simulate_sky(config)
    plot_sky_cart(full_alms[0], full_ll, full_mm, config.nside, theta_max=config.fwhm,
                  title='Input sky, beam=%.1f deg, lmax=%s' % (np.degrees(config.fwhm), config.lmax),
                  savefile=os.path.join(result_dir, 'input_sky.pdf'))

    inp_alms, inp_ll, inp_mm = sample_input_alm(config, full_alms + fg_alms + eor_alms + [beam_alm], full_ll, full_mm)
    fg_alms = inp_alms[nfreqs:2 * nfreqs]
    eor_alms = inp_alms[2 * nfreqs:3 * nfreqs]
    beam_alm = inp_alms[-1]

    rb, uphis, uthetas = simulate_uv_cov(config)

    sel_ll, sel_mm = get_out_lm_sampling(config)

    # plotting the sampling
    # plot_sampling(inp_ll, inp_mm, sel_ll, sel_mm, os.path.join(result_dir, 'lm_sampling.pdf'))

    print '\nBuilding the global YLM matrix...'
    global_inp_ylm = util.SplittedYlmMatrix(inp_ll, inp_mm, uphis, uthetas, rb,
                                            config.cache_dir, keep_in_mem=config.keep_in_mem)

    if config.out_dl != config.inp_dl or config.out_dm != config.inp_dm \
            or config.out_mmax != config.inp_mmax or config.out_mmax_strip != config.inp_mmax_strip:
        global_sel_ylm = util.SplittedYlmMatrix(sel_ll, sel_mm, uphis, uthetas, rb,
                                                config.cache_dir, keep_in_mem=config.keep_in_mem)
    else:
        global_sel_ylm = global_inp_ylm

    alms_rec = []

    for i, freq in enumerate(config.freqs_mhz):
        sel_ll, sel_mm = get_out_lm_sampling(config)

        plot_pool = multiprocessing.Pool(processes=1)
        print "\nProcessing frequency %s MHz" % freq

        lamb = const.c.value / (float(freq) * 1e6)
        bmin = np.floor(lamb * config.uv_rumin)
        bmax = np.ceil(lamb * config.uv_rumax)

        result_freq_dir = os.path.join(result_dir, 'freq_%s' % i)
        os.mkdir(result_freq_dir)

        t = time.time()
        print "Building transformation matrix..."
        inp_ylm = global_inp_ylm.get_chunk(bmin, bmax)
        trm = util.get_alm2vis_matrix(inp_ll, inp_mm, inp_ylm, lamb, order='F')
        print "Done in %.2f s" % (time.time() - t)

        uthetas, uphis, ru = inp_ylm[0].thetas, inp_ylm[0].phis, inp_ylm[0].rb / lamb
        if isinstance(config.noiserms, np.ndarray):
            config.noiserms = config.noiserms[inp_ylm[0].sort_idx_cols]
            config.weights = config.weights[inp_ylm[0].sort_idx_cols]

        uu, vv, ww = util.sph2cart(uthetas, uphis, ru)

        title = 'Type: %s, Nvis: %s, Umin: %s, Umax: %s' % (config.uv_type, len(uu),
                                                            config.uv_rumin, config.uv_rumax)
        plot_uv_cov(uu, vv, ww, config, title, os.path.join(result_freq_dir, 'uv_cov.pdf'))

        alm = inp_alms[i]

        # computing the visibilities in Jansky
        jy2k = ((1e-26 * lamb ** 2) / (2 * const.k_B.value))
        print "\nBuilding visibilities..."
        V = compute_visibilities(alm / jy2k, inp_ll, inp_mm, uphis, uthetas, i, trm)

        if isinstance(config.noiserms, np.ndarray):
            print "Noise in visibility (mean): %.3f Jy" % np.mean(config.noiserms)
        else:
            print "Noise in visibility: %.3f Jy" % config.noiserms

        np.random.seed(None)
        Vobs = V + config.noiserms * np.random.randn(len(V)) + 1j * config.noiserms * np.random.randn(len(V))

        # plotting the visibilities
        plot_pool.apply_async(plot_visibilities, (uu, vv, ww, V, os.path.join(result_freq_dir, 'vis_from_vlm.pdf')))

        if config.uv_type == 'gridded':
            if config.uv_type == 'gridded':
                print "Number total of visibilities:", np.sum(config.weights)
            g_Vobs = get_gridded_visibilities(config, Vobs, uu, vv) * jy2k
            g_V = get_gridded_visibilities(config, V, uu, vv) * jy2k

            fig, ax = plt.subplots()
            cbs = plotutils.ColorbarSetting(plotutils.ColorbarOutterPosition())
            extent = np.array([-5, 5, -5, 5])

            dmap_obs = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g_Vobs)))
            im_mappable = ax.imshow(dmap_obs.real, extent=extent)
            cbs.add_colorbar(im_mappable, ax)
            # ax1.set_xlabel('DEC (deg)')
            ax.set_ylabel('RA (deg)')
            ax.set_title('FFT(Vobs) Stokes I')
            fig.savefig(os.path.join(result_dir, 'dirty_map.pdf'))
            plt.close(fig)

            fig, ax = plt.subplots()
            dmap = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g_V)))
            im_mappable = ax.imshow(dmap.real - dmap_obs.real, extent=extent)
            cbs.add_colorbar(im_mappable, ax)
            # ax1.set_xlabel('DEC (deg)')
            ax.set_ylabel('RA (deg)')
            ax.set_title('FFT(Vobs) Stokes I')
            fig.savefig(os.path.join(result_dir, 'diff_dirty_map.pdf'))
            plt.close(fig)

        idx = util.get_lm_selection_index(inp_ll, inp_mm, sel_ll, sel_mm)

        sel_alm = alm[idx]
        sel_vlm = util.alm2vlm(sel_alm, sel_ll)

        if global_sel_ylm != global_inp_ylm:
            t = time.time()
            print "Building transformation matrix...",
            sel_ylm = global_sel_ylm.get_chunk(bmin, bmax)
            trm = util.get_alm2vis_matrix(sel_ll, sel_mm, inp_ylm, lamb, order='F')
            print "Done in %.2f s" % (time.time() - t)

            uthetas, uphis, ru = sel_ylm[0].thetas, sel_ylm[0].phis, sel_ylm[0].rb / lamb

        alm_rec, alm_rec_noise, Vrec, cov_error = alm_ml_inversion(sel_ll, sel_mm, Vobs, uphis, uthetas,
                                                                   i, trm, config)

        # Convert back to Kelvin
        alm_rec = alm_rec * jy2k
        alm_rec_noise = alm_rec_noise * jy2k
        cov_error = cov_error * jy2k

        print "Post processing..."
        alm_rec, _, _ = alm_post_processing(alm_rec, sel_ll, sel_mm, config)
        alm_rec_noise, _, _ = alm_post_processing(alm_rec_noise, sel_ll, sel_mm, config)
        cov_error, _, _ = alm_post_processing(cov_error, sel_ll, sel_mm, config)
        sel_alm, _, _ = alm_post_processing(sel_alm, sel_ll, sel_mm, config)
        sel_fg, _, _ = alm_post_processing(fg_alms[i][idx], sel_ll, sel_mm, config)
        sel_eor, sel_ll, sel_mm = alm_post_processing(eor_alms[i][idx], sel_ll, sel_mm, config)

        alms_rec.append(alm_rec)

        save_alm(result_freq_dir, sel_ll, sel_mm, sel_alm, sel_fg,
                 sel_eor, alm_rec, alm_rec_noise, cov_error)
        save_visibilities(result_freq_dir, ru, uphis, uthetas, V, Vobs, Vrec)

        sel_vlm = util.alm2vlm(sel_alm, sel_ll)
        vlm_rec = util.alm2vlm(alm_rec, sel_ll)
        vlm_rec_noise = util.alm2vlm(alm_rec_noise, sel_ll)

        print "Plotting result..."
        # plot vlm vs vlm_rec
        plot_pool.apply_async(plot_vlm_vs_vlm_rec, (sel_ll, sel_mm, sel_vlm, vlm_rec,
                                                    os.path.join(result_freq_dir, 'vlm_vs_vlm_rec.pdf')))

        # plot vlm vs vlm_rec in a map
        plot_pool.apply_async(plot_vlm_vs_vlm_rec_map, (sel_ll, sel_mm, sel_vlm, vlm_rec, 4 * np.pi * cov_error,
                                                        os.path.join(result_freq_dir, 'lm_maps_imag.pdf')))

        # plot power spectra
        # plot_power_spectra(sel_ll, sel_mm, sel_alm, alm_rec, config,
        #                    os.path.join(result_freq_dir, 'angular_power_spectra.pdf'))

        # plot vlm diff
        plot_pool.apply_async(plot_vlm_diff, (sel_ll, sel_mm, sel_vlm, vlm_rec, 4 * np.pi * cov_error,
                                              os.path.join(result_freq_dir, 'vlm_minus_vlm_rec.pdf')))

        plot_pool.apply_async(plot_vlm_diff, (sel_ll, sel_mm, np.zeros_like(vlm_rec), vlm_rec_noise,
                                              4 * np.pi * cov_error,
                                              os.path.join(result_freq_dir, 'vlm_minus_vlm_noise.pdf')))

        # plot visibilities diff
        plot_pool.apply_async(plot_vis_simu_diff, (ru, V, Vobs, Vrec, os.path.join(result_freq_dir,
                                                                                   'vis_minus_vis_rec.pdf')))

        # plot output sky
        plot_pool.apply_async(plot_sky_cart_diff, (sel_alm, alm_rec, sel_ll, sel_mm, sel_ll, sel_mm, config.nside),
                              dict(theta_max=config.fwhm, savefile=os.path.join(result_freq_dir, 'output_sky.pdf')))

        # write_gridded_visibilities(result_freq_dir, 'gridded_vis', V, config, freq, 1)

        t = time.time()
        print "Waiting for plotting to finish...",
        plot_pool.close()
        plot_pool.join()
        print "Done in %.2f s" % (time.time() - t)

    global_sel_ylm.close()
    global_inp_ylm.close()

    if len(alms_rec) > 1:
        plot_mf_power_spectra(sel_ll, sel_mm, alms_rec, config.freqs_mhz, config,
                              os.path.join(result_dir, 'mf_power_spectra.pdf'))

    print '\nAll done!'


def do_inversion_gridded(config, result_dir):
    ll, mm = get_out_lm_sampling(config)

    print '\nBuilding the global YLM matrix...'
    freq, uu, vv, Vobs, noiserms = read_gridded_visbilities(config.gridded_fits[0], config)
    ru, uphis, uthetas = util.cart2sph(uu, vv, np.zeros_like(uu))

    global_ylm = util.SplittedYlmMatrix(ll, mm, uphis, uthetas, ru,
                                        config.cache_dir, keep_in_mem=config.keep_in_mem)

    uthetas, uphis, ru = global_ylm[0].thetas, global_ylm[0].phis, global_ylm[0].rb
    uu, vv, ww = util.sph2cart(uthetas, uphis, ru)

    alms_rec = []
    freqs = []

    for i, file in enumerate(sorted(config.gridded_fits)):
        freq, _, _, Vobs, noiserms = read_gridded_visbilities(file, config)

        Vobs = Vobs[global_ylm[0].sort_idx_cols]
        config.noiserms = noiserms[global_ylm[0].sort_idx_cols]

        freqs.append(freq)

        plot_pool = multiprocessing.Pool(processes=4)
        print "\nProcessing frequency %.3f MHz" % (freq * 1e-6)

        result_freq_dir = os.path.join(result_dir, 'freq_%s' % i)
        os.mkdir(result_freq_dir)

        t = time.time()
        print "Building transformation matrix..."
        trm = util.get_alm2vis_matrix(ll, mm, global_ylm, 1, order='F')
        print "Done in %.2f s" % (time.time() - t)

        # plotting the visibilities
        plot_pool.apply_async(plot_visibilities, (uu, vv, ww, Vobs, os.path.join(result_freq_dir, 'visibilities.pdf')))
        plot_pool.apply_async(plot_2d_visibilities, (uu, vv, Vobs,
                                                     os.path.join(result_freq_dir, 'visibilities_2d.pdf')))

        plot_pool.apply_async(plot_2d_visibilities, (uu, vv, config.noiserms,
                                                     os.path.join(result_freq_dir, 'vis_error_2d.pdf')))

        alm_rec, alm_rec_noise, Vrec, cov_error = alm_ml_inversion(ll, mm, Vobs, uphis, uthetas, i, trm, config)

        jy2k = ((1e-26 * (const.c / freq) ** 2) / (2 * const.k_B.value))

        # Convert back to Kelvin
        alm_rec = alm_rec * jy2k
        alm_rec_noise = alm_rec_noise * jy2k
        cov_error = cov_error * jy2k

        print "Post processing..."
        # When w=0, odd l+m modes are not recovered, we compute them by interpolation
        alm_rec, ll2, mm2 = alm_post_processing(alm_rec, ll, mm, config)
        alm_rec_noise, _, _ = alm_post_processing(alm_rec_noise, ll, mm, config)
        cov_error, _, _ = alm_post_processing(cov_error, ll, mm, config)

        vlm_rec = util.alm2vlm(alm_rec, ll2)
        vlm_rec_noise = util.alm2vlm(alm_rec_noise, ll2)

        alms_rec.append(alm_rec)

        save_alm_rec(result_freq_dir, ll2, mm2, alm_rec, alm_rec_noise, cov_error)
        save_visibilities_rec(result_freq_dir, ru, uphis, uthetas, Vobs, Vrec)

        print "Plotting result"

        # plot vlm_rec
        plot_pool.apply_async(plot_vlm_rec_map, (ll2, mm2, vlm_rec, 4 * np.pi * cov_error,
                              os.path.join(result_freq_dir, 'vlm_rec_map.pdf')),
                              dict(vmin=1e-5, vmax=1e-2))

        plot_pool.apply_async(plot_vlm, (ll2, mm2, alm_rec, os.path.join(result_freq_dir, 'vlm_rec.pdf')))
        plot_pool.apply_async(plot_vlm, (ll2, mm2, alm_rec_noise,
                              os.path.join(result_freq_dir, 'vlm_rec_noise.pdf')))

        plot_pool.apply_async(plot_vlm, (ll2, mm2, 4 * np.pi * cov_error,
                              os.path.join(result_freq_dir, 'vlm_cov_error.pdf')))

        # plot visibilities
        plot_pool.apply_async(plot_vis_diff, (ru, Vobs, Vrec,
                              os.path.join(result_freq_dir, 'vis_minus_vis_rec.pdf')))

        # plot output sky
        plot_pool.apply_async(plot_sky_cart, (alm_rec, ll2, mm2, config.nside),
                              dict(theta_max=1 * config.fwhm,
                              savefile=os.path.join(result_freq_dir, 'output_sky.pdf')))

        # plot power spectra
        plot_pool.apply_async(plot_rec_power_sepctra, (ll2, mm2, alm_rec,
                              os.path.join(result_freq_dir, 'angular_power_spectra.pdf')))

        plot_pool.apply_async(plot_vis_vs_vis_rec, (ru, Vobs, Vrec,
                              os.path.join(result_freq_dir, 'vis_vs_vis_rec.pdf')))

        print nputils.stat(Vobs - Vrec)

        t = time.time()
        print "Waiting for plotting to finish...",
        plot_pool.close()
        plot_pool.join()
        print "Done in %.2f s" % (time.time() - t)

    global_ylm.close()

    np.save(os.path.join(result_dir, 'freqs'), freqs)

    if len(alms_rec) > 1:
        plot_2d_power_spectra(ll2, mm2, alms_rec, freqs, config, os.path.join(result_dir, 'power_spectra_2d.pdf'),
                              vmin=1e-3, vmax=1e-1)

    print '\nAll done!'
