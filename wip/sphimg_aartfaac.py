import os
import sys
import imp
import glob
import time
import math
import warnings
import multiprocessing

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import matplotlib.mlab as mlab

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

    thetas, phis = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))

    if savefile is not None:
        plt.gcf().savefig(savefile)
        plt.close()

    #print 'max(map) =%s  min(map) =%s\n' %(max(map), min(map))

    return thetas, phis, map


def plot_sky_cart_diff(alm1, alm2, ll1, mm1, ll2, mm2, nside, theta_max=0.35, savefile=None):
    cbs = plotutils.ColorbarSetting(plotutils.ColorbarInnerPosition(location=2, height="80%", pad=1))
    latra = np.degrees(theta_max)

    map1 = util.fast_alm2map(alm1, ll1, mm1, nside)
    map2 = util.fast_alm2map(alm2, ll2, mm2, nside)
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

###############################

def plot_cov_noise(noise1, noise2, noise3, histbin, savefile=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(noise1, histbin, alpha=0.5, label='Stockes V Map')
    ax.hist(noise2, histbin, alpha=0.5, label='Gridded Stockes V')
    ax.hist(noise3, histbin, alpha=0.5, label='SEFD')
    ax.set_xlabel('Noise')
    ax.set_ylabel('Count')
    ax.legend(loc='best')
    
   
    '''
    # the histogram of the data
    n, bins, patches = ax.hist(noise1, histbin, normed=1, facecolor='green', alpha=0.5, label='Stockes V Map')
    # add a 'best fit' line
    y = mlab.normpdf(bins, np.mean(noise1), np.std(noise1))
    ax.plot(bins, y, 'r--')

    n, bins, patches = ax.hist(noise2, histbin, normed=1, facecolor='blue', alpha=0.5, label='Gridded Stockes V')
    y = mlab.normpdf(bins, np.mean(noise2), np.std(noise2))
    ax.plot(bins, y, 'k--')

    n, bins, patches = ax.hist(noise3, histbin, normed=1, facecolor='red', alpha=0.5, label='SEFD')
    y = mlab.normpdf(bins, np.mean(noise3), np.std(noise3))
    ax.plot(bins, y, 'c--')

    ax.set_xlabel('Noise')
    ax.set_ylabel('Count')
    ax.legend(loc='best')

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    '''


    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)


############################


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
    ax1.scatter(uu, vv, c=V.real, lw=0)
    ax1.set_xlabel('U')
    ax1.set_xlabel('V')
    ax1.set_title('Real')

    ax2.scatter(uu, vv, c=V.imag, lw=0)
    ax2.set_xlabel('U')
    ax2.set_xlabel('V')
    ax2.set_title('Imaginary')

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

    ax1.scatter(mm, abs(vlm_rec), c='orange', marker='+', label='Recovered')
    ax1.set_yscale('log')
    # ax1.set_ylim(1e-5, 1)
    ax1.set_xlim(0, max(mm))
    ax1.set_xlabel('m')
    ax1.set_ylabel('abs(%s)' % name)
    ax1.legend()

    ax2.scatter(ll, abs(vlm_rec), c='orange', marker='+', label='Recovered')
    ax2.set_yscale('log')
    # ax2.set_ylim(1e-5, 5)
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

    #vmin=np.log10(np.amin(np.array(ps)))
    #vmax=np.log10(np.amax(np.array(ps)))   

    #print 'Max ps=%s Min ps =%s\n' %(vmax, vmin)

   
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



def plot_rec_power_sepctra(ll, mm, alm, lamb, savefile=None):
    ps = util.get_power_spectra(alm, ll, mm)
    JytoK = (2 * const.k_B.value)/(10. **(-26) * lamb**2) # changed
    ps = ps / (JytoK**2) # changed
    fig = plt.figure()
    fig, ax1 = plt.subplots()
    ax1.plot(np.unique(ll), ps, label='Beam modulated power spectra')
    ax1.set_yscale('log')
    ax1.set_xlabel('l')
    ax1.set_ylabel('cl [$K^2$ $sr^{-2}$]')

    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)


def plot_power_sepctra(ll, mm, alm, sel_ll, sel_mm, alm_rec, savefile=None):
    # TODO: check the normalization here!
    l_sampled = np.arange(max(sel_ll) + 1)[np.bincount(sel_ll) > 0]
    idx = np.where(np.in1d(sel_ll, l_sampled))[0]

    ps_rec = util.get_power_spectra(alm_rec[idx], sel_ll[idx], sel_mm[idx])
    ps = util.get_power_spectra(alm, ll, mm)

    fig = plt.figure()
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
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
    ax2.text(0.95, 0.05, 'std diff: %.3e\nmax diff: %.3e' % (diff.std(), diff.max()),
             ha='right', va='center', transform=ax2.transAxes)

    if savefile is not None:
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


def plot_vlm_diff(sel_ll, sel_mm, sel_vlm, vlm_rec, savefile=None, name='vlm'):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

    ax1.scatter(sel_mm, abs(vlm_rec.real - sel_vlm.real), c='green', marker='+', label='Input')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-5, 1e-1)
    ax1.set_xlim(0, max(sel_mm))
    ax1.set_xlabel('m')
    ax1.set_ylabel('abs(%s.real - %s_rec.real)' % (name, name))

    diff = vlm_rec.real - sel_vlm.real
    ax2.scatter(sel_ll, abs(diff), c='green', marker='+', label='Input')
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


def poly_fit_nu(freqsmhz, alms, fitorder):
    alms_cube = np.abs(np.array(alms))
    nfreqs = alms_cube.shape[0]
    Modeno = alms_cube.shape[1]
    
    # Fit a polynomial in alm with Freq

    yfit = lambda x: poly(x)
    alms_cube_fit_nu = []

    for i in range(Modeno):
        coeffs = np.polyfit(freqsmhz, alms_cube[:,i].reshape(nfreqs), fitorder)
        poly = np.poly1d(coeffs) # poly is now a polynomial in x that returns y
        #x_new = np.linspace(freqsmhz[0], freqsmhz[-1], 50)
        #alm_fit = f(freqsmhz)
        alm_fit = yfit(freqsmhz)
        alms_cube_fit_nu.append(alm_fit)
                       
    return alms_cube_fit_nu

def poly_logfit_nu(freqsmhz, alms, fitorder):
    alms_cube = np.abs(np.array(alms))
    nfreqs = alms_cube.shape[0]
    Modeno = alms_cube.shape[1]

    # Fit a polynomial in alm with Freq
    
    alms_cube_fit_nu = []
    logx = np.log(freqsmhz)
    logy = np.log(np.abs(alms_cube))
    yfit = lambda x: np.exp(poly(np.log(x)))

    for i in range(Modeno):
        coeffs = np.polyfit(logx, logy[:,i].reshape(nfreqs), fitorder)
        poly = np.poly1d(coeffs) # poly is now a polynomial in log(x) that returns log(y)
        #x_new = np.linspace(freqsmhz[0], freqsmhz[-1], 50)
        alm_fit = yfit(freqsmhz)
        alms_cube_fit_nu.append(alm_fit)

    
    return alms_cube_fit_nu


def pca_mode_subnu(alm_rec_cube_nu, transformed, submode):
    nfreqs = alm_rec_cube_nu.shape[0]
    Modeno = alm_rec_cube_nu.shape[1]

    print 'Size of Input matrix (PCA) is %s X %s\n' %(transformed.shape)

    alm_res_cube_nu = np.zeros((nfreqs, Modeno))

    for i in range(Modeno):
        sum = 0.0
        for j in range(submode):
            #sum = sum + np.abs(transformed[j,i])
            sum = sum + transformed[j,i]

        #print 'Sum of pca modes %s\n' %(sum)
        #alm_res_cube_nu[:,i] = np.abs(alm_rec_cube_nu[:,i]) - sum
        alm_res_cube_nu[:,i] = alm_rec_cube_nu[:,i] - sum

    #print alm_res_cube_nu

    print 'Size of Residual matrix (PCA) is %s X %s\n' %(alm_res_cube_nu.shape)

    return alm_res_cube_nu



def plot_std_alm(freqsmhz, idx, alm_rec_cube_nu, alm_rec_fit3_cube_nu, alm_rec_fit5_cube_nu, alm_res_pca3_nu, alm_res_pca5_nu, savefile=None):
    nfreqs = alm_rec_cube_nu.shape[0]
    Modeno = alm_rec_cube_nu.shape[1]

    Sigma_rec_fit3 = np.zeros(nfreqs)
    Sigma_rec_fit5 = np.zeros(nfreqs)
    Sigma_res_pca3 = np.zeros(nfreqs)
    Sigma_res_pca5 = np.zeros(nfreqs)
        
    # Subtract the k eigen modes/vectors from the data

    for i in range(nfreqs):
        Sigma_rec_fit3[i] = np.std(alm_rec_cube_nu[i,:].reshape(Modeno) - alm_rec_fit3_cube_nu[i,:].reshape(Modeno))
        Sigma_rec_fit5[i] = np.std(alm_rec_cube_nu[i,:].reshape(Modeno) - alm_rec_fit5_cube_nu[i,:].reshape(Modeno))

        Sigma_res_pca3[i] = np.std(alm_res_pca3_nu[i,:].reshape(Modeno))

        Sigma_res_pca5[i] = np.std(alm_res_pca5_nu[i,:].reshape(Modeno))
  
                    
    fig = plt.figure(figsize=(11,6))
    plt.plot(freqsmhz, Sigma_rec_fit3, label='(Inp-Fit_ord3) std')
    plt.plot(freqsmhz, Sigma_rec_fit5, label='(Inp-Fit_ord5) std')
    plt.plot(freqsmhz, Sigma_res_pca3, label='(Inp-pca_3Mode) std') 
    plt.plot(freqsmhz, Sigma_res_pca5, label='(Inp-pca_5Mode) std')
    plt.xlabel('Frequency', fontsize=18)
    plt.ylabel('Sigma', fontsize=18)
    plt.legend(loc='best')

    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)


def plot_polypcapower_spectra(ll, mm, almpoly3, almpoly5, almskl3, almskl5, savefile=None):
    # TODO: check the normalization here!
    
    pspoly3 = util.get_power_spectra(almpoly3, ll, mm)
    pspoly5 = util.get_power_spectra(almpoly5, ll, mm)

    psskl3 = util.get_power_spectra(almskl3, ll, mm)
    psskl5 = util.get_power_spectra(almskl5, ll, mm)
    
    #####################################################
    
    LL = np.unique(ll)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(LL, pspoly3, label='Res power spectra (Poly3)')
    ax.plot(LL, pspoly5, label='Res power spectra (Poly5)')

    ax.plot(LL, psskl3, label='Res power spectra (SKL3)')
    ax.plot(LL, psskl5, label='Res power spectra (SKL5)')

    # ax1.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$\ell$')
    ax.set_ylabel('C_{\ell} [$Jy^2$]')
    ax.legend(loc='best')

    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)
    

def plot_polypcapower_bin_spectra(ll, mm, almpoly3, almpoly5, almskl3, almskl5, binno, savefile=None):
    pspoly3 = util.get_power_spectra(almpoly3, ll, mm)
    pspoly5 = util.get_power_spectra(almpoly5, ll, mm)

    psskl3 = util.get_power_spectra(almskl3, ll, mm)
    psskl5 = util.get_power_spectra(almskl5, ll, mm)

    #######################################

    # Log Bin l and ps values
    
    LL = np.unique(ll)

    kv=binno/np.log10(np.max(LL)/np.min(LL))
    lavg = np.zeros(binno)
    pspoly3avg = np.zeros(binno)
    pspoly5avg = np.zeros(binno)
    psskl3avg = np.zeros(binno)
    psskl5avg = np.zeros(binno)
    count = np.zeros(binno)
 
    for i in range(len(LL)):
        lval = LL[i]
        if(lval > min(LL) and lval < max(LL)):
            NUGrid=int(math.floor(kv*np.log10(lval/min(LL))))
            #print 'NUGrid = %s\n' %(NUGrid)
            if NUGrid < 0:
                NUGrid = 0
            if NUGrid >= binno:
                NUGrid = binno - 1
            
            lavg[NUGrid] = lavg[NUGrid] + lval
            pspoly3avg[NUGrid] = pspoly3avg[NUGrid] + pspoly3[i]
            pspoly5avg[NUGrid] = pspoly5avg[NUGrid] + pspoly5[i]

            psskl3avg[NUGrid] = psskl3avg[NUGrid] + psskl3[i]
            psskl5avg[NUGrid] = psskl5avg[NUGrid] + psskl5[i] 

            count[NUGrid] = count[NUGrid] + 1 


    # Bin Avg

    for i in range(binno):
        if count[i] > 0:
            lavg[i] = lavg[i]/count[i]
            pspoly3avg[i] = pspoly3avg[i]/count[i] 
            pspoly5avg[i] = pspoly5avg[i]/count[i] 
            psskl3avg[i] = psskl3avg[i]/count[i]
            psskl5avg[i] = psskl5avg[i]/count[i]

        else:
            lavg[i] = 0.0
            pspoly3avg[i] = 0.0
            pspoly5avg[i] = 0.0
            psskl3avg[i] = 0.0
            psskl5avg[i] = 0.0

    lavg = lavg[np.nonzero(lavg)]
    pspoly3avg = pspoly3avg[np.nonzero(pspoly3avg)]
    pspoly5avg = pspoly5avg[np.nonzero(pspoly5avg)]
    psskl3avg = psskl3avg[np.nonzero(psskl3avg)]
    psskl5avg = psskl5avg[np.nonzero(psskl5avg)]
      
    # Plot

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lavg, pspoly3avg, label='Res power spectra (Poly3)')
    ax.plot(lavg, pspoly5avg, label='Res power spectra (Poly5)')
    ax.plot(lavg, psskl3avg, label='Res power spectra (SKL3)')
    ax.plot(lavg, psskl5avg, label='Res power spectra (SKL5)')

    # ax1.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$\ell$')
    ax.set_ylabel('$C_{\ell}$ [$Jy^2$]')
    ax.legend(loc='best')

    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)
    
     

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
    config.out_lm_even_only = False
    ll2, mm2 = get_out_lm_sampling(config)
    config.out_lm_even_only = True

    alm2 = np.zeros_like(ll2, dtype=np.complex)

    for m in np.unique(mm):
        x2 = ll2[mm2 == m]
        yr = alm[mm == m].real
        yi = alm[mm == m].imag
        y2r = idct(dct(yr), n=len(x2)) / len(x2)
        y2i = idct(dct(yi), n=len(x2)) / len(x2)
        alm2[mm2 == m] = y2r + 1j * y2i

    return alm2, ll2, mm2


def alm_ml_inversion(ll, mm, Vobs, uphis, uthetas, i, trm, config):

    def get_dct_fct(m, t):
        if t == 'real' and m == 0:
            return config.dct_fct_r_m0
        elif t == 'real' and m > 0:
            return config.dct_fct_r_m1
        elif t == 'imag' and m == 0:
            return config.dct_fct_i_m0
        elif t == 'imag' and m > 0:
            return config.dct_fct_i_m1

    if config.use_dct:
        dct_blocks = []
        # dct_blocks_full = []
        if isinstance(config.dct_dl, (list, np.ndarray)):
            dl = float(config.dct_dl[i])
            dl_m0 = float(config.dct_dl_m0[i])
        else:
            dl = float(config.dct_dl)
            dl_m0 = float(config.dct_dl_m0)

        print "Using DCT with dl=%s and dl_m0=%s" % (dl, dl_m0)
        print "Building DCT Matrix ..."
        # t = time.time()
        for sel_block in [trm.m0_l_even, trm.lm_even, trm.lm_even]:
            for m in np.unique(mm[sel_block]):
                n = len(ll[sel_block][mm[sel_block] == m])
                if m > config.dct_mmax_full_sample:
                    if m == 0:
                        nk = int(np.ceil(n / dl_m0))
                    else:
                        nk = int(np.ceil(n / dl))
                    dct_fct = get_dct_fct(m, 'real')
                    dct_blocks.append(dct_fct(n, nk))
                    # dct_blocks_full.append(dct_fct(n, nk, nki=n))
                else:
                    dct_blocks.append(np.eye(n))
                    # dct_blocks_full.append(np.eye(n))
        # print time.time() - t
        # t = time.time()
        dct_real = block_diag(dct_blocks).tocsr()
        # dct_real_full = block_diag(dct_blocks_full).tocsr()
        # print time.time() - t

        dct_blocks = []
        # dct_blocks_full = []

        for sel_block in [trm.m0_l_odd, trm.lm_odd, trm.lm_odd]:
            for m in np.unique(mm[sel_block]):
                n = len(ll[sel_block][mm[sel_block] == m])
                if m > config.dct_mmax_full_sample:
                    if m == 0:
                        nk = int(np.ceil(n / dl_m0))
                    else:
                        nk = int(np.ceil(n / dl))
                    dct_fct = get_dct_fct(m, 'imag')
                    dct_blocks.append(dct_fct(n, nk))
                    # dct_blocks_full.append(dct_fct(n, nk, nki=n))
                else:
                    dct_blocks.append(np.eye(n))
                    # dct_blocks_full.append(np.eye(n))

        dct_imag = block_diag(dct_blocks).tocsr()
        # dct_imag_full = block_diag(dct_blocks_full).tocsr()

        print "Computing dot products of T and DCT ..."
        # print dct_real.nnz, dct_real.shape
        t = time.time()
        # # X_r = np.dot(trm.T_r.T, dct_real)
        # X_r = (dct_real.T.dot(trm.T_r)).T
        # print time.time() - t
        # dct_real_arr = dct_real.toarray()
        # X_r = np.dot(trm.T_r.T, dct_real_arr)
        # X_i = np.dot(trm.T_i.T, dct_imag.toarray())
        # temp = np.asfortranarray(trm.T_r)
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
    # PERF: quite some time is passed here as well, on both (about 6-s for a 14k by 2k matrix)
    # check C contious, etc...
    X_r_dot_C_Dinv = C_Dinv.T.dot(X_r).T
    lhs_r = np.dot(X_r_dot_C_Dinv, X_r) + np.eye(X_r.shape[1]) * config.reg_lambda
    rhs_r = np.dot(X_r_dot_C_Dinv, Vobs.real)

    X_i_dot_C_Dinv = C_Dinv.T.dot(X_i).T
    lhs_i = np.dot(X_i_dot_C_Dinv, X_i) + np.eye(X_i.shape[1]) * config.reg_lambda
    rhs_i = np.dot(X_i_dot_C_Dinv, Vobs.imag)
    print "Done in %.2f s" % (time.time() - t)

    # print "Building covariance matrix ...",
    # t = time.time()
    # lhs_r_err_1 = np.linalg.inv(lhs_r)
    # lhs_r_err_2 = np.dot(X_r_dot_C_Dinv, X_r)
    # cov_error_r = np.sqrt(np.diag(np.dot(np.dot(lhs_r_err_2, lhs_r_err_1), lhs_r_err_1)))

    # lhs_i_err_1 = np.linalg.inv(lhs_i)
    # lhs_i_err_2 = np.dot(X_i_dot_C_Dinv, X_i)
    # cov_error_i = np.sqrt(np.diag(np.dot(np.dot(lhs_i_err_2, lhs_i_err_1), lhs_i_err_1)))

    # if config.use_dct:
    #     # cov_error_r = np.dot(np.dot(cov_error_r, dct_real.T), dct_real_full)\
    #     # print cov_error_r.shape, dct_real.shape, dct_real_full.shape, dct_real.dot(cov_error_r.T).shape
    #     cov_error_r = dct_real_full.T.dot(dct_real.dot(cov_error_r.T)).T
    #     # cov_error_i = np.dot(np.dot(cov_error_i, dct_imag.T), dct_imag_full)
    #     cov_error_i = dct_imag_full.T.dot(dct_imag.dot(cov_error_i.T)).T

    # cov_error = np.abs(trm.recombine(cov_error_r, cov_error_i))
    cov_error = np.zeros_like(ll)
    # print "Done in %.2f s" % (time.time() - t)

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


def get_config(dirname):
    return imp.load_source('config', os.path.join(dirname, 'config.py'))

def save_data(filename, data, columns_name):
    df = pd.DataFrame(dict(zip(columns_name, data)))
    df.to_csv(filename)

def save_alm_rec(dirname, ll, mm, freq, alm_rec, alm_rec_noise, cov_error):
    columns = ['ll', 'mm', 'freq', 'alm_rec', 'alm_rec_noise', 'cov_error']
    filename = os.path.join(dirname, 'alm_rec.dat')
    print "Saving alm result to:", filename
    save_data(filename, [ll, mm, freq, alm_rec, alm_rec_noise, cov_error], columns)
    
def save_visibilities_rec(dirname, ru, uphis, uthetas, Vobs, Vrec):
    columns = ['ru', 'uphis', 'uthetas', 'Vobs', 'Vrec']
    filename = os.path.join(dirname, 'visibilities_rec.dat')
    print "Saving vis result to:", filename
    save_data(filename, [ru, uphis, uthetas, Vobs, Vrec], columns)
    

def load_alm(dirname):
    filename = os.path.join(dirname, 'alm_rec.dat')
    print "Loading alm result from:", filename
    # Added; avoid 1st line
    f = open(filename,'r')
    lines = f.readlines()[1:]  
    f.close()
    a = [i.strip().split(',') for i in lines]
    a = np.array(a)
    if a.shape[1] == 7:
        x, alm, alm_noise, cov_error, freq, ll, mm = a.T
        return ll, mm, map(float,freq), map(complex,alm), map(complex,alm_noise), map(complex,cov_error)
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
        ll, mm, freq, alm, alm_noise, cov_error = load_alm(freq_dir)
        freq_res.append([freq, alm, alm_noise, cov_error])

    freq, alm, alm_noise, cov_error = zip(*freq_res)

    return ll.astype(int), mm.astype(int), freq, alm, alm_noise, cov_error


def cov_noise(filename, config):
    dirname = os.path.dirname(filename)
    basename = '_'.join(os.path.basename(filename).split('_')[:-1])
    g_real = os.path.join(dirname, basename + '_GR.fits')
    g_imag = os.path.join(dirname, basename + '_GI.fits')
    stockes_v = os.path.join(dirname, basename + '_V.fits')
    g_real_stockes_v = os.path.join(dirname, basename + '_GRV.fits')
    g_imag_stockes_v = os.path.join(dirname, basename + '_GIV.fits')
    weighting = os.path.join(dirname, basename + '_W.fits')

    assert os.path.exists(g_real) and os.path.exists(g_imag)

    f_g_real = pyfits.open(g_real)
    f_g_imag = pyfits.open(g_imag)
    freq = f_g_real[0].header['CRVAL3'] / float(1e6)
    du = f_g_real[0].header['CDELT1']
    dv = f_g_real[0].header['CDELT2']
    nu = f_g_real[0].header['NAXIS1']
    nv = f_g_real[0].header['NAXIS2']
    delnu = f_g_real[0].header['CDELT3'] # in Hz
    
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

    # Method 1: get the noise rms on the image looking at stocks V dirty image
    #           and scale that for each grid points using weighting
    # Method 2: get the absolute value of the stock V visibilities
    # Method 3: calculate from SEFD and scale with grid point weighting

    # Method 1:
    
    print "Using Stockes V file"
    noiserms = np.std(pyfits.open(stockes_v)[0].data[0][0])
    w = pyfits.open(weighting)[0].data[0][0]
    w = w[idx_u, idx_v].flatten()[idx]
    noiserms1 = noiserms * np.sqrt(w)
    print noiserms1.shape

    # Method 2:
    print "Using Gridded Stockes V data"
    g_v_r = pyfits.open(g_real_stockes_v)[0].data[0][0][idx_u, idx_v].flatten()
    g_v_i = pyfits.open(g_imag_stockes_v)[0].data[0][0][idx_u, idx_v].flatten()
    g_v = g_v_r[idx] + 1j * g_v_i[idx]
    noiserms2 = abs(g_v)
    print noiserms2.shape

    # Method 3:
    print "Using SEFD at NCP"
    w = pyfits.open(weighting)[0].data[0][0]
    w = w[idx_u, idx_v].flatten()[idx]
    noiserms3 = (config.SEFD / np.sqrt(2.*delnu*config.Int_time)) * (1. / np.sqrt(w))
    print noiserms3.shape 

    return noiserms1, noiserms2, noiserms3



def read_gridded_visbilities(filename, config):
    dirname = os.path.dirname(filename)
    basename = '_'.join(os.path.basename(filename).split('_')[:-1])
    g_real = os.path.join(dirname, basename + '_GR.fits')
    g_imag = os.path.join(dirname, basename + '_GI.fits')
    weighting = os.path.join(dirname, basename + '_W.fits')

    assert os.path.exists(g_real) and os.path.exists(g_imag)

    f_g_real = pyfits.open(g_real)
    f_g_imag = pyfits.open(g_imag)
    freq = f_g_real[0].header['CRVAL3'] / float(1e6) # in MHz
    du = f_g_real[0].header['CDELT1']
    dv = f_g_real[0].header['CDELT2']
    nu = f_g_real[0].header['NAXIS1']
    nv = f_g_real[0].header['NAXIS2']
    nfreq =  f_g_real[0].header['NAXIS3']
    delnu = f_g_real[0].header['CDELT3'] / float(1e6) # in MHz

    print '\nFreq=%s MHz du=%s dv=%s nu=%s nv=%s nfreq=%s delnu=%s MHz\n' %(freq,du,dv,nu,nv,nfreq,delnu)

    u = du * np.arange(-nu / 2, nu / 2) 
    v = dv * np.arange(-nv / 2, nv / 2)

    #uu, vv = np.meshgrid(u, v)
           
    # Read data
    vis_freq = []
    uu_freq = []
    vv_freq = []
    noiserms_freq = []
    freqs = []

    for i in range(nfreq):
        vis = f_g_real[0].data[i][:][:] + 1j * f_g_imag[0].data[i][:][:]
        uu, vv = np.meshgrid(u, v)
        idx_u, idx_v = np.nonzero(vis)
        vis = vis[idx_u, idx_v].flatten()
        uu = uu[idx_u, idx_v].flatten()
        vv = vv[idx_u, idx_v].flatten()
        ru = np.sqrt(uu ** 2 + vv ** 2)
        idx = (ru >= config.uv_rumin) & (ru <= config.uv_rumax)
        freq = freq + i*delnu
        vis = vis[idx]
        uu = uu[idx]
        vv = vv[idx]

        vis_freq.append(vis)
        uu_freq.append(uu)
        vv_freq.append(vv)
        freqs.append(freq)

        # calculate from SEFD and scale with grid point weighting
        if os.path.exists(weighting):
            print "Using weighting file and SEFD"
            w = pyfits.open(weighting)[0].data[i][:][:]
            w = w[idx_u, idx_v].flatten()[idx]
            noiserms = (config.SEFD / np.sqrt(2.*delnu*config.Int_time)) * (1. / np.sqrt(w))
        else:
            noiserms = config.noiserms
        
        noiserms_freq.append(noiserms)

    
    return freqs, uu_freq, vv_freq, vis_freq, noiserms_freq


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
