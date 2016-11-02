import os
import time

import numpy as np

from astropy.cosmology import WMAP9 as cosmo
import astropy.constants as const
import astropy.units as u

from scipy.special import jv, jvp, sph_jn
from scipy.optimize import brentq

import matplotlib.pyplot as plt

from libwise import plotutils


def progress_report(n):
    t = time.time()

    def report(i):
        print "\r",
        eta = ""
        if i > 0:
            remaining = (np.round((time.time() - t) / float(i) * (n - i)))
            eta = " (ETA: %s)" % time.strftime("%H:%M:%S", time.localtime(time.time() + remaining))
        print "Progress: %s / %s%s" % (i + 1, n, eta),
        if i == n - 1:
            print ""

    return report


def Jn(r, n):
    return (np.sqrt(np.pi / (2 * r)) * jv(n + 0.5, r))


def Jn_zeros(n, nt):
    zerosj = np.zeros((n + 1, nt))
    zerosj[0] = np.arange(1, nt + 1) * np.pi
    points = np.arange(1, nt + n + 1) * np.pi
    racines = np.zeros(nt + n)

    pr = progress_report(n)

    for i in range(1, n + 1):
        pr(i - 1)
        for j in range(nt + n - i):
            foo = brentq(Jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:nt] = racines[:nt]

    return (zerosj)


def rJnp(r, n):
    return (0.5 * np.sqrt(np.pi / (2 * r)) * jv(n + 0.5, r) + np.sqrt(np.pi * r / 2) * jvp(n + 0.5, r))


def rJnp_zeros(n, nt):
    zerosj = np.zeros((n + 1, nt))
    zerosj[0] = (2. * np.arange(1, nt + 1) - 1) * np.pi / 2
    points = (2. * np.arange(1, nt + n + 1) - 1) * np.pi / 2
    racines = np.zeros(nt + n)
    for i in range(1, n + 1):
        for j in range(nt + n - i):
            foo = brentq(rJnp, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:nt] = racines[:nt]
    return (zerosj)


def delta_freq_to_R(delta_freq, z_obs):
    delta_freq = delta_freq * u.s ** -1
    f21 = (const.c / (21 * u.cm)).decompose()

    return ((delta_freq * const.c * (1 + z_obs) ** 2) / (cosmo.H(z_obs) * f21)).decompose()


def get_almn(alm, ll, delta_nu, z_obs, zeros):
    n = alm.shape[0]
    nl = len(np.unique(ll))
    delta_r = delta_freq_to_R(delta_nu, z_obs).value
    rs = np.arange(n) * delta_r
    R = n * delta_r
    qln = zeros
    kln = qln / R

    almn = np.zeros_like(alm)

    pr = progress_report(nl)
    for i, l in enumerate(np.unique(ll)):
        pr(i)
        kln = qln[l] / R
        jns = np.array([sph_jn(l, x)[0][-1] for x in (kln * rs[:, np.newaxis]).flatten()])
        jns = jns.reshape(len(kln), len(rs))
        almn[:, ll == l] = np.dot(alm[:, ll == l].T, jns).T

    return almn


def get_power_spectra(alm, ll, mm):
    l_uniq = np.unique(ll)
    return np.array([np.sum(np.abs(alm[ll == l]) ** 2) / (2 * l + 1) for l in l_uniq])


def get_3d_power_spectra(alms, ll, mm):
    alm_cube = np.array(alms)
    ps = []
    for i in range(alm_cube.shape[0]):
        ps.append(get_power_spectra(alm_cube[i], ll, mm))

    return np.array(ps)


def plot_3d_power_spectra(ps, ll, nn, savefile=None, vmin=1e-14, vmax=1e-10):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    cbs = plotutils.ColorbarSetting(plotutils.ColorbarOutterPosition())
    extent = (min(ll), max(ll), min(nn), max(nn))

    im_mappable = ax.imshow(np.array(ps), aspect='auto', norm=plotutils.LogNorm(),
                            vmin=vmin, vmax=vmax, extent=extent)
    cbs.add_colorbar(im_mappable, ax)
    ax.set_ylabel("n")
    ax.set_xlabel('l')
    ax.set_title("Power spectra")

    if savefile is not None:
        fig.savefig(savefile)
        plt.close(fig)

    return np.array(ps), ax


def test_almn():
    path_to_data = os.path.expanduser('~')

    print "Loading the data..."
    alm_fg_cube = np.load(os.path.join(path_to_data, 'alm_fg_cube.npy'))
    ll = np.load(os.path.join(path_to_data, 'alm_ll.npy'))
    mm = np.load(os.path.join(path_to_data, 'alm_mm.npy'))

    # Select some range of ll
    idx = (ll > 200) & (ll < 1000)

    alm_fg_cube = alm_fg_cube[:, idx]
    ll = ll[idx]
    mm = mm[idx]

    delta_nu = 0.5 * 1e6
    z_obs = 9

    print "Computing the zeroth (for the kln)..."
    zeros = Jn_zeros(max(ll), alm_fg_cube.shape[0])
    np.save('qln', zeros)
    # zeros = np.load('qln.npy')

    print "Computing the almn..."
    almn_fg = get_almn(alm_fg_cube, ll, delta_nu, z_obs, zeros)
    np.save('almn_fg', almn_fg)
    # almn_fg = np.load('almn_fg.npy')

    print "Computing the power spectra and plotting it"
    ps = get_3d_power_spectra(almn_fg, ll, mm)
    nn = np.arange(alm_fg_cube.shape[0])

    plot_3d_power_spectra(ps, ll, nn, vmin=1e-18, vmax=1e-12,
                          savefile=os.path.join(path_to_data, 'power_spectra_fg.pdf'))


if __name__ == '__main__':
    test_almn()
