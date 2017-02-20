import numpy as np

import matplotlib.pyplot as plt

from scipy import stats
from scipy.signal import get_window

import healpy as hp

import astropy.units as u
import astropy.constants as const
from astropy.cosmology import Planck15 as cosmo

import util
from libwise import nputils, plotutils

f21 = 1.4204057 * 1e9 * u.Hz


class EorWindow(object):

    def __init__(self, name, freqs, i_start, i_end, i_fg_start, i_fg_end):
        self.name = name
        self.idx_freq = slice(i_start, i_end)
        self.idx_f_fg = slice(i_fg_start, i_fg_end)
        self.freqs = freqs[self.idx_freq]
        self.fmhz = freqs[self.idx_freq] / 1e6

        # This is the number of k_par that will be computed
        self.M = nputils.get_next_odd(len(self.fmhz))

        # This is the index that goes from FG window (idx_f_fg) to EoR window (idx_freq)
        self.idx_freg_f_fg = slice(self.idx_freq.start - self.idx_f_fg.start, self.idx_freq.stop - self.idx_f_fg.start)

        self.mfreq = self.freqs[0] + (self.freqs[-1] - self.freqs[0]) / 2.
        self.z = freq_to_z(self.mfreq * u.Hz)

        self.df = (freqs[self.idx_freq][1] - freqs[self.idx_freq][0])
        self.bw = (len(freqs[self.idx_freq]) - 1) * self.df


class EorWindowList(object):

    def __init__(self, freqs):
        self.windows = dict()
        self.freqs = freqs

    def add(self, name, i_start, i_end, i_fg_start, i_fg_end):
        self.windows[name] = EorWindow(name, self.freqs, i_start, i_end, i_fg_start, i_fg_end)

    def get(self, name):
        return self.windows[name]


class PowerSpectraGenerator(object):

    def __init__(self, eor_window, window_fct, beam_type, beam_fwhm, theta_fov, f_sky, ll, mm):
        self.eor = eor_window
        z = self.eor.z
        self.ll = ll
        self.mm = mm

        X = cosmo.comoving_transverse_distance(self.eor.z).to(u.Mpc).value * cosmo.h
        Y = ((const.c * (1 + z) ** 2) / (cosmo.H(z) * f21)).to(u.Mpc * u.Hz ** -1).value * cosmo.h
        self.ps_norm = self.eor.df ** 2 / self.eor.bw * X ** 2 * Y

        if window_fct is not None:
            self.window = get_window(window_fct, len(self.eor.freqs))[:, np.newaxis]
            self.ps_norm = self.ps_norm / (self.window ** 2).mean()
        else:
            self.window = None

        self.pb_corr = get_sph_pb_corr(beam_type, beam_fwhm, theta_fov, 1024)

        self.delay = get_delay(self.eor.fmhz, M=self.eor.M)
        self.k_per = l_to_k(np.unique(self.ll), z)
        self.k_par = delay_to_k(self.delay * u.us, z)

        self.all_k = np.sqrt(self.k_per ** 2 + self.k_par[:, np.newaxis] ** 2)
        self.kmin = self.all_k.min()

        self.f_sky = f_sky

    def get_ps2d(self, alm_cube):
        return get_2d_power_spectra(alm_cube, self.ll, self.mm, self.eor.fmhz, M=self.eor.M,
                                    method='nudft', window=self.window)[1] * self.pb_corr * self.ps_norm

    def plot_ps2d(self, ps2d, **kargs):
        plot_2d_power_spectra(ps2d, self.ll, self.delay, kper=self.k_per, kpar=self.k_par, **kargs)

    def get_ps(self, alm_cube):
        return get_power_spectra(alm_cube, self.ll, self.mm) * self.pb_corr

    def plot_ps(self, pspec, **kargs):
        plot_power_spectra(pspec, self.ll, self.eor.fmhz, kper=self.k_per, **kargs)

    def set_kbins(self, kbins):
        self.kbins = kbins
        self.k_mean, _, _ = stats.binned_statistic(self.all_k.flatten(), self.all_k.flatten(),
                                                   'mean', self.kbins)

    def get_ps1d(self, ps2d):
        return get_1d_power_spectra(ps2d * 1e6, self.k_per, self.k_par, self.ll, self.f_sky, self.kbins)[0]

    def get_ps1d_err(self, ps2d):
        ps_1d, ps_1d_err, _ = get_1d_power_spectra(ps2d * 1e6, self.k_per, self.k_par, self.ll,
                                                   self.f_sky, self.kbins)
        return ps_1d, ps_1d_err


def bin_data(x, y, nbins):
    m, bins_edges, _ = stats.binned_statistic(x, y, 'mean', nbins)
    s, _, _ = stats.binned_statistic(x, y, np.std, nbins)
    bins = np.array([(a + b) / 2. for a, b in nputils.pairwise(bins_edges)])

    return bins, m, s


def plot_binned(x, y, nbins, **kargs):
    bins, m, s = bin_data(x, y, nbins)
    plt.errorbar(bins, m, s / np.sqrt(nbins), **kargs)


def ax_plot_binned(ax, x, y, nbins, **kargs):
    bins, m, s = bin_data(x, y, nbins)
    ax.errorbar(bins, m, s / np.sqrt(nbins), **kargs)


def fill_gaps(alm_cube, gaps, fill_with=np.nan):
    alm_filled = []
    for i, alm in enumerate(alm_cube):
        alm_filled.append(alm)
        if i < len(gaps) and gaps[i] > 0:
            alm_filled.extend([np.ones_like(alm_cube[0]) * fill_with] * gaps[i])

    return np.array(alm_filled)


def get_gaps(freqs):
    df = freqs[1] - freqs[0]
    return np.array(np.round(np.diff(freqs) / df) - 1).astype(int)


def freq_to_index(freqs, start=1):
    return np.round((freqs - freqs[0]) / (freqs[1] - freqs[0])).astype(int) + start


def rmean(data, axis=0):
    m = np.mean(data, axis=axis)
    return data - m


def freq_to_z(freq):
    return (f21 / freq).decompose().value - 1


def delay_to_k(delay, z):
    '''Return inverse co-moving distance in h Mpc^-1'''
    return ((2 * np.pi * cosmo.H(z) * f21 * delay) / (const.c * (1 + z) ** 2)).to(u.Mpc ** -1).value / cosmo.h


def l_to_k(ll, z):
    '''Return inverse co-moving distance in h Mpc^-1'''
    return (ll / cosmo.comoving_transverse_distance(z)).to(u.Mpc ** -1).value / cosmo.h


def wedge_fct(fwhm, z, k_per):
    ''' From Dillon 2013'''
    dm = cosmo.comoving_transverse_distance(z)
    return np.sin(fwhm) * (dm * cosmo.H(z) / (const.c * (1 + z))).decompose() * k_per


def get_sph_pb_corr(beam_type, beam_fwhm, theta_max, nside):
    thetas, phis = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    beam = util.get_beam(thetas, beam_type, beam_fwhm, None)
    if theta_max:
        fov_cut = util.tophat_beam(thetas, 2 * theta_max)
    else:
        fov_cut = 1
    pb_corr = 1 / ((beam * fov_cut) ** 2).mean()

    return pb_corr


def get_cart_pb_corr(beam_type, beam_fwhm, res, img_shape):
    nx, ny = img_shape

    thxval = res * np.arange(-nx / 2., nx / 2.)
    thyval = res * np.arange(-ny / 2., ny / 2.)
    thx, thy = np.meshgrid(thxval, thyval)

    fov_map = (res * nx) * (res * ny)
    beam_map = util.get_beam(np.sqrt(thx ** 2 + thy ** 2), beam_type, beam_fwhm, None)
    pb_corr = fov_map / (beam_map ** 2).mean()

    return pb_corr


def nudft(x, y, M=None, w=None, dx=None):
    if M is None:
        M = len(x)

    if dx is None:
        dx = x[1] - x[0]

    if w is not None:
        y = y * w  # [:, np.newaxis]

    df = 1 / (dx * M)
    k = df * np.arange(-(M / 2), M - (M / 2))

    X = np.exp(-2 * np.pi * 1j * k * x[:, np.newaxis])

    return k, np.tensordot(y, X.T, axes=[0, 1]).T


def lssa(x, y, M, w=None, dx=None):
    if dx is None:
        dx = x[1] - x[0]

    k = np.fft.fftshift(np.fft.fftfreq(M, float(dx)))

    if w is not None:
        y *= w  # [:, np.newaxis]

    # Version with noise covariance matrix (does not really improve stuff):
    # C_Dinv = np.diag(1 / noiserms ** 2)
    # Y = np.dot(np.linalg.pinv(np.dot(np.dot(A.T, C_Dinv), A)), np.dot(A.T, C_Dinv))

    A = np.exp(2. * np.pi * 1j * k * x[:, np.newaxis]) / len(x)
    Y = np.dot(np.linalg.pinv(np.dot(A.T, A)), A.T)

    return k, np.tensordot(y.T, Y.T, axes=[1, 0]).T


def get_power_spectra(alm, ll, mm):
    l_uniq = np.unique(ll)
    dim = alm.ndim
    if dim == 1:
        alm = alm[np.newaxis, :]
    ps = np.array([np.sum(np.abs(alm[:, ll == l]) ** 2, axis=1) / (l + 1) for l in l_uniq]).T
    if dim == 1:
        ps = ps[0]
    return ps


def get_delay(freqs, M=None, dx=None, half=True):
    if dx is None:
        dx = freqs[1] - freqs[0]
    if M is None:
        M = len(freqs)

    df = 1 / (dx * M)
    delay = df * np.arange(-(M / 2), M - (M / 2))

    if half:
        M = len(delay)
        delay = delay[M / 2 + 1:]

    return delay


def get_2d_power_spectra(alm, ll, mm, freqs, M=None, window=None, dx=None, half=True, method='nudft'):
    if method == 'nudft':
        delay, nudft_cube = nudft(freqs, rmean(alm), M=M, w=window, dx=dx)
    else:
        delay, nudft_cube = lssa(freqs, rmean(alm), M=M, w=window, dx=dx)

    ps2d = get_power_spectra(nudft_cube, ll, mm)

    if half:
        M = len(delay)
        delay = delay[M / 2 + 1:]
        ps2d = 0.5 * (ps2d[M / 2 + 1:] + ps2d[:M / 2][::-1])

    return delay, ps2d


def get_power_spectra_cart(cart_map, res, el):
    m_u = 1 / res * np.linspace(-1 / 2., 1 / 2., cart_map.shape[0])
    m_v = 1 / res * np.linspace(-1 / 2., 1 / 2., cart_map.shape[1])
    m_uu, m_vv = np.meshgrid(m_u, m_v)

    l = [(a - (b - a) / 2., b + (b - a) / 2.) for a, b in nputils.pairwise(el)]
    bins_edges = np.array([k[0] for k in l] + [l[-2][1], l[-1][1]]) / (2 * np.pi)

    _, ps_rec_cart, _ = bin_data(np.sqrt(m_uu ** 2 + m_vv ** 2).flatten(),
                                 np.abs(np.fft.fftshift(np.fft.ifft2(cart_map) ** 2).flatten()),
                                 bins_edges)

    return ps_rec_cart


def get_2d_power_spectra_cart(cart_cube, res, el, freqs, M=None, window=None, dx=None, half=True,
                              method='nudft'):
    nf, nx, ny = cart_cube.shape
    # cart_cube_ravel = cart_cube.reshape(nf, nx * ny)
    if method == 'nudft':
        delay, nudft_cube = nudft(freqs, rmean(cart_cube), M=M, w=window, dx=dx)
    else:
        delay, nudft_cube = lssa(freqs, rmean(cart_cube), M=M, w=window, dx=dx)

    # nudft_cube = nudft_cube.reshape(M, nx, ny)

    if half:
        delay = delay[M / 2 + 1:]
        nudft_cube = nudft_cube[M / 2 + 1:]

    ps2d = np.array([get_power_spectra_cart(cart_map, res, el) for cart_map in nudft_cube])

    return delay, ps2d


def get_1d_ps_sample_count(k_par, k_per, el, bins, f_sky):
    el_full = np.arange(min(el), max(el))
    k_per_full = el_full * k_per[0] / el[0]
    k_full = np.sqrt(k_per_full ** 2 + k_par[:, np.newaxis] ** 2)
    mcount = np.repeat((2 * el_full + 1.)[np.newaxis, :], len(k_par), axis=0)
    bins_mcount, _, _ = stats.binned_statistic(k_full.flatten(), mcount.flatten(), 'sum', bins)

    return bins_mcount * f_sky


def get_1d_power_spectra(ps2d, k_per, k_par, ll, f_sky, bins, k_par_start=0):
    k = np.sqrt(k_per ** 2 + k_par[k_par_start:, np.newaxis] ** 2)

    k_mean, _, _ = stats.binned_statistic(k.flatten(), k.flatten(), 'mean', bins)
    k_norm = k_mean ** 3 / (2 * np.pi ** 2)

    dsp, bins, _ = stats.binned_statistic(k.flatten(), ps2d[k_par_start:].flatten(), 'mean', bins)
    dsp = dsp * k_norm

    bins_mcount = get_1d_ps_sample_count(k_par, k_per, np.unique(ll), bins, f_sky)

    dsp_err = np.sqrt(2 / bins_mcount) * dsp

    return dsp, dsp_err, k_mean


def plot_power_spectra(ps, ll, freqs, ax=None, title=None, kper=None,
                       kper_only=True, fill_gap=True, **kargs):
    if ax is None:
        fig, ax = plt.subplots()

    if kper is not None:
        extent = (min(kper), max(kper), min(freqs), max(freqs))
        ax.set_xlabel('$k_{\\bot} (h cMpc^{-1})$')
        if not kper_only:
            axb = ax.twiny()
            axb.set_xlim(min(ll), max(ll))
            axb.set_xlabel('l')
    else:
        extent = (min(ll), max(ll), min(freqs), max(freqs))
        ax.set_xlabel('l')

    if fill_gap:
        ps = fill_gaps(ps, get_gaps(freqs * 1e6))

    cbs = plotutils.ColorbarSetting(plotutils.ColorbarOutterPosition())

    im_mappable = ax.imshow(ps, aspect='auto', norm=plotutils.LogNorm(),
                            extent=extent, **kargs)
    cbs.add_colorbar(im_mappable, ax)
    ax.set_ylabel("Frequency (MHz)")

    if not kper_only:
        # Hack to fix the second axes (http://stackoverflow.com/questions/34979781)
        fig.canvas.draw()
        axb.set_position(ax.get_position())

    if title is not None:
        ax.set_title(title)


def plot_2d_power_spectra(ps, ll, delay, ax=None, title=None, kper=None, kpar=None,
                          kper_only=True, log_norm=True, **kargs):
    if ax is None:
        fig, ax = plt.subplots()
    pad = '5%'

    if kper is not None:
        extent = (min(kper), max(kper), min(kpar), max(kpar))
        ax.set_xlabel('$k_{\\bot}\,\mathrm{[h\,cMpc^{-1}]}}$')
        ax.set_ylabel('$k_{\parallel}\,\mathrm{[h\,cMpc^{-1}]}}$')
        if not kper_only:
            axb = ax.twiny()
            axb.set_xlim(min(ll), max(ll))
            axc = ax.twinx()
            axc.set_ylim(min(delay), max(delay))
            axb.set_xlabel('l')
            axc.set_ylabel("Delay (us)")
            pad = '15%'
    else:
        extent = (min(ll), max(ll), min(delay), max(delay))
        ax.set_xlabel('l')
        ax.set_ylabel("Delay (us)")

    if log_norm:
        norm = plotutils.LogNorm()
    else:
        norm = None

    cbs = plotutils.ColorbarSetting(plotutils.ColorbarOutterPosition(pad=pad))

    im_mappable = ax.imshow(ps, aspect='auto', norm=norm,
                            extent=extent, **kargs)
    cbs.add_colorbar(im_mappable, ax)

    if not kper_only:
        # Hack to fix the second axes (http://stackoverflow.com/questions/34979781)
        fig.canvas.draw()
        axb.set_position(ax.get_position())
        axc.set_position(ax.get_position())

    if title is not None:
        ax.set_title(title)


def plot_1d_power_spectra(ps2d_rec, ps2d_rec_v, ps2d_sub, k_par, k_per, bins,
                          ll, fwhm, nsigma=2, ax=None, title=None, k_par_start=0, diff_bias=None):
    if ax is None:
        fig, ax = plt.subplots()

    # kbin = np.array([a + (b - a) / 2. for a, b in nputils.pairwise(bins)])

    dsp_rec, dsp_rec_err, k_mean = get_1d_power_spectra(ps2d_rec, k_per, k_par, ll, fwhm, bins, k_par_start)

    dsp_sub, dsp_sub_err, _ = get_1d_power_spectra(ps2d_sub, k_per, k_par, ll, fwhm, bins, k_par_start)

    dsp_v, dsp_v_err, _ = get_1d_power_spectra(ps2d_rec_v, k_per, k_par, ll, fwhm, bins, k_par_start)

    dsp_diff, _, _ = get_1d_power_spectra(ps2d_sub - ps2d_rec_v, k_per, k_par, ll, fwhm, bins, k_par_start)

    # ax.errorbar(k_mean, dsp_rec, yerr=nsigma * dsp_rec_err, marker='+', label='I', c=plotutils.green)

    if diff_bias is not None:
        dsp_diff += diff_bias

    ax.errorbar(k_mean, dsp_sub, yerr=nsigma * dsp_sub_err, marker='+',
                label='I - model', c=plotutils.blue)
    ax.errorbar(k_mean, dsp_v, yerr=nsigma * dsp_v_err, marker='+', label='V', c=plotutils.red)
    ax.errorbar(k_mean, dsp_diff, yerr=nsigma * np.sqrt(dsp_sub_err ** 2 + dsp_v_err ** 2),
                marker='o', label='(I - model) - V', c=plotutils.orange, mec=plotutils.orange, ms=4)

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_ylabel('$\Delta^2 (k)\,[\mathrm{mK^2}]$')
    ax.set_xlabel('$k\,[\mathrm{h\,cMpc^{-1}}]$')

    ax.set_xlim(bins.min(), bins.max())
    # ax.legend(loc=4)

    if title is not None:
        ax.set_title(title)

    return k_mean, dsp_diff


def alm_to_cartmap(alm, ll, mm, nside, theta_max, n):
    # x = theta_max * np.linspace(-1, 1, n)
    # y = theta_max * np.linspace(-1, 1, n)
    x = 2 * theta_max / n * np.arange(-n / 2., n / 2.)
    y = 2 * theta_max / n * np.arange(-n / 2., n / 2.)
    xx, yy = np.meshgrid(x, y)
    _, phis, thetas = util.cart2sph(xx, yy, np.ones_like(x))

    idx = hp.ang2pix(nside, thetas, phis)

    _map = util.fast_alm2map(alm, ll, mm, nside)
    return _map[idx]


def plot_cart_map(cart_map, theta_max, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    cbs = plotutils.ColorbarSetting(plotutils.ColorbarOutterPosition())

    extent = np.degrees(np.array([-theta_max, theta_max, -theta_max, theta_max]))

    im_mappable = ax.imshow(cart_map, extent=extent)
    cbs.add_colorbar(im_mappable, ax)
    ax.set_xlabel('DEC (deg)')
    ax.set_ylabel('RA (deg)')

    if title is not None:
        ax.set_title(title)
