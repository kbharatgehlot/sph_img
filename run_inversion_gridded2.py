import os
import sys
import time
import shutil

import matplotlib as mpl
mpl.use('Agg')

from libwise import scriptshelper as sh
from libwise import profileutils

import sphimg
import util

import numpy as np
import pandas as pd

from astropy import constants as const
from astropy.io import fits as pyfits


USAGE = '''Run the ML spherical harmonics inversion

Usage: run_inversion.py name

Additional options:
--config, -c: configuration file to be used instead of the default config.py
'''


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


def read_gridded_visbilities(filename, config):
    dirname = os.path.dirname(filename)
    basename = '_'.join(os.path.basename(filename).split('_')[:-1])
    g_real = os.path.join(dirname, basename + '_GR.fits')
    g_imag = os.path.join(dirname, basename + '_GI.fits')
    weighting = os.path.join(dirname, basename + '_W.fits')

    assert os.path.exists(g_real) and os.path.exists(g_imag)

    f_g_real = pyfits.open(g_real)
    f_g_imag = pyfits.open(g_imag)
    freq = f_g_real[0].header['CRVAL3'] / float(1e6)  # in MHz
    du = f_g_real[0].header['CDELT1']
    dv = f_g_real[0].header['CDELT2']
    nu = f_g_real[0].header['NAXIS1']
    nv = f_g_real[0].header['NAXIS2']
    nfreq = f_g_real[0].header['NAXIS3']
    delnu = f_g_real[0].header['CDELT3'] / float(1e6)  # in MHz

    print '\nFreq=%s MHz du=%s dv=%s nu=%s nv=%s nfreq=%s delnu=%s MHz\n' % (freq, du, dv, nu, nv, nfreq, delnu)

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
        freq = freq + i * delnu
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
            noiserms = (config.SEFD / np.sqrt(2. * delnu * config.Int_time)) * (1. / np.sqrt(w))
        else:
            noiserms = config.noiserms

        noiserms_freq.append(noiserms)

    return freqs, uu_freq, vv_freq, vis_freq, noiserms_freq


def do_inversion_gridded(config, result_dir):
    config.out_lm_even_only = False
    ll2, mm2 = sphimg.get_out_lm_sampling(config)
    config.out_lm_even_only = True

    ll, mm = sphimg.get_out_lm_sampling(config)

    print '\nBuilding the global YLM matrix...'

    freqs, uufreq, vvfreq, Vobsfreq, noisermsfreq = read_gridded_visbilities(config.gridded_fits[0], config)

    # Read uv values for 1st channel

    uu = uufreq[0]
    vv = vvfreq[0]

    ru, uphis, uthetas = util.cart2sph(uu, vv, np.zeros_like(uu))

    global_ylm = util.SplittedYlmMatrix(ll, mm, uphis, uthetas, ru,
                                        config.cache_dir, keep_in_mem=config.keep_in_mem)

    uthetas, uphis, ru = global_ylm[0].thetas, global_ylm[0].phis, global_ylm[0].rb
    uu, vv, ww = util.sph2cart(uthetas, uphis, ru)

    alms_rec = []

    for i in range(len(freqs)):
        freq = freqs[i]
        Vobs = Vobsfreq[i]
        noiserms = noisermsfreq[i]

        Vobs = Vobs[global_ylm[0].sort_idx_cols]

        config.noiserms = noiserms

        print "\nProcessing frequency %s MHz" % freq

        lamb = const.c.value / (float(freq) * 1e6)  # in m

        print "\nProcessing SB %s m" % lamb

        result_freq_dir = os.path.join(result_dir, 'freq_%s' % i)
        os.mkdir(result_freq_dir)

        t = time.time()
        print "Building transformation matrix..."
        trm = util.get_alm2vis_matrix(ll, mm, global_ylm, 1, order='F')
        print "Done in %.2f s" % (time.time() - t)

        # plotting the visibilities
        sphimg.plot_visibilities(uu, vv, ww, Vobs, os.path.join(result_freq_dir, 'visibilities.pdf'))
        sphimg.plot_2d_visibilities(uu, vv, Vobs, os.path.join(result_freq_dir, 'visibilities_2d.pdf'))

        alm_rec, alm_rec_noise, Vrec, cov_error = sphimg.alm_ml_inversion(ll, mm, Vobs, uphis, uthetas, i, trm, config)

        # When w=0, odd l, +m modes are not recovered, we compute them by interpolation
        alm_rec, ll2, mm2 = sphimg.interpolate_lm_odd(alm_rec, ll, mm, config)
        alm_rec_noise, _, _ = sphimg.interpolate_lm_odd(alm_rec_noise, ll, mm, config)
        cov_error, _, _ = sphimg.interpolate_lm_odd(cov_error, ll, mm, config)

        vlm_rec = util.alm2vlm(alm_rec, ll2)
        vlm_rec_noise = util.alm2vlm(alm_rec_noise, ll2)

        alms_rec.append(alm_rec)

        save_alm_rec(result_freq_dir, ll2, mm2, freq, alm_rec, alm_rec_noise, cov_error)
        save_visibilities_rec(result_freq_dir, ru, uphis, uthetas, Vobs, Vrec)

        print "Plotting result"
        t = time.time()

        # plot vlm_rec
        sphimg.plot_vlm_rec_map(ll2, mm2, vlm_rec, cov_error, os.path.join(result_freq_dir, 'vlm_rec_map.pdf'),
                                vmin=1e-2, vmax=1e3)

        sphimg.plot_vlm(ll2, mm2, vlm_rec, os.path.join(result_freq_dir, 'vlm_rec.pdf'))
        sphimg.plot_vlm(ll2, mm2, vlm_rec_noise, os.path.join(result_freq_dir, 'vlm_rec_noise.pdf'))

        # plot visibilities
        sphimg.plot_vis_diff(ru, Vobs, Vrec, os.path.join(result_freq_dir, 'vis_minus_vis_rec.pdf'))

        # plot output sky
        sphimg.plot_sky_cart(alm_rec, ll2, mm2, config.nside, theta_max=1 * config.fwhm,
                             savefile=os.path.join(result_freq_dir, 'output_sky.pdf'))

        # plot power spectra

        sphimg.plot_rec_power_sepctra(ll2, mm2, alm_rec, lamb, os.path.join(
            result_freq_dir, 'angular_power_spectra.pdf'))

        sphimg.plot_vis_vs_vis_rec(ru, Vobs, Vrec, os.path.join(result_freq_dir, 'vis_vs_vis_rec.pdf'))

        print "Done in %.2f s" % (time.time() - t)

    global_ylm.close()

    print '\nAll done!'


def main():
    sh.init(0.1, USAGE)

    config_file = sh.get_opt_value('config', 'c', default='config_gridded.py')
    args = sh.get_args(min_nargs=1)
    result_dir = args[0]

    print "Result will be stored in:", result_dir

    if os.path.exists(result_dir):
        if not sh.askbool("Result directory already exist. Overwriting ?"):
            sys.exit(0)
        else:
            print "Removing:", result_dir
            shutil.rmtree(result_dir)

    if not os.path.isfile(config_file):
        print "Configuration file %s does not exist" % config_file
        sh.usage(True)

    if 'OMP_NUM_THREADS' not in os.environ:
        print "\nOMP_NUM_THREADS not set"
    else:
        print "Maximum number of threads used:", os.environ['OMP_NUM_THREADS']

    print "\nCreating test directory: %s\n" % result_dir
    os.mkdir(result_dir)
    shutil.copyfile(config_file, os.path.join(result_dir, 'config.py'))

    config = sphimg.get_config(result_dir)

    profileutils.start()
    do_inversion_gridded(config, result_dir)
    profileutils.done(stdout=False, file=os.path.join(result_dir, 'stats.dmp'))


if __name__ == '__main__':
    main()
