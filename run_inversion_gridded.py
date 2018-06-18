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

from astropy import constants as const
from astropy.io import fits as pyfits


USAGE = '''Run the ML spherical harmonics inversion

Usage: run_inversion.py name

Additional options:
--config, -c: configuration file to be used instead of the default config.py
'''


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
    else:
        noiserms = config.noiserms

    return freq, uu, vv, vis, noiserms


def do_inversion_gridded(config, result_dir):
    ll, mm = sphimg.get_out_lm_sampling(config)

    print '\nBuilding the global YLM matrix...'
    freq, uu, vv, Vobs, noiserms = read_gridded_visbilities(config.gridded_fits[0], config)
    ru, uphis, uthetas = util.cart2sph(uu, vv, np.zeros_like(uu))

    global_ylm = util.SplittedYlmMatrix(ll, mm, uphis, uthetas, ru,
                                        config.cache_dir, keep_in_mem=config.keep_in_mem)

    uthetas, uphis, ru = global_ylm[0].thetas, global_ylm[0].phis, global_ylm[0].rb
    uu, vv, ww = util.sph2cart(uthetas, uphis, ru)

    alms_rec = []
    freqs = []

    pt = util.progress_tracker(len(config.gridded_fits))

    for i, file in enumerate(sorted(config.gridded_fits)):
        freq, _, _, Vobs, noiserms = sphimg.read_gridded_visbilities(file, config)

        Vobs = Vobs[global_ylm[0].sort_idx_cols]
        config.noiserms = noiserms[global_ylm[0].sort_idx_cols]

        freqs.append(freq)

        print "\nProcessing frequency %.3f MHz (%s)" % (freq * 1e-6, pt(i))

        result_freq_dir = os.path.join(result_dir, 'freq_%s' % i)
        os.mkdir(result_freq_dir)

        t = time.time()
        print "Building transformation matrix..."
        trm = util.get_alm2vis_matrix(ll, mm, global_ylm, 1, order='F')
        print "Done in %.2f s" % (time.time() - t)

        # plotting the visibilities
        if config.do_plot:
            sphimg.plot_visibilities(uu, vv, ww, Vobs,
                                     os.path.join(result_freq_dir, 'visibilities.pdf'))

            sphimg.plot_2d_visibilities(uu, vv, Vobs,
                                        os.path.join(result_freq_dir, 'visibilities_2d.pdf'))

            sphimg.plot_2d_visibilities(uu, vv, config.noiserms,
                                        os.path.join(result_freq_dir, 'vis_error_2d.pdf'))

        alm_rec, alm_rec_noise, Vrec, cov_error = sphimg.alm_ml_inversion(ll, mm, Vobs, uphis, uthetas, i, trm, config)

        jy2k = ((1e-26 * (const.c.value / freq) ** 2) / (2 * const.k_B.value))

        # Convert back to Kelvin
        alm_rec = alm_rec * jy2k
        alm_rec_noise = alm_rec_noise * jy2k
        cov_error = cov_error * jy2k

        # Saving full alm before post-processing
        sphimg.save_alm_rec(result_freq_dir, ll, mm, alm_rec, alm_rec_noise, cov_error,
                            filename='alm_rec_full.dat')

        print "Post processing..."
        # When w=0, odd l+m modes are not recovered, we compute them by interpolation
        alm_rec, ll2, mm2 = sphimg.alm_post_processing(alm_rec, ll, mm, config)
        alm_rec_noise, _, _ = sphimg.alm_post_processing(alm_rec_noise, ll, mm, config)
        cov_error, _, _ = sphimg.alm_post_processing(cov_error, ll, mm, config)

        vlm_rec = util.alm2vlm(alm_rec, ll2)
        vlm_rec_noise = util.alm2vlm(alm_rec_noise, ll2)

        alms_rec.append(alm_rec)

        sphimg.save_alm_rec(result_freq_dir, ll2, mm2, alm_rec, alm_rec_noise, cov_error)
        sphimg.save_visibilities_rec(result_freq_dir, ru, uphis, uthetas, Vobs, Vrec)

        if config.do_plot:
            print "Plotting result"
            t = time.time()
            # plot output sky
            sphimg.plot_sky_cart(alm_rec, ll2, mm2, config.nside, theta_max=1 * config.fwhm,
                                 savefile=os.path.join(result_freq_dir, 'output_sky.pdf'))

            # plot vlm_rec
            sphimg.plot_vlm_rec_map(ll2, mm2, vlm_rec, 4 * np.pi * cov_error,
                                    os.path.join(result_freq_dir, 'vlm_rec_map.pdf'), vmin=1e-5, vmax=1e-2)

            sphimg.plot_vlm(ll2, mm2, vlm_rec, os.path.join(result_freq_dir, 'vlm_rec.pdf'))
            sphimg.plot_vlm(ll2, mm2, vlm_rec_noise, os.path.join(result_freq_dir, 'vlm_rec_noise.pdf'))

            sphimg.plot_vlm(ll2, mm2, 4 * np.pi * cov_error, os.path.join(result_freq_dir, 'vlm_cov_error.pdf'))

            # plot visibilities
            sphimg.plot_vis_diff(ru, Vobs, Vrec, os.path.join(result_freq_dir, 'vis_minus_vis_rec.pdf'))

            # plot power spectra
            sphimg.plot_rec_power_sepctra(ll2, mm2, alm_rec, config, os.path.join(
                result_freq_dir, 'angular_power_spectra.pdf'))

            sphimg.plot_vis_vs_vis_rec(ru, Vobs, Vrec, os.path.join(result_freq_dir, 'vis_vs_vis_rec.pdf'))

            print "Done in %.2f s" % (time.time() - t)

        if config.do_ft_inv:
            print "\nStarting FT ML inversion ..."
            jybeam2k = (jy2k / (config.ft_inv_res ** 2))

            ml_cart_map_rec = sphimg.ft_ml_inversion(uu, vv, ww, Vobs, config) * jybeam2k

            res = config.ft_inv_res
            umin = config.l_sampling_lmin / (2 * np.pi)
            umax = config.l_sampling_lmax / (2 * np.pi)

            ml_cart_map_rec_bp = util.filter_cart_map(ml_cart_map_rec, res, umin, umax)

            # if config.de_apodize:
            #     apodize_window = np.squeeze(pyfits.getdata(config.apodize_window_file))
            #     nx, ny = apodize_window.shape
            #     x = (np.arange(nx) - nx / 2) * config.apodize_window_res
            #     y = (np.arange(ny) - ny / 2) * config.apodize_window_res

            #     interp_fct = RectBivariateSpline(x, y, cart_map)

            #     idx = (thetas < min(nx, ny) / 2 * res)
            #     hp_map[idx] = interp_fct.ev(sph_x[idx], sph_y[idx])

            sphimg.save_fits_img(ml_cart_map_rec, res, float(freq) * 1e6, 1, result_freq_dir, 'cart_map_rec.fits')

            sphimg.plot_cart_rec_power_sepctra(ml_cart_map_rec, np.unique(ll2), config,
                                               savefile=os.path.join(result_freq_dir, 'power_spectra_ft_ml.pdf'))

            sphimg.plot_cart_map_rec(ml_cart_map_rec_bp, config, savefile=os.path.join(
                result_freq_dir, 'cart_map_ft_ml.pdf'))

            print "Done FT ML inversion"

    global_ylm.close()

    np.save(os.path.join(result_dir, 'freqs'), freqs)

#    if len(alms_rec) > 1:
#        plot_2d_power_spectra(ll2, mm2, alms_rec, freqs, config, os.path.join(result_dir, 'power_spectra_2d.pdf'),
#                              vmin=1e-3, vmax=1e-1)

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
    sphimg.do_inversion_gridded(config, result_dir)
    profileutils.done(stdout=False, file=os.path.join(result_dir, 'stats.dmp'))


if __name__ == '__main__':
    main()
