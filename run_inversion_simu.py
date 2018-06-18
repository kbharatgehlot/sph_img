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


USAGE = '''Run the ML spherical harmonics inversion

Usage: run_inversion.py name

Additional options:
--config, -c: configuration file to be used instead of the default config.py
'''


def do_inversion(config, result_dir):
    nfreqs = len(config.freqs_mhz)

    assert len(config.freqs_mhz) == len(config.cl_freq)

    if config.use_dct and isinstance(config.dct_dl, (list, np.ndarray)):
        assert len(config.dct_dl) == nfreqs, "Lenght of dct_dl should be the same as the number of frequencies"

    full_ll, full_mm, full_alms, fg_alms, eor_alms = sphimg.simulate_sky(config)
    sphimg.plot_sky_cart(full_alms[0], full_ll, full_mm, config.nside, theta_max=config.fwhm,
                         title='Input sky, beam=%.1f deg, lmax=%s' % (np.degrees(config.fwhm), config.lmax),
                         savefile=os.path.join(result_dir, 'input_sky.pdf'))

    inp_alms, inp_ll, inp_mm = sphimg.sample_input_alm(config, full_alms + fg_alms + eor_alms, full_ll, full_mm)
    fg_alms = inp_alms[nfreqs:2 * nfreqs]
    eor_alms = inp_alms[2 * nfreqs:3 * nfreqs]

    rb, uphis, uthetas = sphimg.simulate_uv_cov(config)

    sel_ll, sel_mm = sphimg.get_out_lm_sampling(config)

    # plotting the sampling
    # plot_sampling(inp_ll, inp_mm, sel_ll, sel_mm, os.path.join(result_dir, 'lm_sampling.pdf'))

    print '\nBuilding the global YLM matrix...'
    global_inp_ylm = util.SplittedYlmMatrix(inp_ll, inp_mm, uphis, uthetas, rb,
                                            config.cache_dir, keep_in_mem=config.keep_in_mem)

    if config.out_dl != config.inp_dl or config.out_dm != config.inp_dm \
            or config.out_mmax != config.inp_mmax or config.out_mmax_strip != config.inp_mmax_strip \
            or config.out_theta_max != config.inp_theta_max or config.out_mmax_bias != config.inp_mmax_bias \
            or config.out_lmax != config.inp_lmax:
        global_sel_ylm = util.SplittedYlmMatrix(sel_ll, sel_mm, uphis, uthetas, rb,
                                                config.cache_dir, keep_in_mem=config.keep_in_mem)
    else:
        global_sel_ylm = global_inp_ylm

    alms_rec = []

    pt = util.progress_tracker(len(config.freqs_mhz))

    for i, freq in enumerate(config.freqs_mhz):
        sel_ll, sel_mm = sphimg.get_out_lm_sampling(config)

        print "\nProcessing frequency %s MHz (%s)" % (freq, pt(i))

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

        # title = 'Type: %s, Nvis: %s, Umin: %s, Umax: %s' % (config.uv_type, len(uu),
        #                                                     config.uv_rumin, config.uv_rumax)
        # plot_uv_cov(uu, vv, ww, config, title, os.path.join(result_freq_dir, 'uv_cov.pdf'))

        alm = inp_alms[i]

        # computing the visibilities in Jansky
        jy2k = ((1e-26 * lamb ** 2) / (2 * const.k_B.value))
        print "\nBuilding visibilities..."
        V = sphimg.compute_visibilities(alm / jy2k, inp_ll, inp_mm, uphis, uthetas, i, trm)

        if isinstance(config.noiserms, np.ndarray):
            print "Noise in visibility (mean): %.3f Jy" % np.mean(config.noiserms)
        else:
            print "Noise in visibility: %.3f Jy" % config.noiserms

        np.random.seed(config.vis_rnd_seed + i)
        Vnoise = config.noiserms * np.random.randn(len(V)) + 1j * config.noiserms * np.random.randn(len(V))
        Vobs = V + Vnoise

        if config.simu_vis_only:
            sphimg.save_visibilities(result_freq_dir, ru, uphis, uthetas, V, Vobs, Vobs)
            continue

        # plotting the visibilities
        # plot_pool.apply_async(plot_visibilities, (uu, vv, ww, V, os.path.join(result_freq_dir, 'vis_from_vlm.pdf')))

        # plot_pool.apply_async(plot_2d_visibilities, (uu, vv, Vobs,
        #                                              os.path.join(result_freq_dir, 'visibilities_2d.pdf')))

        # if config.uv_type == 'gridded':
        #     if config.uv_type == 'gridded':
        #         print "Number total of visibilities:", np.sum(config.weights)
        #     g_Vobs = get_gridded_visibilities(config, Vobs, uu, vv) * jy2k
        #     g_V = get_gridded_visibilities(config, V, uu, vv) * jy2k

        #     fig, ax = plt.subplots()
        #     cbs = plotutils.ColorbarSetting(plotutils.ColorbarOutterPosition())
        #     extent = np.array([-5, 5, -5, 5])

        #     dmap_obs = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g_Vobs)))
        #     im_mappable = ax.imshow(dmap_obs.real, extent=extent)
        #     cbs.add_colorbar(im_mappable, ax)
        #     # ax1.set_xlabel('DEC (deg)')
        #     ax.set_ylabel('RA (deg)')
        #     ax.set_title('FFT(Vobs) Stokes I')
        #     fig.savefig(os.path.join(result_dir, 'dirty_map.pdf'))
        #     plt.close(fig)

        #     fig, ax = plt.subplots()
        #     dmap = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g_V)))
        #     print nputils.stat(dmap.real)
        #     print nputils.stat(dmap.real - dmap_obs.real)
        #     im_mappable = ax.imshow(dmap.real - dmap_obs.real, extent=extent)
        #     cbs.add_colorbar(im_mappable, ax)
        #     # ax1.set_xlabel('DEC (deg)')
        #     ax.set_ylabel('RA (deg)')
        #     ax.set_title('FFT(Vobs) Stokes I')
        #     fig.savefig(os.path.join(result_dir, 'diff_dirty_map.pdf'))
        #     plt.close(fig)

        idx = util.get_lm_selection_index(inp_ll, inp_mm, sel_ll, sel_mm)

        sel_alm = alm[idx]

        if global_sel_ylm != global_inp_ylm:
            t = time.time()
            print "Building transformation matrix...",
            sys.stdout.flush()
            sel_ylm = global_sel_ylm.get_chunk(bmin, bmax)
            trm = util.get_alm2vis_matrix(sel_ll, sel_mm, sel_ylm, lamb, order='F')
            print "Done in %.2f s" % (time.time() - t)

            uthetas, uphis, ru = sel_ylm[0].thetas, sel_ylm[0].phis, sel_ylm[0].rb / lamb

        alm_rec, alm_rec_noise, Vrec, cov_error = sphimg.alm_ml_inversion(sel_ll, sel_mm, Vobs, uphis, uthetas,
                                                                          i, trm, config)

        # Convert back to Kelvin
        alm_rec = alm_rec * jy2k
        alm_rec_noise = alm_rec_noise * jy2k
        cov_error = cov_error * jy2k

        # Saving full alm before post-processing
        sphimg.save_alm(result_freq_dir, sel_ll, sel_mm, sel_alm, fg_alms[i][idx],
                        eor_alms[i][idx], alm_rec, alm_rec_noise, cov_error, filename='alm_full.dat')

        t = time.time()
        print "Post processing...",
        sys.stdout.flush()

        alm_rec, _, _ = sphimg.alm_post_processing(alm_rec, sel_ll, sel_mm, config)
        alm_rec_noise, _, _ = sphimg.alm_post_processing(alm_rec_noise, sel_ll, sel_mm, config)
        cov_error, _, _ = sphimg.alm_post_processing(cov_error, sel_ll, sel_mm, config, sampling_alone=True)
        sel_alm, _, _ = sphimg.alm_post_processing(sel_alm, sel_ll, sel_mm, config)
        sel_fg, _, _ = sphimg.alm_post_processing(fg_alms[i][idx], sel_ll, sel_mm, config)
        sel_eor, sel_ll, sel_mm = sphimg.alm_post_processing(eor_alms[i][idx], sel_ll, sel_mm, config)

        print "Done in %.2f s" % (time.time() - t)

        alms_rec.append(alm_rec)

        sphimg.save_alm(result_freq_dir, sel_ll, sel_mm, sel_alm, sel_fg,
                        sel_eor, alm_rec, alm_rec_noise, cov_error)
        sphimg.save_visibilities(result_freq_dir, ru, uphis, uthetas, V, Vobs, Vrec)

        sel_vlm = util.alm2vlm(sel_alm, sel_ll)
        vlm_rec = util.alm2vlm(alm_rec, sel_ll)
        vlm_rec_noise = util.alm2vlm(alm_rec_noise, sel_ll)

        if config.do_plot:
            t = time.time()
            print "Plotting result..."
            # plot vlm vs vlm_rec
            sphimg.plot_vlm_vs_vlm_rec(sel_ll, sel_mm, sel_vlm, vlm_rec,
                                       os.path.join(result_freq_dir, 'vlm_vs_vlm_rec.pdf'))

            # plot vlm vs vlm_rec in a map
            sphimg.plot_vlm_vs_vlm_rec_map(sel_ll, sel_mm, sel_vlm, vlm_rec, 4 * np.pi * cov_error,
                                           os.path.join(result_freq_dir, 'lm_maps_imag.pdf'))

            # plot power spectra
            sphimg.plot_power_spectra(sel_ll, sel_mm, sel_alm, alm_rec, config, alm_rec_noise,
                                      os.path.join(result_freq_dir, 'angular_power_spectra.pdf'))

            # plot vlm diff
            sphimg.plot_vlm_diff(sel_ll, sel_mm, sel_vlm, vlm_rec, 4 * np.pi * cov_error,
                                 os.path.join(result_freq_dir, 'vlm_minus_vlm_rec.pdf'))

            sphimg.plot_vlm_diff(sel_ll, sel_mm, np.zeros_like(vlm_rec), vlm_rec_noise,
                                 4 * np.pi * cov_error,
                                 os.path.join(result_freq_dir, 'vlm_minus_vlm_noise.pdf'))

            # plot visibilities diff
            sphimg.plot_vis_simu_diff(ru, V, Vobs, Vrec, os.path.join(result_freq_dir, 'vis_minus_vis_rec.pdf'))

            # plot output sky
            sphimg.plot_sky_cart_diff(sel_alm, alm_rec, sel_ll, sel_mm, sel_ll, sel_mm, config.nside,
                                      theta_max=config.fwhm, savefile=os.path.join(result_freq_dir, 'output_sky.pdf'))

            # write_gridded_visibilities(result_freq_dir, 'gridded_vis', V, config, freq, 1)

            print "Done in %.2f s" % (time.time() - t)

        if config.do_ft_inv:
            print "\nStarting FT ML inversion ..."
            jybeam2k = (jy2k / (config.ft_inv_res ** 2))

            cart_map = util.alm_to_cartmap(alm, inp_ll, inp_mm, config.ft_inv_res,
                                           config.ft_inv_nx, config.ft_inv_ny,
                                           cache_dir=config.cache_dir)
            ml_cart_map_rec = sphimg.ft_ml_inversion(uu, vv, ww, Vobs, config) * jybeam2k
            ml_cart_map_rec_noise = sphimg.ft_ml_inversion(uu, vv, ww, Vnoise, config) * jybeam2k

            res = config.ft_inv_res
            umin = config.l_sampling_lmin / (2 * np.pi)
            umax = config.l_sampling_lmax / (2 * np.pi)

            cart_map_bp = util.filter_cart_map(cart_map, res, umin, umax)
            ml_cart_map_rec_bp = util.filter_cart_map(ml_cart_map_rec, res, umin, umax)
            # ml_cart_map_rec_noise_bp = util.filter_cart_map(ml_cart_map_rec_noise, res, umin, umax)

            sphimg.save_fits_img(cart_map, res, float(freq) * 1e6, 1, result_freq_dir, 'cart_map_input.fits')
            sphimg.save_fits_img(ml_cart_map_rec, res, float(freq) * 1e6, 1, result_freq_dir, 'cart_map_rec_I.fits')
            sphimg.save_fits_img(ml_cart_map_rec_noise, res, float(
                freq) * 1e6, 1, result_freq_dir, 'cart_map_rec_V.fits')

            sphimg.plot_cart_power_spectra(cart_map, ml_cart_map_rec, sel_ll, config, ml_cart_map_rec_noise,
                                           savefile=os.path.join(result_freq_dir, 'power_spectra_ft_ml.pdf'))

            sphimg.plot_cart_map_diff(cart_map_bp, ml_cart_map_rec_bp, config,
                                      savefile=os.path.join(result_freq_dir, 'cart_map_ft_ml.pdf'))

            print "Done FT ML inversion"

    global_sel_ylm.close()
    global_inp_ylm.close()

    if len(alms_rec) > 1:
        sphimg.plot_mf_power_spectra(sel_ll, sel_mm, alms_rec, config.freqs_mhz, config,
                                     os.path.join(result_dir, 'mf_power_spectra.pdf'))

    print '\nAll done!'


def main():
    sh.init(0.1, USAGE)

    config_file = sh.get_opt_value('config', 'c', default='config.py')
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
    sphimg.do_inversion(config, result_dir)
    profileutils.done(stdout=False, file=os.path.join(result_dir, 'stats.dmp'))


if __name__ == '__main__':
    main()
