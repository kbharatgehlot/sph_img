#
# Configuration file for the ML spherical harmonic inversion
#

import os
import util
import numpy as np

# =================================================================
# Input sky. Simulate sky and apply a beam
# =================================================================

freqs_mhz = np.array([110.])

# lmax should be about 3 times nside. nside needs to be a power of 2
lmax = 300
nside = 512

# Sample the input blm
inp_lmax = lmax
inp_mmax = lmax
inp_mmin = 0
inp_lmin = 50
inp_dm = 1
inp_dl = 1
inp_mmax_strip = True

# When W=0, we can not recover odd l+m modes
inp_lm_even_only = False

# beam_type can be one of: gaussian, sinc2, tophat, none
beam_type = 'gaussian'
beam_sinc_n_sidelobe = 1
fwhm = np.radians(10)

out_theta_max = 1 * fwhm
out_mmax_bias = 1

# =================================================================
# Simulate FG using synfast (if add_fg is False)
# =================================================================

synfast_rnd_seed = 125
vis_rnd_seed = None

# The angular power spectrum profile
cl = 5 * (np.arange(lmax + 1) + 1) ** -2.

# Frequencies (in MHz) and spectral power spectra profile
cl_freq = (freqs_mhz / freqs_mhz[0]) ** -0.8

# =================================================================
# FG input from FITS files
# =================================================================

add_fg = False
fg_file = '../fg_standard.fits'
fg_res = np.radians(1.17 / 60.)

# Define the frequencies that should be extracted from the fits file:
# start, stop: first and last slice
# step: should be something like delta_nu / delta_nu_eor
# The numbers of frequencies extractd should match freqs_mhz
fg_freq_step = 4
fg_freq_start = 0
fg_freq_stop = -1

# =================================================================
# EoR signal
# =================================================================

add_eor = False
eor_file = '../eor_standard.fits'
eor_res = np.radians(1.17 / 60.)

# Define the frequencies that should be extracted from the fits file:
# start, stop: first and last slice
# step: should be something like delta_nu / delta_nu_eor
# The numbers of frequencies extractd should match freqs_mhzeor_freq_step = 4
eor_freq_step = 4
eor_freq_start = 0
eor_freq_stop = -1

# =================================================================
# UV coverage
# =================================================================

# uv_type can be one of: polar or lofar
# umax should be of the order of lmax / 2pi.
uv_type = 'polar'
uv_rumin = 10.
uv_rumax = 50.

# polar: Pseudo polar u,v grid. rumax should be of the order of lmax / 2pi.
# nr is the numbers total of baselines. The actual numbers for each frequencies will vary
polar_nr = 60.
polar_nphi = 200.
polar_rnd_w = True
polar_rnd_ru = True

# cartesian: Cartesian u,v grid. rumax should be of the order of lmax / 2pi.
# n is 2 * umax / du.
cart_du = 1.
cart_rnd_w = True

# umax should be of the order of lmax / 2pi.
lofar_dec_deg = 90
lofar_hal = -6
lofar_har = 6
lofar_timeres = 800.
lofar_include_conj = True

# For uv_type='gridded', you need to provide a weighting file
gridded_weights = ''

# =================================================================
# sampled lm modes that will be recovered
# =================================================================

out_lmax = lmax
out_mmax = lmax
out_mmin = 0
out_lmin = inp_lmin
out_dm = 1

# When W=0, we can not recover odd l+m modes
out_lm_even_only = False

# You might want to use out_dl = np.ceil(np.pi / out_theta_max)
out_dl = 1

# strip mmax < sin(theta_max) * l
out_mmax_strip = True

# =================================================================
# FT Inversion parameters
# =================================================================

do_ft_inv = False
ft_inv_nx = 40
ft_inv_ny = 40
ft_inv_res = 0.5 * 1 / float(uv_rumax)

# =================================================================
# Inversion parameters
# =================================================================

# Parameters to compute the noise rms per visibility
SEFD = 4000

n_nights = 100

int_time = (24 * 3600.) / polar_nphi  # For polar configuration
# int_time = lofar_timeres  # For lofar configuration

bandwidth = 0.5e6

noiserms = SEFD / (2 * bandwidth * n_nights * int_time) ** 0.5

reg_lambda = 0

cg_tol = 1e-14

cg_maxiter = 5000

# Enable dct sampling
use_dct = True

# You might want to use something like dct_dl = np.ceil(np.pi / out_theta_max)
dct_dl = 5.

# using psparse will be faster for ncore > 4, slower otherwise.
use_psparse = True

compute_alm_noise = True

# =================================================================
# Post processing
# =================================================================

# If W=0, odd l+m modes are not recovered but we can estimates them by interpolation
do_lm_interp = inp_lm_even_only

# even and odd modes are estimated separately, and l-smoothing help making getting a converged solution
# This works by getting ride of high spatial frequency term in l direction
do_l_smoothing = True

# Save only a sampled set of all lm modes
do_l_sampling = True
l_sampling_dl = dct_dl
l_sampling_lmin = out_lmin
l_sampling_lmax = out_lmax

# =================================================================
# General parameters
# =================================================================

n_decimal_tol = 12

cache_dir = 'cache'

keep_in_mem = True
