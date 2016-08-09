#
# Configuration file for the ML spherical harmonic inversion
#

import os
import util
import numpy as np

# =================================================================
# Input sky. Simulate sky using healpix synfast, and apply a beam
# =================================================================

# lmax should be about 3 times nside. nside needs to be a power of 2
lmax = 100
nside = 128

synfast_rnd_seed = 125

# Sample the input blm
inp_lmax = lmax
inp_mmax = lmax
inp_mmin = 0
inp_lmin = 20
inp_dm = 1
inp_dl = 1
inp_mmax_strip = True

# The angular power spectrum profile
cl = (np.arange(lmax + 1) + 1) ** -2.

# Frequencies (in MHz) and spectral power spectra profile
freqs_mhz = np.array([110.])
cl_freq = (freqs_mhz / freqs_mhz[0]) ** -0.8

# beam_type can be one of: gaussian, sinc2, tophat, none
beam_type = 'gaussian'
fwhm = np.radians(10)

out_theta_max = 1 * fwhm

# =================================================================
# EoR signal
# =================================================================

add_eor = True
eor_file = '../eor_standard.fits'
eor_res = np.radians(1.17 / 60.)

# step in frequency: should be something like delta_nu / delta_nu_eor
eor_freq_res_n = 4

# =================================================================
# UV coverage
# =================================================================

# uv_type can be one of: polar or lofar
# umax should be of the order of lmax / 2pi.
uv_type = 'polar'
uv_rumin = 2.
uv_rumax = 20.

# polar: Pseudo polar u,v grid. rumax should be of the order of lmax / 2pi.
# nr is the numbers total of baselines. The actual numbers for each frequencies will vary
polar_nr = 20.
polar_nphi = 50.
polar_rnd_w = True
polar_rnd_ru = True

# umax should be of the order of lmax / 2pi.
lofar_freq_mhz = 150
lofar_dec_deg = 90
lofar_hal = -6
lofar_har = 6
lofar_timeres = 800.
lofar_include_conj = True

# =================================================================
# sampled lm modes that will be recovered
# =================================================================

out_lmax = lmax
out_mmax = lmax
out_mmin = 0
out_lmin = inp_lmin
out_dm = 1

# You might want to use out_dl = np.ceil(np.pi / out_theta_max)
out_dl = 1

# strip mmax < sin(theta_max) * l
out_mmax_strip = True

# =================================================================
# Inversion parameters
# =================================================================

noiserms = 1e-5

reg_lambda = 0

cg_tol = 1e-14

cg_maxiter = 5000

# Enable dct sampling
use_dct = True

dct_fct_r_m0 = util.get_dct2
dct_fct_i_m0 = util.get_dct2
dct_fct_r_m1 = util.get_dct2
dct_fct_i_m1 = util.get_dct2

# You might want to use something like dct_dl = np.ceil(np.pi / out_theta_max)
dct_dl = 5.

# Modes m=0,1 are usually not well recovered with dct
dct_mmax_full_sample = -1

# using psparse will be faster for ncore > 4, slower otherwise.
use_psparse = True

# =================================================================
# General parameters
# =================================================================

n_decimal_tol = 12

cache_dir = 'cache'

keep_in_mem = True
