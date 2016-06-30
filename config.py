#
# Configuration file for the ML spherical harmonic inversion
#

import os
import numpy as np

# =================================================================
# Input sky. Simulate sky using healpix synfast, and apply a beam
# =================================================================

# lmax should be about 3 times nside. nside needs to be a power of 2
lmax = 180
nside = 256

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
freqs_mhz = np.array([35., 40., 45.])
cl_freq = (freqs_mhz / freqs_mhz[0]) ** -0.8

# beam_type can be one of: gaussian, sinc2, tophat, none
beam_type = 'gaussian' 
fwhm = np.radians(10)

# =================================================================
# UV coverage
# =================================================================

# uv_type can be one of: cart, cart_nu, polar, polar_nu, lofar
# umax should be of the order of lmax / 2pi.
uv_type = 'lofar'

# cart: Cartesian u,v grid, fixed 
cart_umax = 25
cart_n = 50
cart_rnd_w = True

# cart_nu: Cartesian u,v grid with fixed baseline length, i.e variable umax.
cart_bmax = 50

# polar: Pseudo polar u,v grid. rumax should be of the order of lmax / 2pi.
polar_rumin = 2
polar_rumax = 40
polar_nr = 50.
polar_nphi = 100.
polar_rnd_w = True
polar_rnd_ru = True

# polar_nu: fixed baseline length, i.e variable umax.
polar_bmax = 30
polar_bmin = 4

# umax should be of the order of lmax / 2pi.
lofar_freq_mhz = 150
lofar_dec_deg = 90
lofar_hal = -6
lofar_har = 6
lofar_umin = 10
lofar_umax = 40
lofar_min_max_is_baselines = False # If true, the above min and max are for the baselines
lofar_timeres = 800.
lofar_include_conj = True

# =================================================================
# sampled lm modes that will be recovered
# =================================================================

out_theta_max = 1 * fwhm

out_lmax = lmax
out_mmax = lmax
out_mmin = 0
out_lmin = inp_lmin
out_dm = 1

# You might want to use out_dl = np.ceil(np.pi / out_theta_max)
out_dl = 1

# to catch the (-i)**l, you might want to sample with 4 consecutives coeff in l
out_four_consective = False

# strip mmax < sin(theta_max) * l
out_mmax_strip = True

# The first modes in m are usually the ones with the most power, because of the 
# effect of the beam, you might want to have full sampling for them
out_mmax_full_sample = 4

# =================================================================
# Inversion parameters
# =================================================================

noiserms = 1e-5

reg_lambda = 0

cg_tol = 1e-14

cg_maxiter = 10000

# =================================================================
# General parameters
# =================================================================

n_decimal_tol = 12

cache_dir = 'cache'

keep_in_mem = True
