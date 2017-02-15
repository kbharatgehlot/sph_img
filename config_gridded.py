#
# Configuration file for the ML spherical harmonic inversion
#

import glob
import util
import numpy as np

# =================================================================
# Input Data
# =================================================================

fwhm = np.radians(4)

gridded_fits = glob.glob('/home/flo/data/NCP/L90490/L90490_SAP000_SB15*_uv_003.MS_UL50-250_GR.fits')

uv_rumin = 50.
uv_rumax = 160.

# =================================================================
# sampled lm modes that will be recovered
# =================================================================

# lmax should be about 3 times nside. nside needs to be a power of 2
lmax = 1000
lmin = 300
nside = 1024

out_lmax = lmax
out_lmin = lmin
out_mmax = lmax
out_mmin = 0
out_dm = 1
out_dl = 1

# strip mmax < sin(theta_max) * l
out_mmax_strip = True
out_theta_max = 1 * fwhm

# When w=0, l+m odd modes are always null
out_lm_even_only = True

# =================================================================
# Inversion parameters
# =================================================================

noiserms = 1e-3

reg_lambda = 0

cg_tol = 1e-14

cg_maxiter = 5000

# Enable dct sampling
use_dct = True

# You might want to use something like dct_dl = np.ceil(np.pi / out_theta_max)
# dct_dl = np.ceil(np.pi / (2.* out_theta_max))
dct_dl = 16

# using psparse will be faster for ncore > 4, slower otherwise.
use_psparse = True

compute_alm_noise = True

# =================================================================
# FT Inversion parameters
# =================================================================

# ft_inv_nx & ft_inv_ny should preferably be even
do_ft_inv = False
ft_inv_nx = 40
ft_inv_ny = 40
ft_inv_res = 0.5 * 1 / float(uv_rumax)


# =================================================================
# Post processing
# =================================================================

# If W=0, odd l+m modes are not recovered but we can estimates them by interpolation
do_lm_interp = inp_lm_even_only

# even and odd modes are estimated separately, and l-smoothing help getting a converged solution
# This works by getting ride of high spatial frequency term in l direction. Not needed with do_reduce_fov!
do_l_smoothing = True

# Save only a sampled set of all lm modes
do_l_sampling = True
l_sampling_dl = dct_dl
l_sampling_lmin = out_lmin
l_sampling_lmax = out_lmax

# Reduce FoV. Simply convert alm to map, apply a tophat beam and convert back to alm.
# It is advised to set reduce_fov_lmin & reduce_fov_lmax to the range
do_reduce_fov = False
reduce_fov_theta_max = fwhm
reduce_fov_lmin = out_lmin
reduce_fov_lmax = out_lmax

# De-apodize
do_de_apodize = False
apodize_window_file = ''
apodize_window_res = 0

do_plot = True


# =================================================================
# General parameters
# =================================================================

n_decimal_tol = 12

cache_dir = 'cache'

keep_in_mem = True
