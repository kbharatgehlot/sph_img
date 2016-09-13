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

dct_fct_r_m0 = util.get_dct2
dct_fct_i_m0 = util.get_dct2
dct_fct_r_m1 = util.get_dct2
dct_fct_i_m1 = util.get_dct2

# You might want to use something like dct_dl = np.ceil(np.pi / out_theta_max)
# dct_dl = np.ceil(np.pi / (2.* out_theta_max))
dct_dl = 16

dct_mmax_full_sample = -1

dct_dl_m0 = dct_dl

# using psparse will be faster for ncore > 4, slower otherwise.
use_psparse = True

compute_alm_noise = True

# =================================================================
# General parameters
# =================================================================

n_decimal_tol = 12

cache_dir = 'cache'

keep_in_mem = True
