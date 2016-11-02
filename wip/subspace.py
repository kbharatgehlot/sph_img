import os
import sys
import time
import shutil

import numpy as np

import matplotlib as mpl
mpl.use('Agg')

from libwise import nputils

import sphimg
import util

path = 'test_lofar_mf_u120_dl6'

result_dir = os.path.join(path, 'subspace_f9_nodct')

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

ll, mm, alm, alm_rec, fisher_error, ru, uphis, uthetas, Vobs, Vrec = sphimg.load_results(path)
config = sphimg.get_config(path)

ll = ll.astype(int)
mm = mm.astype(int)

uthetas = uthetas[-1]
uphis = uphis[-1]
ru = ru[-1]
alm_rec = alm_rec[-1]
alm = alm[-1]
Vobs = Vobs[-1]

alm_rec_300 = alm_rec[ll > 300]
alm_rec_0 = alm_rec[ll <= 300]
ll_300 = ll[ll > 300]
mm_300 = mm[ll > 300]

ll_0 = ll[ll <= 300]
mm_0 = mm[ll <= 300]
alm_0 = alm[ll <= 300]

global_ylm = util.YlmCachedMatrix(ll, mm, uphis, uthetas, config.cache_dir, keep_in_mem=config.keep_in_mem)

V_300 = sphimg.compute_visibilities(alm_rec_300, ll_300, mm_300, uphis, uthetas, ru, global_ylm)

V_0 = Vobs - V_300

config.use_dct = False
alm_rec, Vrec, fisher_error = sphimg.alm_ml_inversion(ll_0, mm_0, V_0, uphis, uthetas, ru, global_ylm, config)

print nputils.stat(alm_rec - alm_0)
print nputils.stat(Vrec - Vobs)

sphimg.save_alm(result_dir, ll_0, mm_0, alm_0, alm_rec, fisher_error)
sphimg.save_visibilities(result_dir, ru, uphis, uthetas, V_0, Vrec)

# ru, uphis, uthetas, V_0, Vrec = sphimg.load_visibilities(result_dir)
# ll_0, mm_0, alm_0, alm_rec, fisher_error = sphimg.load_alm(result_dir)
# ll_0 = ll_0.astype(int)
# mm_0 = mm_0.astype(int)

vlm_rec = util.alm2vlm(alm_rec, ll_0)
vlm_0 = util.alm2vlm(alm_0, ll_0)

print "Plotting result"
# plot vlm vs vlm_rec
sphimg.plot_vlm_vs_vlm_rec(ll_0, mm_0, vlm_0, vlm_rec, os.path.join(result_dir, 'vlm_vs_vlm_rec.pdf'))

# plot vlm vs vlm_rec in a map
sphimg.plot_vlm_vs_vlm_rec_map(ll_0, mm_0, vlm_0, vlm_rec, fisher_error, 
    os.path.join(result_dir, 'lm_maps_imag.pdf'))

# plot power spectra
sphimg.plot_power_sepctra(ll_0, mm_0, alm_0, ll_0, mm_0, alm_rec, config,
    os.path.join(result_dir, 'angular_power_spectra.pdf'))

# plot vlm diff
sphimg.plot_vlm_diff(ll_0, mm_0, vlm_0, vlm_rec, 
    os.path.join(result_dir, 'vlm_minus_vlm_rec.pdf'))

# plot visibilities diff
sphimg.plot_vis_diff(ru, V_0, V_0, Vrec, os.path.join(result_dir, 'vis_minus_vis_rec.pdf'))

# plot output sky
sphimg.plot_sky_cart_diff(alm_0, alm_rec, ll_0, mm_0, ll_0, mm_0, config.nside, theta_max=config.fwhm, 
    savefile=os.path.join(result_dir, 'output_sky.pdf'))

