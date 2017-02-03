#!/usr/bin/env python

import os
import sys
import glob
import shutil

import numpy as np

sys.path.append(os.path.expanduser('~/Developpement/Workspace/ska/sph_img/code'))

import util
import sphimg

from libwise import scriptshelper as sh


def do_l_sampling(ll, mm, dl):
    ll2, mm2 = util.get_lm(ll.max(), lmin=ll.min() + dl / 2, dl=dl, mmax=mm.max())
    mmax = np.zeros(ll.max() + 1)
    mmax[np.unique(ll)] = np.array([mm[ll == l_].max() for l_ in np.unique(ll)])
    ll2, mm2 = util.strip_mm(ll2, mm2, lambda l: mmax[l])
    idx = util.get_lm_selection_index(ll, mm, ll2, mm2)

    return ll2, mm2, idx


sh.init(0.1, 'reprocess_alm.py config_file input_data_dir output_data_dir')

config_file, input_data_dir, output_data_dir = sh.get_args(min_nargs=3)

assert os.path.isfile(config_file)
assert os.path.exists(input_data_dir)

if not os.path.exists(output_data_dir):
    os.mkdir(output_data_dir)

shutil.copy(config_file, os.path.join(output_data_dir, 'config_reprocess.py'))
config_reprocess = sphimg.get_config(output_data_dir, filename='config_reprocess.py')

print 'Saving reprocessed data in %s' % output_data_dir

key_fct = lambda a: int(a.split('_')[-1])
freq_dirs = sorted(glob.glob(os.path.join(input_data_dir, 'freq_*')), key=key_fct)

pr = util.progress_report(len(freq_dirs))
for i, freq_dir in enumerate(freq_dirs):
    pr(i)
    ll, mm, alm, alm_fg, alm_eor, alm_rec, alm_rec_noise, cov_error = sphimg.load_alm(freq_dir, filename='alm_full.dat')
    ll = ll.astype(int)
    mm = mm.astype(int)

    print "Post processing..."
    alm_rec, _, _ = sphimg.alm_post_processing(alm_rec, ll, mm, config_reprocess)
    alm_rec_noise, _, _ = sphimg.alm_post_processing(alm_rec_noise, ll, mm, config_reprocess)
    cov_error, _, _ = sphimg.alm_post_processing(cov_error, ll, mm, config_reprocess, sampling_alone=True)
    alm, _, _ = sphimg.alm_post_processing(alm, ll, mm, config_reprocess)
    alm_fg, _, _ = sphimg.alm_post_processing(alm_fg, ll, mm, config_reprocess)
    alm_eor, ll, mm = sphimg.alm_post_processing(alm_eor, ll, mm, config_reprocess)

    output_freq_dir = os.path.join(output_data_dir, os.path.basename(freq_dir))

    if not os.path.exists(output_freq_dir):
        os.mkdir(output_freq_dir)

    sphimg.save_alm(output_freq_dir, ll, mm, alm, alm_fg, alm_eor, alm_rec, alm_rec_noise, cov_error)

    shutil.copy(os.path.join(freq_dir, 'visibilities.dat'),
                os.path.join(output_freq_dir, 'visibilities.dat'))

shutil.copy(os.path.join(input_data_dir, 'config.py'), os.path.join(output_data_dir, 'config.py'))

print 'All done'
