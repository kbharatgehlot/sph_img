import os
import imp
import sys
import imp
import time
import shutil
import warnings

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from libwise import plotutils, imgutils, nputils
from libwise import scriptshelper as sh

from scipy.special import sph_harm, jv, jn, sph_jn
from scipy.sparse.linalg import cg, cgs, spilu, lgmres

from uncertainties import ufloat

import healpy as hp

import util
import sphimg


USAGE = '''Run the ML spherical harmonics inversion 

Usage: run_inversion.py name

Additional options:
--config, -c: configuration file to be used instead of the default config.py
'''

def main():
    sh.init(0.1, USAGE)

    config_file = sh.get_opt_value('config', 'c', default='config.py')
    args = sh.get_args(min_nargs=1)
    result_dir = args[0] #+ time.strftime('_%Y%m%d_%H%M%S')

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

    print "\nCreating test directory: %s\n" % result_dir
    os.mkdir(result_dir)
    shutil.copyfile(config_file, os.path.join(result_dir, 'config.py'))

    config = imp.load_source('config', os.path.join(result_dir, 'config.py'))

    sphimg.do_inversion(config, result_dir)


if __name__ == '__main__':
    main()
