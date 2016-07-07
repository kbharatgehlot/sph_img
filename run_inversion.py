import os
import sys
import shutil

import matplotlib as mpl
mpl.use('Agg')

from libwise import scriptshelper as sh

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

    sphimg.do_inversion(config, result_dir)


if __name__ == '__main__':
    main()
