import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext = Extension('psparse',
                ['psparse.pyx', 'cs_gaxpy.c'],
                extra_compile_args=['-fopenmp', '-O3', '-ffast-math'],
                include_dirs=[np.get_include(), '.'],
                extra_link_args=['-fopenmp'])

setup(name='psparse',
      version='0.1',
      cmdclass={'build_ext': build_ext},
      ext_modules=[ext])
