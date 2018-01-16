from distutils.core import setup
from Cython.Build import cythonize
import numpy


print("Numpy Location:"+numpy.get_include())
setup(ext_modules=cythonize(["*.pyx"], annotate=True,
		compiler_directives={
			'optimize.use_switch': True,
			'initializedcheck':False,
			'overflowcheck':False,
			'optimize.unpack_method_calls':True,
			'boundscheck':False,
			'profile': False,
			'infer_types':True,
			'cdivision_warnings': False,
			'cdivision': True,
			'wraparound': False,
			'boundscheck': False},
                include_path=[numpy.get_include()]
),
include_dirs=[numpy.get_include()]
)
