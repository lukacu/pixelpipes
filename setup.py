import sys
import os
import glob
import setuptools
from typing import Callable

from setuptools_dso import DSO, Extension, setup, build_ext

__version__ = '0.0.3'

platform = os.getenv("PYTHON_PLATFORM", sys.platform)

root = os.path.abspath(os.path.dirname(__file__))

if platform.startswith('linux'):
    library_prefix = 'lib'
    library_suffix = '.so'
elif platform in ['darwin']:
    library_prefix = 'lib'
    library_suffix = '.dylib'
elif platform.startswith('win'):
    library_prefix = ''
    library_suffix = '.dll'

include_dirs=[
    os.path.join(root, "pixelpipes", "core"),
    os.path.join(root, "pixelpipes", "geometry"),
    os.path.join(root, "pixelpipes", "image")
]
library_dirs = []
runtime_dirs = []

import pybind11
include_dirs.append(pybind11.get_include())

import numpy
include_dirs.append(numpy.get_include())

#from torch.utils import cpp_extension as torch_extension
#include_dirs.extend(torch_extension.include_paths())
#torch_path = os.path.dirname(os.path.dirname(os.path.abspath(torch_extension.__file__)))
#library_dirs.append(os.path.join(torch_path, 'lib'))

if platform.startswith('win'):
    #opencv_version = "420"
    opencv_version = "310"
    libraries = ["opencv_world{}".format(opencv_version)]
else:
    libraries = ["opencv_imgcodecs", "opencv_core", "opencv_imgproc"]
    
define_macros = []

#define_macros.append(("_GLIBCXX_USE_CXX11_ABI", "0")) <- this causes problems with opencv_imgcodecs
#libraries.extend(['c10', 'torch', 'torch_python'])

if "PIXELPIPES_DEBUG" in os.environ:
    define_macros.append(("PIXELPIPES_DEBUG", None))

compiler_args = ['-std=c++17', '-pthread']

if "CONDA_PREFIX" in os.environ:
    conda_path = os.environ["CONDA_PREFIX"]
    library_dirs.append(os.path.join(conda_path, "lib"))
    include_dirs.append(os.path.join(conda_path, "include"))
    include_dirs.append(os.path.join(conda_path, "include", "opencv4"))
    if not platform.startswith('win'):
        runtime_dirs.append(os.path.join(conda_path, "lib"))
else:
    include_dirs.append(os.path.join("/usr", "include", "opencv4"))

class SharedLibrary(Extension): 
    pass

lib_core = DSO('pixelpipes.pixelpipes', sources= [
        "src/queue.cpp",
        "src/random.cpp",
        "src/module.cpp",
        "src/operation.cpp",
        "src/types.cpp",
        "src/pipeline.cpp",
        "src/numbers.cpp",
        "src/list.cpp",
        "src/geometry/geometry.cpp",
        "src/geometry/view.cpp",
        "src/geometry/points.cpp",
        "pixelpipes/image/image.cpp",
        "pixelpipes/image/arithmetic.cpp",
        "pixelpipes/image/render.cpp",
        "pixelpipes/image/geometry.cpp",
        "pixelpipes/image/filter.cpp",
        "pixelpipes/image/processing.cpp",
        ],
    extra_compile_args=compiler_args,
    define_macros=define_macros + [("PIXELPIPES_BUILD_CORE", None)],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    runtime_library_dirs = runtime_dirs,
    libraries=list(libraries) + ["dl"],
    language='c++'
)
    
lib_modules = [
    lib_core
]

ext_core = Extension(
        'pixelpipes.pypixelpipes',
        ["src/python/wrapper.cpp"],
        extra_compile_args=['-std=c++17'],
        define_macros=define_macros,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        runtime_library_dirs = runtime_dirs,
        libraries=list(libraries),
        dsos=['pixelpipes.pixelpipes'],
        language='c++'
    )


# Sort input source files to ensure bit-for-bit reproducible builds
# (https://github.com/pybind/python_example/pull/53)
ext_modules = [
    ext_core,
    ext_geometry,
    ext_image
]

setup(
    name='pixelpipes',
    version=__version__,
    author='Luka Cehovin Zajc',
    author_email='luka.cehovin@gmail.com',
    url='https://github.com/lukacu/pixelpipes',
    description='Infinite data streams for deep learning',
    long_description='',
    ext_modules=ext_modules,
    x_dsos=lib_modules,
    packages=setuptools.find_packages(),
    include_package_data=True,
    setup_requires=["pybind11>=2.5.0", "numpy>=1.20"],
    install_requires=[
        "numpy>=1.20",
        "bidict>=0.21",
        "intbitset>=2.4",
        "attributee>=0.1.7"
    ],
    extras_require = {
        'torch': ['torch']
    },
    python_requires='>=3.6',
    zip_safe=False,
)
