from distutils.command.build import build
import sys
import os
import setuptools
import distutils.log
import distutils.file_util as file_util

from setuptools import Extension, setup
from setuptools.command import build_py

__version__ = '0.0.3'

platform = os.getenv("PYTHON_PLATFORM", sys.platform)

class BuildPyCommand(build_py.build_py):
  """Custom build command."""

  def run(self):
    self.run_command('cmake_build')
    build_py.build_py.run(self)

class CMakeBuildCommand(build_py.build_py):

  description = 'Build CMake libraries'
  user_options = []

  def run(self):
    import subprocess
    import glob
    import os
    """Run command."""
    if self.dry_run:
        return

    build_dir = os.environ.get("BUILD_ROOT", os.path.join(root, 'build'))
    target_dir = os.path.join(self.build_lib, 'pixelpipes')
    self.mkpath(target_dir)
    self.mkpath(build_dir)

    env = dict(**os.environ)

    command = ['cmake', '-DBUILD_PYTHON=OFF', '-DBUILD_INPLACE=OFF', root]
    self.announce(
        'Running command: %s' % str(command),
        level=distutils.log.INFO)
    subprocess.check_call(command, cwd=build_dir, env=env)
    self.announce("Done", level=distutils.log.INFO)

    command = ['cmake', '--build', build_dir, '-j']
    self.announce(
        'Running command: %s' % str(command),
        level=distutils.log.INFO)
    subprocess.check_call(command, env=env)
    self.announce("Done", level=distutils.log.INFO)

    # copy binaries generated with cmake to the package
    for file in glob.glob(os.path.join(build_dir, "libpixelpipes*.*")):
        if os.path.isfile(file):
            file_util.copy_file(file, os.path.join(target_dir, os.path.basename(file)), update=True, verbose=True)

    # copy C++ headers to the
    header_dir =  os.path.join(target_dir, "include", "pixelpipes")
    self.mkpath(header_dir)
    for file in glob.glob(os.path.join(os.path.join(root, "include", "pixelpipes"), "*.hpp")):
        file_util.copy_file(file, os.path.join(header_dir, os.path.basename(file)), update=True, verbose=True)


root = os.path.abspath(os.path.dirname(__file__))

include_dirs = []
library_dirs = []
runtime_dirs = []

import pybind11
include_dirs.append(pybind11.get_include())

print(include_dirs)

import numpy
include_dirs.append(numpy.get_include())

#from torch.utils import cpp_extension as torch_extension
#include_dirs.extend(torch_extension.include_paths())
#torch_path = os.path.dirname(os.path.dirname(os.path.abspath(torch_extension.__file__)))
#library_dirs.append(os.path.join(torch_path, 'lib'))

libraries = ["pixelpipes"]
    
define_macros = []

#define_macros.append(("_GLIBCXX_USE_CXX11_ABI", "0")) <- this causes problems with opencv_imgcodecs
#libraries.extend(['c10', 'torch', 'torch_python'])

if "PIXELPIPES_DEBUG" in os.environ:
    define_macros.append(("PIXELPIPES_DEBUG", None))

compiler_args = ['-std=c++17', '-pthread']

include_dirs.append(os.path.join(root, "include"))
library_dirs.append(os.path.join(root, "pixelpipes")) # inplace build

class SharedLibrary(Extension): 
    pass

ext_core = Extension(
        'pixelpipes.pypixelpipes',
        ["src/python/wrapper.cpp", "src/python/image.cpp"],
        extra_compile_args=['-std=c++17'],
        define_macros=define_macros,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        runtime_library_dirs = runtime_dirs,
        libraries=list(libraries),
        language='c++'
    )

# Sort input source files to ensure bit-for-bit reproducible builds
# (https://github.com/pybind/python_example/pull/53)
ext_modules = [
    ext_core
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
    packages=setuptools.find_packages(),
    include_package_data=True,
    setup_requires=["pybind11>=2.5.0", "numpy>=1.19"],
    install_requires=[
        "numpy>=1.19",
        "bidict>=0.21",
        "intbitset>=2.4",
        "attributee>=0.1.7"
    ],
    extras_require = {
        'torch': ['torch']
    },
    python_requires='>=3.6',
    zip_safe=False,
    cmdclass={
        'cmake_build': CMakeBuildCommand,
        'build_py': BuildPyCommand,
    },
)
