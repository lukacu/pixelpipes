import sys
import os
import glob
import setuptools
from typing import Callable

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

__version__ = '0.0.2'

class get_pybind_include(object):
    def __call__(self):
        import pybind11
        return pybind11.get_include()

class get_numpy_include(object):
    def __call__(self):
        import numpy
        return numpy.get_include()

class get_torch_include(object):
    def __call__(self):
        from torch.utils.cpp_extension import include_paths
        return include_paths()

class get_torch_libraries(object):
    def __call__(self):
        import torch.utils.cpp_extension
        torch_path = os.path.dirname(os.path.dirname(os.path.abspath(torch.utils.cpp_extension.__file__)))
        return os.path.join(torch_path, 'lib')

include_dirs=[
    get_pybind_include,
    get_numpy_include,
    get_torch_include
]

runtime_dirs = []
library_dirs = [
    get_torch_libraries
]

if os.name == "nt":
    #opencv_version = "420"
    opencv_version = "310"
    libraries = ["opencv_world{}".format(opencv_version)]
else:
    libraries = ["opencv_core", "opencv_imgcodecs"]
    
define_macros = []

define_macros.append(("__PP_PYTORCH", None))
define_macros.append(("_GLIBCXX_USE_CXX11_ABI", "0"))
libraries.extend(['c10', 'torch', 'torch_python'])

if "CONDA_PREFIX" in os.environ:
    conda_path = os.environ["CONDA_PREFIX"]
    library_dirs.append(os.path.join(conda_path, "lib"))
    include_dirs.append(os.path.join(conda_path, "include"))
    include_dirs.append(os.path.join(conda_path, "include", "opencv4"))
    if not os.name == "nt":
        runtime_dirs.append(os.path.join(conda_path, "lib"))

ext_modules = [
    Extension(
        'pixelpipes.engine',
        # Sort input source files to ensure bit-for-bit reproducible builds
        # (https://github.com/pybind/python_example/pull/53)
        sorted(glob.glob("pixelpipes/engine/*.cpp")),
        define_macros=define_macros,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        runtime_library_dirs = runtime_dirs,
        libraries=libraries,
        language='c++'
    ),
]


# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    import os
    with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.
    The newer version is prefered over c++11 (when it is available).
    """
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')

        def _realize(c):

            if not isinstance(c, str):
                paths = c()
                if isinstance(paths, str):
                    return [paths]
                elif isinstance(paths, list):
                    return paths
                else:
                    return _realize(paths)
            else:
                return [c]

        for ext in self.extensions:
            ext.define_macros = getattr(ext, "define_macros", []) + [('VERSION_INFO', '"{}"'.format(self.distribution.get_version()))]
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts

            ext.include_dirs = [p for x in ext.include_dirs for p in _realize(x) ]
            ext.library_dirs = [p for x in ext.library_dirs for p in _realize(x) ]
            ext.runtime_library_dirs = [p for x in ext.runtime_library_dirs for p in _realize(x)]

        build_ext.build_extensions(self)


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
    setup_requires=["pybind11>=2.5.0", "numpy>=1.19", "torch"],
    install_requires=[
        "numpy>=1.19",
        "bidict>=0.21",
        "intbitset>=2.4",
        "attributee>=0.1.2"
    ],
    extras_require = {
        'torch': ['torch']
    },
    python_requires='>=3.6',
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
