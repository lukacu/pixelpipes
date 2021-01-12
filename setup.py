import sys
import os
import glob
import setuptools

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

try:
    from torch.utils.cpp_extension import include_paths, library_paths
    with_torch = True
except ImportError:
    with_torch = False


__version__ = '0.0.1'

class get_pybind_include(object):
    def __str__(self):
        import pybind11
        return pybind11.get_include()

class get_numpy_include(object):
    def __str__(self):
        import numpy
        return numpy.get_include()

include_dirs=[
    get_pybind_include(),
    get_numpy_include()
]

runtime_dirs = []
library_dirs = []

if os.name == "nt":
    #opencv_version = "420"
    opencv_version = "310"
    libraries = ["opencv_world{}".format(opencv_version)]
else:
    libraries = ["opencv_core", "opencv_imgcodecs"]
    
define_macros = []

if with_torch:
    import torch.utils.cpp_extension
    include_dirs.extend(include_paths(cuda=False))
#    library_dirs.extend(library_paths(cuda=False))

    torch_path = os.path.dirname(os.path.dirname(os.path.abspath(torch.utils.cpp_extension.__file__)))
    library_dirs.append(os.path.join(torch_path, 'lib'))
    define_macros.append(("__PP_PYTORCH", None))
    define_macros.append(("_GLIBCXX_USE_CXX11_ABI", "0"))
    libraries.extend(['c10', 'torch', 'torch_python'])

if "CONDA_PREFIX" in os.environ:
    conda_path = os.environ["CONDA_PREFIX"]
    library_dirs.append(os.path.join(conda_path, "lib"))
    include_dirs.append(os.path.join(conda_path, "include"))
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

        for ext in self.extensions:
            ext.define_macros = getattr(ext, "define_macros", []) + [('VERSION_INFO', '"{}"'.format(self.distribution.get_version()))]
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


setup(
    name='Pipeline for image augmentation',
    version=__version__,
    author='Luka Cehovin Zajc',
    author_email='luka.cehovin@gmail.com',
    url='https://github.com/lukacu/pixelpipes',
    description='Infinite data streams for deep learning',
    long_description='',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    install_requires=[
        "numpy>=1.19",
        "bidict>=0.21",
        "attributee"
    ],
    python_requires='>=3.6',
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)