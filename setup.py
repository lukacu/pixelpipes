from pixelpipes import __version__
import numpy
import pybind11
import sys
import os
import setuptools
import platform
import distutils.log
import distutils.file_util as file_util
from distutils.command import build
from distutils.cmd import Command

from setuptools import Extension, setup
from setuptools.command import build_py, build_ext

platid = os.getenv("PYTHON_PLATFORM", platform.system()).lower()

if platid == "linux":
    rpath = ["$ORIGIN"]
    libext = ".so"
elif platid == "darwin":
    rpath = ["@loader_path"]
    libext = ".dylib"
    
else:
    rpath = []
    libext = ".dll"


# Override build command
def lib_name(name):
    from distutils.ccompiler import new_compiler
    compiler = new_compiler()
    return compiler.shared_lib_format % (name, libext)


class BuildCommand(build.build):

    def initialize_options(self):
        build.build.initialize_options(self)
        self.build_base = os.environ.get("BUILD_ROOT", "build")


class BuildPyCommand(build_py.build_py):
    """Custom build command."""

    def run(self):
        self.run_command('build_lib')
        build_py.build_py.run(self)


class BuildExtCommand(build_ext.build_ext):
    """Custom build command."""

    def run(self):

        target_dir = os.path.join(os.environ.get("BUILD_ROOT", os.path.dirname(
            self.build_lib)), os.path.basename(self.build_lib), "pixelpipes")

        self.library_dirs.append(target_dir)

        if not self.rpath:
            self.rpath = rpath
        build_ext.build_ext.run(self)

    def build_extensions(self):
    
        ct = getattr(self.compiler, 'compiler_type', None) #self.compiler.compiler_type
        
        if ct == "msvc":
           for e in self.extensions:
               e.extra_compile_args += ['/std:c++17']
        elif ct == "unix":
            for e in self.extensions:
                e.extra_compile_args += ['-std=c++17', '-pthread']            
        else:
            print("Unknown compiler %s" % ct)
          
        build_ext.build_ext.build_extensions(self)


class CMakeBuildCommand(Command):

    description = 'Build core libraries using CMake'
    user_options = []

    description = "\"build\" pure Python modules (copy to build directory)"

    user_options = [
        ('build-lib=', 'd', "directory to \"build\" (copy) to"),
        ('inplace', 'i', "compile to source structure"),
    ]

    boolean_options = ['inplace']

    def initialize_options(self):
        self.build_lib = None
        self.inplace = None

    def finalize_options(self):
        self.set_undefined_options('build',
                                   ('build_lib', 'build_lib'))

    def run(self):
        import subprocess
        import glob
        import os

        build_dir = os.environ.get(
            "BUILD_ROOT", os.path.dirname(self.build_lib))
        target_dir = os.path.join(os.environ.get("BUILD_ROOT", os.path.dirname(
            self.build_lib)), os.path.basename(self.build_lib), "pixelpipes")
        
        self.mkpath(target_dir)
        self.mkpath(build_dir)

        env = dict(**os.environ)

        cmake_args = ["-DBUILD_PYTHON=OFF",
                      "-DBUILD_TEST=OFF",
                      "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON"]

        cmake_args += ["-DCMAKE_OSX_DEPLOYMENT_TARGET=12.00"]
        cmake_args += ["-DBUILD_DEBUG=ON"]
        if not self.inplace:
            cmake_args += ['-DBUILD_INPLACE=OFF']
        else:
            cmake_args += ['-DBUILD_INPLACE=ON']

        command = ['cmake', *cmake_args, root]
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

        print(build_dir)
        print(glob.glob(os.path.join(build_dir, lib_name("pixelpipes*"))))
        if not self.inplace:
            # copy binaries generated with cmake to the package
            # TODO: probably include .lib file as well for the main lib on Windows
            for file in glob.glob(os.path.join(build_dir, lib_name("pixelpipes*"))):
                if os.path.isfile(file):
                    file_util.copy_file(file, os.path.join(
                        target_dir, os.path.basename(file)), update=True, verbose=True)

            # copy C++ headers to the
            header_dir = os.path.join(target_dir, "include", "pixelpipes")
            self.mkpath(header_dir)
            for file in glob.glob(os.path.join(os.path.join(root, "include", "pixelpipes"), "**", "*.hpp"), recursive=True):
                file_util.copy_file(file, os.path.join(
                    header_dir, os.path.basename(file)), update=True, verbose=True)


root = os.path.abspath(os.path.dirname(__file__))

include_dirs = []
library_dirs = []
runtime_dirs = []

include_dirs.append(pybind11.get_include())
include_dirs.append(numpy.get_include())

#from torch.utils import cpp_extension as torch_extension
# include_dirs.extend(torch_extension.include_paths())
#torch_path = os.path.dirname(os.path.dirname(os.path.abspath(torch_extension.__file__)))
#library_dirs.append(os.path.join(torch_path, 'lib'))

libraries = ["pixelpipes"]

define_macros = []

# define_macros.append(("_GLIBCXX_USE_CXX11_ABI", "0")) <- this causes problems with opencv_imgcodecs
#libraries.extend(['c10', 'torch', 'torch_python'])

if "PIXELPIPES_DEBUG" in os.environ:
    define_macros.append(("PIXELPIPES_DEBUG", None))


include_dirs.append(os.path.join(root, "include"))
#library_dirs.append(os.path.join(root, "pixelpipes"))  # inplace build
#library_dirs.append(os.path.join(root, "build"))  # custom build

ext_core = Extension(
    'pixelpipes.pypixelpipes',
    ["src/python/wrapper.cpp", "src/python/array.cpp"],
    define_macros=define_macros,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    runtime_library_dirs=runtime_dirs,
    libraries=list(libraries),
    language='c++'
)

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
    packages=setuptools.find_packages(include=["pixelpipes", "pixelpipes.*"]),
    include_package_data=True,
    setup_requires=["pybind11>=2.5.0", "numpy>=1.19"],
    install_requires=[
        "numpy>=1.19",
        "bidict>=0.21",
        "intbitset>=2.4",
        "attributee>=0.1.7"
    ],
    extras_require={
        'torch': ['torch']
    },
    python_requires='>=3.6',
    zip_safe=False,
    cmdclass={
        'build_lib': CMakeBuildCommand,
        'build_py': BuildPyCommand,
        'build_ext': BuildExtCommand,
        "build": BuildCommand
    },
)
