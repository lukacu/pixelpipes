"""Setuptools metadata and CMake-backed native build integration."""

import os
import re
import subprocess
import sys

from setuptools import Command, find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution


ROOT = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(ROOT, "pixelpipes", "__init__.py"), encoding="utf-8") as version_file:
    version_match = re.search(
        r"^__version__\s*=\s*['\"]([^'\"]+)['\"]",
        version_file.read(),
        re.MULTILINE,
    )
if version_match is None:
    raise RuntimeError("Could not determine package version")


class CMakeBuildCommand(Command):
    """Build and install every native target through the project CMake build."""

    description = "Build PixelPipes native libraries and Python extension with CMake"
    user_options = [
        ("build-lib=", "d", "directory for the built package"),
        ("inplace", "i", "install native artifacts into the source package"),
    ]
    boolean_options = ["inplace"]

    def initialize_options(self):
        self.build_lib = None
        self.inplace = None

    def finalize_options(self):
        build_py_cmd = self.get_finalized_command("build_py")
        if self.build_lib is None:
            self.build_lib = build_py_cmd.build_lib
        if self.inplace is None:
            self.inplace = getattr(build_py_cmd, "inplace", False)

    def run(self):
        build_dir = os.path.abspath(os.environ.get("BUILD_ROOT", os.path.join(ROOT, "build")))
        if self.inplace:
            install_prefix = ROOT
        else:
            install_prefix = os.path.abspath(self.build_lib)

        self.mkpath(build_dir)
        self.mkpath(install_prefix)

        debug_build = "PIXELPIPES_DEBUG" in os.environ
        cmake_args = [
            "-DCMAKE_BUILD_TYPE={}".format("Debug" if debug_build else "Release"),
            "-DBUILD_PYTHON=ON",
            "-DBUILD_TOOL=OFF",
            "-DBUILD_INPLACE=OFF",
            "-DPIXELPIPES_FETCH_DEPENDENCIES={}".format(
                os.environ.get("PIXELPIPES_FETCH_DEPENDENCIES", "ON")
            ),
            "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON",
            "-DCMAKE_INSTALL_PREFIX={}".format(install_prefix),
            "-DPython3_EXECUTABLE={}".format(sys.executable),
        ]
        if "PIXELPIPES_LODEPNG_SOURCE_DIR" in os.environ:
            cmake_args.append(
                "-DPIXELPIPES_LODEPNG_SOURCE_DIR={}".format(
                    os.environ["PIXELPIPES_LODEPNG_SOURCE_DIR"]
                )
            )
        if "PIXELPIPES_DEPENDENCY_ROOT" in os.environ:
            dependency_root = os.path.abspath(os.environ["PIXELPIPES_DEPENDENCY_ROOT"])
            installed_root = os.path.join(dependency_root, "installed")
            cmake_args.extend([
                "-DPIXELPIPES_DEPENDENCY_ROOT={}".format(dependency_root),
                "-DCMAKE_PREFIX_PATH={};{};{}".format(
                    os.path.join(installed_root, "opencv"),
                    os.path.join(installed_root, "xtensor"),
                    installed_root,
                ),
                "-DOpenCV_DIR={}".format(
                    os.path.join(installed_root, "opencv", "lib", "cmake", "opencv4")
                ),
                "-DPIXELPIPES_ZLIBNG_ROOT={}".format(
                    os.path.join(installed_root, "zlib")
                ),
            ])
        if "PIXELPIPES_TURBOJPEG_SOURCE_DIR" in os.environ:
            cmake_args.append(
                "-DPIXELPIPES_TURBOJPEG_SOURCE_DIR={}".format(
                    os.path.abspath(os.environ["PIXELPIPES_TURBOJPEG_SOURCE_DIR"])
                )
            )
        if debug_build:
            cmake_args.append("-DBUILD_DEBUG=ON")
        else:
            cmake_args.append("-DBUILD_DEBUG=OFF")

        configure = ["cmake", "-S", ROOT, "-B", build_dir, *cmake_args]
        self.announce("Configuring CMake: {}".format(" ".join(configure)))
        subprocess.check_call(configure)

        build = ["cmake", "--build", build_dir]
        workers = int(os.environ.get("BUILD_WORKERS", "0"))
        if workers > 0:
            build.extend(["--parallel", str(workers)])
        elif workers == 0:
            build.append("--parallel")
        self.announce("Building native targets: {}".format(" ".join(build)))
        subprocess.check_call(build)

        install = ["cmake", "--install", build_dir]
        self.announce("Installing native package files: {}".format(" ".join(install)))
        subprocess.check_call(install)


class BuildPyCommand(build_py):
    """Build native artifacts first, then copy the Python package."""

    def run(self):
        self.run_command("build_lib")
        super().run()

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        package_dir = os.path.join(os.path.abspath(self.build_lib), "pixelpipes")
        if os.path.isdir(package_dir):
            for current_dir, _, filenames in os.walk(package_dir):
                outputs.extend(os.path.join(current_dir, filename) for filename in filenames
                               if filename.endswith((".so", ".dylib", ".dll", ".lib", ".hpp")))
        return outputs


class BinaryDistribution(Distribution):
    """Mark wheels as platform-specific despite building via CMake targets."""

    def has_ext_modules(self):
        return True


setup(
    name="pixelpipes",
    version=version_match.group(1),
    author="Luka Cehovin Zajc",
    author_email="luka.cehovin@gmail.com",
    url="https://github.com/lukacu/pixelpipes",
    description="Infinite data streams for deep learning",
    long_description="",
    distclass=BinaryDistribution,
    packages=find_packages(include=["pixelpipes", "pixelpipes.*"]),
    include_package_data=True,
    install_requires=["numpy>=1.23", "bidict>=0.21", "attributee>=0.1.9"],
    python_requires=">=3.8",
    zip_safe=False,
    cmdclass={"build_lib": CMakeBuildCommand, "build_py": BuildPyCommand},
)
