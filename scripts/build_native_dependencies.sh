#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DEPENDENCY_ROOT=${PIXELPIPES_DEPENDENCY_ROOT:-"${ROOT}/build_wheel/dependencies"}
CONFIGURE_ROOT=${DEPENDENCY_CONFIGURE_ROOT:-"${DEPENDENCY_ROOT}/configure"}
BUILD_WORKERS=${BUILD_WORKERS:-2}

cmake_version=0.0.0
if command -v cmake >/dev/null 2>&1; then
    cmake_version=$(cmake --version | awk 'NR == 1 { print $3 }')
fi
IFS=. read -r cmake_major cmake_minor _ <<< "${cmake_version}"
if ((cmake_major < 3 || (cmake_major == 3 && cmake_minor < 15))); then
    CMAKE_PYTHON=${CMAKE_PYTHON:-/opt/python/cp310-cp310/bin/python}
    "${CMAKE_PYTHON}" -m pip install --cache-dir "${DEPENDENCY_ROOT}/pip-cache" "cmake>=3.15,<4"
    export PATH="$(dirname "${CMAKE_PYTHON}"):${PATH}"
fi

cmake -S "${ROOT}" -B "${CONFIGURE_ROOT}" \
    -DBUILD_PYTHON=OFF \
    -DBUILD_TOOL=OFF \
    -DBUILD_INPLACE=OFF \
    -DBUILD_DEBUG=OFF \
    -DPIXELPIPES_FETCH_DEPENDENCIES=ON \
    -DPIXELPIPES_DEPENDENCY_ROOT="${DEPENDENCY_ROOT}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${DEPENDENCY_ROOT}/installed"

cmake --build "${CONFIGURE_ROOT}" \
    --target xtensor zlib opencv lodepng \
    --parallel "${BUILD_WORKERS}"

# lodepng is the one dependency built as a top-level target; the other
# ExternalProject dependencies install into the shared prefix as they build.
cmake --install "${CONFIGURE_ROOT}" --component dependencies
