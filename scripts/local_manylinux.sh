#!/usr/bin/env bash


IMAGE=quay.io/pypa/manylinux2014_x86_64

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

docker run --rm \
    -v "${SCRIPT_DIR}/..:/io" \
    -w /io \
    -e "SKIP=cp36-cp36m,cp37-cp37m,pp37-pypy37_pp73,cp313-cp313,cp313-cp313t" \
    -e "WHEEL_ROOT=/io/build_wheel/dist" \
    -e "BUILD_ROOT=/io/build_wheel/build" \
    -e "PIXELPIPES_DEPENDENCY_ROOT=/io/build_wheel/dependencies" \
    -e "BUILD_WORKERS=4" \
    "${IMAGE}" bash /io/scripts/build_native_dependencies.sh

docker run --rm \
    -v "${SCRIPT_DIR}/..:/io" \
    -w /io \
    -e "SKIP=cp36-cp36m,cp37-cp37m,pp37-pypy37_pp73,cp313-cp313,cp313-cp313t" \
    -e "WHEEL_ROOT=/io/build_wheel/dist" \
    -e "BUILD_ROOT=/io/build_wheel/build" \
    -e "PIXELPIPES_DEPENDENCY_ROOT=/io/build_wheel/dependencies" \
    -e "BUILD_WORKERS=4" \
    "${IMAGE}" bash /io/scripts/build_manylinux.sh
