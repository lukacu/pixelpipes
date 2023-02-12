#!/usr/bin/env bash


IMAGE=quay.io/pypa/manylinux2014_x86_64

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

docker run --rm -ti -v "${SCRIPT_DIR}/..:/io" -v "${SCRIPT_DIR}/build_manylinux.sh:/build.sh" -e "SKIP=cp36-cp36m,cp37-cp37m,pp37-pypy37_pp73" -e "WHEEL_ROOT=/io/build_wheel/dist" -e "BUILD_ROOT=/io/build_wheel/build" -e "BUILD_WORKERS=4" $IMAGE sh build.sh
