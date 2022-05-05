#!/usr/bin/env bash


IMAGE=quay.io/pypa/manylinux2014_x86_64

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

docker run --rm -ti -v "${SCRIPT_DIR}/..:/io" -v "${SCRIPT_DIR}/build_manylinux.sh:/build.sh" -e "WHEEL_ROOT=/io/build_wheel/dist" -e "BUILD_ROOT=/io/build_wheel/build" $IMAGE sh build.sh
