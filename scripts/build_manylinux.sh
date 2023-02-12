#!/bin/bash
set -e -u

auditwheel -V

: ${WHEEL_ROOT:=/io/dist}
: ${PLAT:=manylinux2014_x86_64}

function exists_in_list() {
    LIST=$1
    DELIMITER=$2
    VALUE=$3
    echo $LIST | tr "$DELIMITER" '\n' | grep -F -q -x "$VALUE"
}

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel -v repair "$wheel" -w "${WHEEL_ROOT}"
    fi
}

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    #"${PYBIN}/pip" install -r /io/dev-requirements.txt
    PYDIST=`basename $(dirname $PYBIN)`
    if exists_in_list "$SKIP" "," "$PYDIST"; then
        echo "Skipping $PYDIST"
        continue
    fi
    echo "Building $PYDIST"
    "${PYBIN}/pip" wheel /io/ --no-deps -w ${WHEEL_ROOT}
done

# Bundle external shared libraries into the wheels
# TODO: this does not work, includes already included libraries
for whl in ${WHEEL_ROOT}/*.whl; do
    repair_wheel "$whl"
done

# Install packages and test
#for PYBIN in /opt/python/*/bin/; do
#    "${PYBIN}/pip" install pixelpipes --no-index -f /io/wheelhouse
#    (cd "$HOME"; "${PYBIN}/nosetests" pymanylinuxdemo)
#done
