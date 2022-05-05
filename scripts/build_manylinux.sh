#!/bin/bash
set -e -u -x

: ${WHEEL_ROOT:=/io/dist}
: ${PLAT:=manylinux2014_x86_64}

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w ${WHEEL_ROOT}
    fi
}

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    #"${PYBIN}/pip" install -r /io/dev-requirements.txt
    PYDIST=`basename $(dirname $PYBIN)`
    "${PYBIN}/pip" wheel /io/ --no-deps -w ${WHEEL_ROOT}
done

# Bundle external shared libraries into the wheels
for whl in ${WHEEL_ROOT}/*.whl; do
    repair_wheel "$whl"
done

# Install packages and test
#for PYBIN in /opt/python/*/bin/; do
#    "${PYBIN}/pip" install pixelpipes --no-index -f /io/wheelhouse
#    (cd "$HOME"; "${PYBIN}/nosetests" pymanylinuxdemo)
#done
