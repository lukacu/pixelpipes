#!/usr/bin/env bash
set -euo pipefail

auditwheel -V

: "${WHEEL_ROOT:=/io/dist}"
: "${PLAT:=manylinux2014_x86_64}"
: "${SKIP:=}"
: "${PYDIST_FILTER:=}"
: "${BUILD_ROOT:=/io/build_wheel/build}"
: "${PIXELPIPES_DEPENDENCY_ROOT:=/io/build_wheel/dependencies}"

RAW_WHEEL_ROOT="${WHEEL_ROOT}/raw"
mkdir -p "${RAW_WHEEL_ROOT}" "${WHEEL_ROOT}"

function exists_in_list() {
    LIST=$1
    DELIMITER=$2
    VALUE=$3
    echo "${LIST}" | tr "${DELIMITER}" '\n' | grep -F -q -x "${VALUE}"
}

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel ${wheel}"
    else
        auditwheel -v repair "${wheel}" -w "${WHEEL_ROOT}"
    fi
}

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    #"${PYBIN}/pip" install -r /io/dev-requirements.txt
    PYDIST=$(basename "$(dirname "${PYBIN}")")
    if [[ -n "${PYDIST_FILTER}" && "${PYDIST}" != "${PYDIST_FILTER}" ]]; then
        echo "Skipping ${PYDIST}"
        continue
    fi
    if exists_in_list "$SKIP" "," "$PYDIST"; then
    echo "Skipping ${PYDIST}"
        continue
    fi
    echo "Building ${PYDIST}"
    BUILD_ROOT="${BUILD_ROOT}/${PYDIST}" \
    PIXELPIPES_DEPENDENCY_ROOT="${PIXELPIPES_DEPENDENCY_ROOT}" \
    PIXELPIPES_FETCH_DEPENDENCIES=OFF \
        "${PYBIN}/pip" wheel /io/ --no-deps -w "${RAW_WHEEL_ROOT}"
done

# Bundle external shared libraries into the wheels
# TODO: this does not work, includes already included libraries
for whl in "${RAW_WHEEL_ROOT}"/*.whl; do
    [ -e "${whl}" ] || continue
    repair_wheel "${whl}"
done

# Install the repaired wheel into an isolated target and exercise its native
# extension and a minimal pipeline from outside the checkout.
if [[ -n "${PYDIST_FILTER}" ]]; then
    PYTHON_BIN="/opt/python/${PYDIST_FILTER}/bin/python"
    wheel=$(find "${WHEEL_ROOT}" -maxdepth 1 -type f \
        -name "*-${PYDIST_FILTER}-*.whl" -print -quit)
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        echo "Missing Python interpreter for ${PYDIST_FILTER}: ${PYTHON_BIN}" >&2
        exit 1
    fi
    if [[ -z "${wheel}" ]]; then
        echo "No repaired wheel found for ${PYDIST_FILTER} in ${WHEEL_ROOT}" >&2
        exit 1
    fi

    SMOKE_ROOT=$(mktemp -d)
    trap 'rm -rf "${SMOKE_ROOT}"' EXIT
    "${PYTHON_BIN}" -m pip install --disable-pip-version-check --no-cache-dir \
        --target "${SMOKE_ROOT}" "${wheel}"
    (
        cd "${SMOKE_ROOT}"
        PYTHONPATH="${SMOKE_ROOT}" "${PYTHON_BIN}" - <<'PY'
import pixelpipes.pypixelpipes
from pixelpipes.compiler import Compiler
from pixelpipes.graph import Constant, Graph, outputs

with Graph() as graph:
    value = Constant(7)
    outputs(value)

pipeline = Compiler().build(graph)
assert pipeline.run(1)[0] == 7
PY
    )
fi
