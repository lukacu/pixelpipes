

ExternalProject_Add(xtl
PREFIX xtl
GIT_REPOSITORY    https://github.com/xtensor-stack/xtl
GIT_TAG           0.7.4
TEST_COMMAND      ""
CMAKE_ARGS   -DCMAKE_INSTALL_PREFIX=${CMAKE_EXTERNALS_DIR}xtensor/
)

ExternalProject_Add(xsimd
PREFIX xsimd
GIT_REPOSITORY    https://github.com/xtensor-stack/xsimd
GIT_TAG           10.0.0
TEST_COMMAND      ""
CMAKE_ARGS   -DCMAKE_INSTALL_PREFIX=${CMAKE_EXTERNALS_DIR}xtensor/
)

ExternalProject_Add(xtensor
PREFIX xtensor
GIT_REPOSITORY    https://github.com/xtensor-stack/xtensor
GIT_TAG           0.24.3
TEST_COMMAND      ""
CMAKE_ARGS   -DCMAKE_INSTALL_PREFIX=${CMAKE_EXTERNALS_DIR}xtensor/
)

ADD_DEPENDENCIES(xtensor xtl xsimd)

SET(XTENSOR_TARGET xtl xsimd xtensor)

SET(XTENSOR_INCLUDE_DIRS "${CMAKE_EXTERNALS_DIR}xtensor/include/")