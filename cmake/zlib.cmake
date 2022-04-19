
ExternalProject_Add(zlib
PREFIX zlib
GIT_REPOSITORY    https://github.com/zlib-ng/zlib-ng
GIT_TAG           c882034d48afc0b32a38e8f7ca63a2e4e91ab42d
CMAKE_ARGS        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/deps -DCMAKE_C_FLAGS=-fPIC
INSTALL_DIR       "${CMAKE_BINARY_DIR}/deps"
TEST_COMMAND      ""
)

add_library(zlib_lib STATIC IMPORTED)
set_target_properties(zlib_lib PROPERTIES IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/deps/${CMAKE_INSTALL_LIBDIR}/libz-ng.a)

add_dependencies(zlib_lib zlib)