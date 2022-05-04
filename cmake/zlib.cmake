
SET(ZLIB_CMAKE_FLAGS
-DCMAKE_INSTALL_PREFIX=${CMAKE_EXTERNALS_DIR}zlib 
-DZLIB_ENABLE_TESTS=OFF 
-DCMAKE_POSITION_INDEPENDENT_CODE=ON
-DCMAKE_BUILD_TYPE=Release
-DCMAKE_DEBUG_POSTFIX=
)


ExternalProject_Add(zlib
PREFIX zlib
GIT_REPOSITORY    https://github.com/zlib-ng/zlib-ng
GIT_TAG           c882034d48afc0b32a38e8f7ca63a2e4e91ab42d
CMAKE_ARGS        ${ZLIB_CMAKE_FLAGS}
TEST_COMMAND      ""
)

add_library(zlib_lib STATIC IMPORTED)
set_target_properties(zlib_lib PROPERTIES IMPORTED_LOCATION ${CMAKE_EXTERNALS_DIR}zlib/${CMAKE_INSTALL_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}z-ng${CMAKE_STATIC_LIBRARY_SUFFIX})

add_dependencies(zlib_lib zlib)

SET(ZLIB_LIBS "zlib_lib")
SET(ZLIB_INCLUDE_DIRS "${CMAKE_EXTERNALS_DIR}zlib/include/")