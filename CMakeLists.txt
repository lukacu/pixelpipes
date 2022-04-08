
PROJECT(pixelpipes)
CMAKE_MINIMUM_REQUIRED(VERSION 3.11 FATAL_ERROR)

INCLUDE(GNUInstallDirs)
INCLUDE(FetchContent)
INCLUDE(ExternalProject)

SET(PROJECT_VERSION 0.1.0)

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
SET(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
SET(CMAKE_CXX_STANDARD 17)

option(BUILD_INPLACE "Build to source directory" OFF)
option(BUILD_DEBUG "Enable debug output" OFF)

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined" )

FIND_PACKAGE(Threads REQUIRED)

LINK_LIBRARIES(Threads::Threads)

if(BUILD_DEBUG)
    add_definitions(-DPIXELPIPES_DEBUG)
    add_definitions(-DPIXELPIPES_SOURCE_COMPILE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/src/")
endif()

FetchContent_Declare(lodepng
  GIT_REPOSITORY https://github.com/lvandeve/lodepng.git
  GIT_TAG        5601b8272a6850b7c5d693dd0c0e16da50be8d8d
)

FetchContent_Declare(xtl
  GIT_REPOSITORY https://github.com/xtensor-stack/xtl
  GIT_TAG        e697c91e2a3ac571d120d2b093fb3b250d060a7d
)

FetchContent_Declare(xtensor
  GIT_REPOSITORY https://github.com/xtensor-stack/xtensor
  GIT_TAG        004370c79716a2d7b3cf9d04af947363d813a29f
)

FetchContent_Declare(xsimd
  GIT_REPOSITORY https://github.com/xtensor-stack/xsimd
  GIT_TAG        dbba514a7e918b4147396ae4d84ba349f5d9e893
)

FetchContent_MakeAvailable(lodepng xtl xtensor xsimd)

ADD_LIBRARY(lodepng STATIC ${lodepng_SOURCE_DIR}/lodepng.cpp)

ExternalProject_Add(snappy
  PREFIX snappy
  GIT_REPOSITORY    https://github.com/google/snappy.git
  GIT_TAG           8dd58a519f79f0742d4c68fbccb2aed2ddb651e8
  CMAKE_ARGS        -DSNAPPY_BUILD_TESTS=OFF -DSNAPPY_BUILD_BENCHMARKS=OFF -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/deps -DCMAKE_CXX_FLAGS=-fPIC
  INSTALL_DIR       "${CMAKE_BINARY_DIR}/deps"
  TEST_COMMAND      ""
)

add_library(snappy_lib STATIC IMPORTED)
set_target_properties(snappy_lib PROPERTIES IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/deps/lib/libsnappy.a)

add_dependencies(snappy_lib snappy)

FIND_PACKAGE(OpenCV REQUIRED core imgproc imgcodecs video calib3d)

SET(CORE_SOURCE
    src/queue.cpp
    src/random.cpp
    src/pipeline.cpp
    src/module.cpp
    src/type.cpp
    src/token.cpp
    src/serialization/serialization.cpp
    src/serialization/compression.cpp
    src/operation.cpp
    src/geometry.cpp
    src/image.cpp
    src/numbers.cpp
    src/list.cpp
)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS} ${lodepng_SOURCE_DIR}
    ${xsimd_SOURCE_DIR}/include ${lodepng_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/deps/include include/ )
    
LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/deps/lib)

ADD_LIBRARY(pixelpipes SHARED ${CORE_SOURCE})
TARGET_LINK_LIBRARIES(pixelpipes ${CMAKE_DL_LIBS} lodepng snappy_lib)
target_compile_definitions(pixelpipes PRIVATE PIXELPIPES_BUILD_CORE)

ADD_LIBRARY(pixelpipes_geometry SHARED 
    src/geometry/common.cpp
    src/geometry/points.cpp
    src/geometry/view.cpp
    )

TARGET_LINK_LIBRARIES(pixelpipes_geometry pixelpipes ${OpenCV_LIBS})

ADD_LIBRARY(pixelpipes_image SHARED 
    src/image/common.cpp
    src/image/arithmetic.cpp
    src/image/filter.cpp
    src/image/geometry.cpp
    src/image/processing.cpp
    src/image/render.cpp
    )

TARGET_LINK_LIBRARIES(pixelpipes_image pixelpipes ${OpenCV_LIBS})


# CMake config file
INCLUDE(CMakePackageConfigHelpers)

# https://martinopilia.com/posts/2018/09/15/building-python-extension.html

find_package (Python COMPONENTS Interpreter Development NumPy)


INCLUDE_DIRECTORIES(${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})

ADD_LIBRARY(pypixelpipes SHARED 
    src/python/wrapper.cpp
    src/python/image.cpp 
)

TARGET_LINK_LIBRARIES(pypixelpipes pixelpipes ${Python_LIBRARIES})

set_target_properties(
    pypixelpipes
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME "pypixelpipes"
        LINKER_LANGUAGE C
    )

IF (BUILD_INPLACE)

    set_target_properties(pixelpipes pypixelpipes pixelpipes_geometry pixelpipes_image
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/pixelpipes"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/pixelpipes"
    )

ENDIF()