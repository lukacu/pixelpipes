
PROJECT(pixelpipes)
CMAKE_MINIMUM_REQUIRED(VERSION 3.11 FATAL_ERROR)

INCLUDE(GNUInstallDirs)
INCLUDE(FetchContent)
INCLUDE(ExternalProject)

SET(PROJECT_VERSION 0.1.0)
SET(CMAKE_CXX_STANDARD 17)

option(BUILD_INPLACE "Build to source directory" OFF)
option(BUILD_PYTHON "Build the Python wrapper" ON)
option(BUILD_DEBUG "Enable debug output" OFF)
option(BUILD_TEST "Enable test snippets" OFF)

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined" )

FIND_PACKAGE(Threads REQUIRED)
LINK_LIBRARIES(Threads::Threads)

if(BUILD_DEBUG)
    add_definitions(-DPIXELPIPES_DEBUG)
    add_definitions(-DPIXELPIPES_SOURCE_COMPILE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/src/")
endif()

IF (NOT CMAKE_CXX_BYTE_ORDER)
    SET(BIG_ENDIAN 1)
    SET(LITTLE_ENDIAN 1)
    include (TestBigEndian)
    TEST_BIG_ENDIAN(IS_BIG_ENDIAN)
    if(IS_BIG_ENDIAN)
    SET(CMAKE_CXX_BYTE_ORDER BIG_ENDIAN)
    else()
    SET(CMAKE_CXX_BYTE_ORDER LITTLE_ENDIAN)
    endif()
ENDIF()

IF (CMAKE_CXX_BYTE_ORDER EQUAL BIG_ENDIAN)
    add_definitions(-DPIXELPIPES_BIGENDIAN)
ENDIF()

INCLUDE(cmake/lodepng.cmake)
INCLUDE(cmake/zlib.cmake)
INCLUDE(cmake/opencv.cmake)

#FIND_PACKAGE(OpenCV REQUIRED core imgproc imgcodecs video calib3d)

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
    src/io.cpp
)

INCLUDE_DIRECTORIES(${OPENCV_INCLUDE_DIRS} ${lodepng_SOURCE_DIR}
    ${xsimd_SOURCE_DIR}/include ${lodepng_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/deps/include include/ )
    
LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/deps/lib)

ADD_LIBRARY(pixelpipes SHARED ${CORE_SOURCE})
TARGET_LINK_LIBRARIES(pixelpipes ${CMAKE_DL_LIBS} lodepng zlib_lib)
target_compile_definitions(pixelpipes PRIVATE PIXELPIPES_BUILD_CORE)

ADD_LIBRARY(pixelpipes_geometry SHARED 
    src/geometry/common.cpp
    src/geometry/points.cpp
    src/geometry/view.cpp
    )

TARGET_LINK_LIBRARIES(pixelpipes_geometry pixelpipes ${OPENCV_LIBS})

ADD_LIBRARY(pixelpipes_image SHARED 
    src/image/common.cpp
    src/image/arithmetic.cpp
    src/image/filter.cpp
    src/image/geometry.cpp
    src/image/processing.cpp
    src/image/render.cpp
    )

MESSAGE(${OPENCV_LIBS}) 

TARGET_LINK_LIBRARIES(pixelpipes_image pixelpipes ${OPENCV_LIBS})
 

# CMake config file
INCLUDE(CMakePackageConfigHelpers)

# https://martinopilia.com/posts/2018/09/15/building-python-extension.html

IF(BUILD_PYTHON)

  find_package (pybind11)

  find_package (Python COMPONENTS Interpreter Development NumPy)

  INCLUDE_DIRECTORIES(${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS})

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

ENDIF()

# Where to place outputs
IF (BUILD_INPLACE)
    set_target_properties(pixelpipes pypixelpipes pixelpipes_geometry pixelpipes_image
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/pixelpipes"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/pixelpipes"
    )
ENDIF()

IF (BUILD_TEST)
    ADD_EXECUTABLE(test_compression src/serialization/test.cpp src/serialization/compression.cpp)
    TARGET_LINK_LIBRARIES(test_compression zlib_lib)
ENDIF()