
IF(${CMAKE_VERSION} GREATER_EQUAL "3.15")
    CMAKE_POLICY(SET CMP0091 NEW)
ENDIF()

IF(${CMAKE_VERSION} GREATER_EQUAL "3.19")
    CMAKE_POLICY(SET CMP0114 NEW)
ENDIF()

PROJECT(pixelpipes)
CMAKE_MINIMUM_REQUIRED(VERSION 3.11 FATAL_ERROR)

INCLUDE(GNUInstallDirs)
INCLUDE(FetchContent)
INCLUDE(ExternalProject)

SET(PROJECT_VERSION 0.1.0)
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_VISIBILITY_PRESET hidden)
SET(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_INPLACE "Build to source directory" OFF)
option(BUILD_PYTHON "Build the Python wrapper" ON)
option(BUILD_DEBUG "Enable debug output" OFF)
option(BUILD_TOOL "Build CLI tool" ON)

SET(CMAKE_EXTERNALS_DIR "${CMAKE_BINARY_DIR}/external/")

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ADD_COMPILE_OPTIONS(/W4 /WX /WL /wd4251 /permissive-)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    ADD_COMPILE_OPTIONS(-Wall -Wextra -pedantic -Werror)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    ADD_COMPILE_OPTIONS(-Wall -Wextra -pedantic -Werror)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined" )
endif()

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
INCLUDE(cmake/xtensor.cmake)

add_compile_definitions(XTENSOR_USE_XSIMD)
#add_compile_options(-march=native)

SET(CORE_SOURCE
    src/queue.cpp
    src/type.cpp
    src/token.cpp    
    src/module.cpp
    src/operation.cpp
    src/pipeline.cpp
    src/numbers.cpp
    src/buffer.cpp
    src/tensor.cpp
    src/geometry.cpp    
    src/image.cpp
    src/io.cpp 
    src/list.cpp
    src/pipeline/optimization.cpp
    src/serialization/compression.cpp
    src/serialization/serialization.cpp
)

INCLUDE_DIRECTORIES(SYSTEM ${OPENCV_INCLUDE_DIRS} ${lodepng_SOURCE_DIR} ${ZLIB_INCLUDE_DIRS}
    ${XTENSOR_INCLUDE_DIRS} ${lodepng_SOURCE_DIR}/include )
    
INCLUDE_DIRECTORIES(include/)

LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/deps/lib)

ADD_LIBRARY(pixelpipes SHARED ${CORE_SOURCE})
ADD_DEPENDENCIES(pixelpipes ${XTENSOR_TARGET})
TARGET_LINK_LIBRARIES(pixelpipes ${CMAKE_DL_LIBS} lodepng ${ZLIB_LIBS})
target_compile_definitions(pixelpipes PRIVATE PIXELPIPES_BUILD_CORE)


ADD_LIBRARY(pixelpipes_opencv SHARED 
    src/opencv/common.cpp
    src/opencv/loading.cpp
    src/opencv/view.cpp
    src/opencv/filter.cpp
    src/opencv/geometry.cpp
    src/opencv/processing.cpp
    src/opencv/render.cpp
    )

TARGET_LINK_LIBRARIES(pixelpipes_opencv pixelpipes ${OPENCV_LIBS})

# CMake config file
INCLUDE(CMakePackageConfigHelpers)

SET(BINARY_TARGETS pixelpipes pixelpipes_opencv)

# https://martinopilia.com/posts/2018/09/15/building-python-extension.html

IF(BUILD_PYTHON)

  find_package (pybind11)

  find_package (Python COMPONENTS Interpreter Development NumPy)

  INCLUDE_DIRECTORIES(${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS})

  ADD_LIBRARY(pypixelpipes SHARED 
      src/python/wrapper.cpp
      src/python/array.cpp 
  )

  TARGET_LINK_LIBRARIES(pypixelpipes pixelpipes ${Python_LIBRARIES})

  set_target_properties(
      pypixelpipes
      PROPERTIES
          PREFIX ""
          OUTPUT_NAME "pypixelpipes"
          LINKER_LANGUAGE C
      )

  LIST(APPEND BINARY_TARGETS pypixelpipes)

ENDIF()

# Where to place outputs
IF (BUILD_INPLACE)
    set_target_properties(${BINARY_TARGETS}
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/pixelpipes"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/pixelpipes"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/pixelpipes"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/pixelpipes"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/pixelpipes"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/pixelpipes"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/pixelpipes"
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/pixelpipes"
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/pixelpipes"
    )
ELSE()
    set_target_properties(${BINARY_TARGETS}
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
    )
ENDIF()

if(MSVC)
    target_link_options(pixelpipes PUBLIC /DEBUG)
endif()

IF (BUILD_TOOL)
    ADD_EXECUTABLE(pixelpipes_cli src/tool/main.cpp)
    SET_TARGET_PROPERTIES(pixelpipes_cli PROPERTIES OUTPUT_NAME "pixelpipes")
    TARGET_LINK_LIBRARIES(pixelpipes_cli pixelpipes)
ENDIF()