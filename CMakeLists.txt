
PROJECT(pixelpipes)
CMAKE_MINIMUM_REQUIRED(VERSION 3.11 FATAL_ERROR)

INCLUDE(GNUInstallDirs)

SET(PROJECT_VERSION 0.1.0)

SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
SET(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
SET(CMAKE_CXX_STANDARD 17)

option(BUILD_INLINE "Build to source directory" OFF)
option(BUILD_DEBUG "Enable debug output" OFF)

if(BUILD_DEBUG)
    add_definitions(-DPIXELPIPES_DEBUG)
    add_definitions(-DPIXELPIPES_SOURCE_COMPILE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/src/")
endif()

FIND_PACKAGE(OpenCV REQUIRED core imgproc imgcodecs video calib3d)

SET(CORE_SOURCE
    src/queue.cpp
    src/random.cpp
    src/pipeline.cpp
    src/module.cpp
    src/type.cpp
    src/token.cpp
    src/serialization.cpp
    src/operation.cpp
    src/geometry.cpp
    src/image.cpp
    src/numbers.cpp
    src/list.cpp
)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS} include/)

ADD_LIBRARY(pixelpipes SHARED ${CORE_SOURCE})
TARGET_LINK_LIBRARIES(pixelpipes)
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

find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)

EXEC_PROGRAM(${PYTHON_EXECUTABLE}
             ARGS "-c \"import numpy; print(numpy.get_include())\""
             OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
             RETURN_VALUE NUMPY_NOT_FOUND
            )
IF(NUMPY_NOT_FOUND)
    message(FATAL_ERROR "NumPy headers not found")
endif()

INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIR})

ADD_LIBRARY(pypixelpipes SHARED 
    src/python/wrapper.cpp
    src/python/image.cpp 
)

TARGET_LINK_LIBRARIES(pypixelpipes pixelpipes)

set_target_properties(
    pypixelpipes
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME "pypixelpipes"
        LINKER_LANGUAGE C
    )

IF (BUILD_INLINE)

    set_target_properties(pixelpipes pypixelpipes pixelpipes_geometry pixelpipes_image
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/pixelpipes"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/pixelpipes"
    )

ENDIF()