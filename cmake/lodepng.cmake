
if(PIXELPIPES_FETCH_DEPENDENCIES)
    FetchContent_Declare(lodepng
        GIT_REPOSITORY ${PIXELPIPES_LODEPNG_REPOSITORY}
        GIT_TAG ${PIXELPIPES_LODEPNG_REVISION}
    )
    FetchContent_MakeAvailable(lodepng)
    set(PIXELPIPES_LODEPNG_SOURCE_DIR "${lodepng_SOURCE_DIR}" CACHE PATH "lodepng sources" FORCE)
    ADD_LIBRARY(lodepng STATIC ${lodepng_SOURCE_DIR}/lodepng.cpp)
    set(LODEPNG_INCLUDE_DIRS "${lodepng_SOURCE_DIR};${lodepng_SOURCE_DIR}/include")
else()
    set(PIXELPIPES_LODEPNG_SOURCE_DIR "" CACHE PATH "Directory containing provisioned lodepng sources")
    if(PIXELPIPES_LODEPNG_SOURCE_DIR AND EXISTS "${PIXELPIPES_LODEPNG_SOURCE_DIR}/lodepng.cpp")
        ADD_LIBRARY(lodepng STATIC "${PIXELPIPES_LODEPNG_SOURCE_DIR}/lodepng.cpp")
        set(LODEPNG_INCLUDE_DIRS "${PIXELPIPES_LODEPNG_SOURCE_DIR};${PIXELPIPES_LODEPNG_SOURCE_DIR}/include")
    else()
        find_path(LODEPNG_INCLUDE_DIR NAMES lodepng.h)
        find_library(LODEPNG_LIBRARY NAMES lodepng)
        if(NOT LODEPNG_INCLUDE_DIR OR NOT LODEPNG_LIBRARY)
            message(FATAL_ERROR "No-network mode requires lodepng sources via PIXELPIPES_LODEPNG_SOURCE_DIR or an installed lodepng library/header")
        endif()
        add_library(lodepng UNKNOWN IMPORTED)
        set_target_properties(lodepng PROPERTIES IMPORTED_LOCATION "${LODEPNG_LIBRARY}")
        set(LODEPNG_INCLUDE_DIRS "${LODEPNG_INCLUDE_DIR}")
    endif()
endif()

get_target_property(LODEPNG_IS_IMPORTED lodepng IMPORTED)
if(NOT LODEPNG_IS_IMPORTED)
    set_property(TARGET lodepng PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
target_include_directories(lodepng INTERFACE ${LODEPNG_INCLUDE_DIRS})

if (MSVC AND NOT LODEPNG_IS_IMPORTED)
    target_compile_options(lodepng PUBLIC /WX-)
endif()

if(PIXELPIPES_FETCH_DEPENDENCIES)
    install(TARGETS lodepng
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dependencies
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dependencies
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT dependencies
    )
    install(FILES "${lodepng_SOURCE_DIR}/lodepng.h"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT dependencies
    )
endif()
