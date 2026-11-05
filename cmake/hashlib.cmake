if(PIXELPIPES_FETCH_DEPENDENCIES)
    FetchContent_Declare(hashlib
        GIT_REPOSITORY ${PIXELPIPES_HASHLIB_REPOSITORY}
        GIT_TAG ${PIXELPIPES_HASHLIB_REVISION}
    )
    FetchContent_MakeAvailable(hashlib)
    set(PIXELPIPES_HASHLIB_SOURCE_DIR "${hashlib_SOURCE_DIR}" CACHE PATH
        "hash-library sources" FORCE)
else()
    set(PIXELPIPES_HASHLIB_SOURCE_DIR "" CACHE PATH
        "Directory containing provisioned hash-library sources")
    if(NOT PIXELPIPES_HASHLIB_SOURCE_DIR OR
       NOT EXISTS "${PIXELPIPES_HASHLIB_SOURCE_DIR}/sha256.cpp")
        message(FATAL_ERROR
            "No-network mode requires hash-library sources via "
            "PIXELPIPES_HASHLIB_SOURCE_DIR")
    endif()
endif()

set(HASHLIB_SOURCES
    "${PIXELPIPES_HASHLIB_SOURCE_DIR}/crc32.cpp"
    "${PIXELPIPES_HASHLIB_SOURCE_DIR}/md5.cpp"
    "${PIXELPIPES_HASHLIB_SOURCE_DIR}/sha1.cpp"
    "${PIXELPIPES_HASHLIB_SOURCE_DIR}/sha256.cpp"
    "${PIXELPIPES_HASHLIB_SOURCE_DIR}/keccak.cpp"
    "${PIXELPIPES_HASHLIB_SOURCE_DIR}/sha3.cpp"
)

add_library(hashlib STATIC ${HASHLIB_SOURCES})
set_property(TARGET hashlib PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(hashlib PUBLIC "${PIXELPIPES_HASHLIB_SOURCE_DIR}")

if (MSVC)
    target_compile_options(hashlib PUBLIC /WX-)
endif()

if(PIXELPIPES_FETCH_DEPENDENCIES)
    install(TARGETS hashlib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dependencies
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dependencies
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT dependencies
    )
    install(FILES
        "${PIXELPIPES_HASHLIB_SOURCE_DIR}/hash.h"
        "${PIXELPIPES_HASHLIB_SOURCE_DIR}/hmac.h"
        "${PIXELPIPES_HASHLIB_SOURCE_DIR}/crc32.h"
        "${PIXELPIPES_HASHLIB_SOURCE_DIR}/md5.h"
        "${PIXELPIPES_HASHLIB_SOURCE_DIR}/sha1.h"
        "${PIXELPIPES_HASHLIB_SOURCE_DIR}/sha256.h"
        "${PIXELPIPES_HASHLIB_SOURCE_DIR}/keccak.h"
        "${PIXELPIPES_HASHLIB_SOURCE_DIR}/sha3.h"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT dependencies
    )
endif()
