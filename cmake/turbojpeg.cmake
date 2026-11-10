set(PIXELPIPES_TURBOJPEG_ROOT "" CACHE PATH
    "Prefix containing a TurboJPEG installation")

find_path(PIXELPIPES_TURBOJPEG_INCLUDE_DIR NAMES turbojpeg.h
    HINTS
        "${PIXELPIPES_TURBOJPEG_ROOT}/include"
)
find_library(PIXELPIPES_TURBOJPEG_LIBRARY NAMES turbojpeg libturbojpeg
    HINTS
        "${PIXELPIPES_TURBOJPEG_ROOT}/lib"
        "${PIXELPIPES_TURBOJPEG_ROOT}/lib64"
)

if(NOT PIXELPIPES_TURBOJPEG_INCLUDE_DIR OR NOT PIXELPIPES_TURBOJPEG_LIBRARY)
    message(FATAL_ERROR
        "PixelPipes' JPEG decoder requires libjpeg-turbo (turbojpeg.h and "
        "libturbojpeg). Install its development package or set "
        "PIXELPIPES_TURBOJPEG_ROOT to a provisioned prefix.")
endif()

add_library(pixelpipes_turbojpeg UNKNOWN IMPORTED)
set_target_properties(pixelpipes_turbojpeg PROPERTIES
    IMPORTED_LOCATION "${PIXELPIPES_TURBOJPEG_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${PIXELPIPES_TURBOJPEG_INCLUDE_DIR}"
)

if(UNIX)
    set_property(TARGET pixelpipes_turbojpeg APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES "m;Threads::Threads")
endif()
