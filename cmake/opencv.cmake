

SET(OPENCV_STATIC_FLAGS
  -DCMAKE_INSTALL_PREFIX=${CMAKE_EXTERNALS_DIR}opencv/
  -DOPENCV_LIB_ARCHIVE_INSTALL_PATH=${CMAKE_EXTERNALS_DIR}opencv/lib
  -DOPENCV_INCLUDE_INSTALL_PATH=${CMAKE_EXTERNALS_DIR}opencv/include
  -DOPENCV_3P_LIB_INSTALL_PATH=${CMAKE_EXTERNALS_DIR}opencv/lib
  -DOPENCV_DEBUG_POSTFIX=
  -DOPENCV_DLLVERSION=
  -DCMAKE_BUILD_TYPE=Release
  -DENABLE_PIC=ON
  -DBUILD_WITH_STATIC_CRT=OFF
  -DBUILD_SHARED_LIBS=OFF
  -DBUILD_ZLIB=ON
  -DBUILD_PNG=ON
  -DWITH_JPEG=ON
  -DWITH_OPENEXR=OFF
  -DWITH_JASPER=OFF
  -DWITH_TIFF=ON
  -DWITH_WEBP=OFF
  -DWITH_OPENCL=OFF
  -DWITH_GTK=OFF
  -DWITH_FFMPEG=OFF
  -DWITH_GSTREAMER=OFF
  -DWITH_V4L=OFF
  -DWITH_1394=OFF
  -DWITH_IPP=OFF
  -DWITH_TBB=OFF
  -DWITH_OPENMP=OFF
  -DWITH_CUDA=OFF
  -DWITH_PTHREADS_PF=OFF
  -DBUILD_JPEG=ON
  -DBUILD_TIFF=ON
  -DBUILD_OPENJPEG=ON
  -DBUILD_TESTS=OFF
  -DBUILD_PERF_TESTS=OFF
  -DBUILD_JAVA=OFF
  -DBUILD_opencv_apps=OFF
  -DBUILD_opencv_dnn=OFF
  -DBUILD_opencv_gapi=OFF
  -DBUILD_opencv_highgui=OFF
  -DBUILD_opencv_java=OFF
  -DBUILD_opencv_js=OFF
  -DBUILD_opencv_ml=OFF
  -DBUILD_opencv_objc=OFF
  -DBUILD_opencv_objdetect=OFF
  -DBUILD_opencv_photo=OFF
  -DBUILD_opencv_stitching=OFF
  -DBUILD_opencv_ts=OFF
  -DBUILD_opencv_world=OFF
  -DBUILD_opencv_video=OFF
  -DBUILD_opencv_videoio=OFF
  -DBUILD_opencv_python3=OFF
  -DBUILD_opencv_python2=OFF
)

SET(OPENCV_LIBS "")
SET(OPENCV_INCLUDE_DIRS "${CMAKE_EXTERNALS_DIR}opencv/include/")

SET(_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
SET(_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})


IF(NOT EXISTS "${CMAKE_EXTERNALS_DIR}opencv/lib/${_PREFIX}opencv_core${_SUFFIX}")
ExternalProject_Add(opencv
  PREFIX opencv
  GIT_REPOSITORY https://github.com/opencv/opencv
  GIT_TAG 4.5.5
  CMAKE_ARGS ${OPENCV_STATIC_FLAGS}
  TEST_COMMAND ""
)
ENDIF()

foreach(M IN ITEMS zlib ittnotify libpng libopenjp2 libjpeg-turbo libtiff)
  add_library(opencv_3dparty_${M} STATIC IMPORTED)
  set_target_properties(opencv_3dparty_${M}
    PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "C"
    IMPORTED_LOCATION ${CMAKE_EXTERNALS_DIR}opencv/lib/${_PREFIX}${M}${_SUFFIX})
  add_dependencies(opencv_3dparty_${M} opencv)
endforeach()

foreach(M IN ITEMS core imgproc imgcodecs calib3d features2d flann)
  add_library(opencv_${M} STATIC IMPORTED)
  set_target_properties(opencv_${M} PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION ${CMAKE_EXTERNALS_DIR}opencv/lib/${_PREFIX}opencv_${M}${_SUFFIX})
  add_dependencies(opencv_${M} opencv)
  LIST(APPEND OPENCV_LIBS "opencv_${M}")
endforeach()

IF(APPLE)
  LIST(APPEND OPENCV_LIBS "-framework Accelerate")
ENDIF()

set_target_properties(opencv_core PROPERTIES
  INTERFACE_LINK_LIBRARIES "\$<LINK_ONLY:opencv_3dparty_zlib>;\$<LINK_ONLY:opencv_3dparty_ittnotify>"
)

set_target_properties(opencv_flann PROPERTIES
  INTERFACE_LINK_LIBRARIES "opencv_core;opencv_core;"
)

set_target_properties(opencv_imgproc PROPERTIES
  INTERFACE_LINK_LIBRARIES "opencv_core;opencv_core;"
)

set_target_properties(opencv_features2d PROPERTIES
  INTERFACE_LINK_LIBRARIES "opencv_core;opencv_flann;opencv_imgproc;opencv_core;opencv_flann;opencv_imgproc"
)

set_target_properties(opencv_imgcodecs PROPERTIES
  INTERFACE_LINK_LIBRARIES "opencv_core;opencv_imgproc;opencv_core;opencv_imgproc;\$<LINK_ONLY:opencv_3dparty_libjpeg-turbo>;\$<LINK_ONLY:opencv_3dparty_libopenjp2>;\$<LINK_ONLY:opencv_3dparty_libpng>;\$<LINK_ONLY:opencv_3dparty_zlib>;\$<LINK_ONLY:opencv_3dparty_libtiff>"
)

set_target_properties(opencv_calib3d PROPERTIES
  INTERFACE_LINK_LIBRARIES "opencv_core;opencv_flann;opencv_imgproc;opencv_features2d;opencv_core;opencv_flann;opencv_imgproc;opencv_features2d"
)
