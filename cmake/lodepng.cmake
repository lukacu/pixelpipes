
FetchContent_Declare(lodepng
GIT_REPOSITORY https://github.com/lvandeve/lodepng.git
GIT_TAG        5601b8272a6850b7c5d693dd0c0e16da50be8d8d
)

FetchContent_MakeAvailable(lodepng)

ADD_LIBRARY(lodepng STATIC ${lodepng_SOURCE_DIR}/lodepng.cpp)
set_property(TARGET lodepng PROPERTY POSITION_INDEPENDENT_CODE ON)

if (MSVC)
    target_compile_options(lodepng PUBLIC /WX-)
endif()