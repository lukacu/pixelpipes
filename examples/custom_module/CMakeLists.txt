cmake_minimum_required(VERSION 3.13)
project(pixelpipes_example LANGUAGES CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pixelpipes; print(pixelpipes.include_dirs()[0])"
    OUTPUT_VARIABLE PIXELPIPES_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PIXELPIPES_INCLUDE_RESULT
)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pixelpipes; print(pixelpipes.link_dirs()[0])"
    OUTPUT_VARIABLE PIXELPIPES_LIBRARY_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PIXELPIPES_LIBRARY_RESULT
)
if(NOT PIXELPIPES_INCLUDE_RESULT EQUAL 0 OR NOT PIXELPIPES_LIBRARY_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not query the active PixelPipes SDK paths")
endif()

add_library(pixelpipes_example MODULE example.cpp)
target_compile_features(pixelpipes_example PRIVATE cxx_std_17)
target_include_directories(pixelpipes_example PRIVATE "${PIXELPIPES_INCLUDE_DIR}")
target_link_directories(pixelpipes_example PRIVATE "${PIXELPIPES_LIBRARY_DIR}")
target_link_libraries(pixelpipes_example PRIVATE pixelpipes)
set_target_properties(pixelpipes_example PROPERTIES BUILD_RPATH "${PIXELPIPES_LIBRARY_DIR}")

if(WIN32)
    set_target_properties(pixelpipes_example PROPERTIES PREFIX "")
else()
    set_target_properties(pixelpipes_example PROPERTIES PREFIX "lib")
endif()
