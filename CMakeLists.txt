cmake_minimum_required(VERSION 3.16)

include(cmake_includes/init_property.cmake)

set(PROJECT_NAME "benchmark")
set(PROJECT_TARGET_NAME ${PROJECT_NAME})
set(CMAKE_VERBOSE_MAKEFILE ON)

project(${PROJECT_NAME} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)

set(SOURCES
    sources/main.cpp
    sources/ne10/benchmark_ne10.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})
include_directories(${PROJECT_TARGET_NAME} ${CMAKE_SOURCE_DIR}/ ${CMAKE_SOURCE_DIR}/sources/ne10)
include_directories(${PROJECT_TARGET_NAME} ${CMAKE_SOURCE_DIR}/ ${CMAKE_SOURCE_DIR}/sources/ne10/libNE10_1.2.1/inc)
target_link_libraries(${PROJECT_TARGET_NAME} ${CMAKE_SOURCE_DIR}/sources/ne10/libNE10_1.2.1/libNE10.a)
target_compile_options(${PROJECT_NAME} PRIVATE
    -march=armv7-a
    # -mcpu=cortex-a17
    -mtune=cortex-a17
    -mfpu=neon-vfpv4
    -mfloat-abi=hard
    -ffast-math
    -fassociative-math
    -ftree-vectorize
    -ftree-vectorizer-verbose=6
    -Ofast
    $<$<CONFIG:Dubug>:-O0;-ggdb>
    $<$<CONFIG:Release>:-Wextra;-Wall;-Wpedantic;-O3>
)