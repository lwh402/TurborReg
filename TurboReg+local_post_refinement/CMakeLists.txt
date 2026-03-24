cmake_minimum_required(VERSION 3.10)
project(TurboReg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_DEMO_CPP "Build C++ demo" ON)
option(BUILD_PYBIND "Build Python bindings with pybind11" ON)

# LibTorch
find_package(Torch REQUIRED)

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# TurboReg Core Library
include_directories(turboreg/include)
add_subdirectory(turboreg)

# Demos
if (BUILD_DEMO_CPP)
    add_subdirectory(demo_cpp)
endif()
