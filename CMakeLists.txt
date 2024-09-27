cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(ovadd_3d)

# requirements
find_package(Torch REQUIRED)

# Define our library target
add_library(ovadd_3d SHARED folding3d.cpp)
add_library(window_3d SHARED folding3d.cpp)
add_library(ovadd_3d_noblock SHARED folding3d.cpp)
add_library(window_3d_noblock SHARED folding3d.cpp)

# Enable C++14
target_compile_features(ovadd_3d PRIVATE cxx_std_14)

# Link against LibTorch
target_link_libraries(ovadd_3d ${LIBTORCH})
