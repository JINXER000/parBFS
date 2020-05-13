cmake_minimum_required(VERSION 2.8)
project("par-bucket-heap")
add_compile_options(-std=c++11)
#set(CMAKE_BUILD_TYPE Debug)
find_package(CUDA)
# ------------------------------
set(CUDA_SEPARABLE_COMPILATION TRUE)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

if( ${CMAKE_BUILD_TYPE} STREQUAL Release)
    SET(CUDA_NVCC_FLAGS "-arch=sm_61;-O3;-use_fast_math;-ftz=true;-prec-div=false;-prec-sqrt=false;-rdc=true" CACHE STRING "nvcc flags" FORCE)
    message("build CUDA with -O3")
else()
    message("build CUDA with Debug")
    SET(CUDA_NVCC_FLAGS "-g;-G;-arch=sm_61;-ftz=true;-rdc=true" CACHE STRING "nvcc flags" FORCE)#;-rdc=true
    SET(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
endif()

include_directories(
  /usr/include/eigen3
    ./include
    ./src/DijkstraShortestPath
  )
find_package(OpenCV)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)

  cuda_add_executable(test_bucket
    src/mainBucketHeap.cpp
	src/uiucbfs.cu	
    )
set(CUDA_cudadevrt_LIBRARY "/usr/local/cuda/lib64/libcudadevrt.a")
  target_link_libraries(test_bucket
      ${OpenCV_LIBRARIES}
      ${Boost_LIBRARIES}
      #-----------------------
#      /usr/local/cuda/lib64/libcudadevrt.a
      ${CUDA_cudadevrt_LIBRARY}
      )
