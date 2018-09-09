cmake_minimum_required(VERSION 2.8)

project(extension)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
find_package(CUDA)

if(CUDA_FOUND)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -x cu")
	include_directories(./modules_gpu)
	add_subdirectory(./modules_gpu)
endif()

include_directories(./modules_cpu)
add_subdirectory(./modules_cpu)
