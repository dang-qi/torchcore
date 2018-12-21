cmake_minimum_required(VERSION 2.8)

if(APPLE)
  add_library(modules_cpu SHARED overlaps_cpu.cpp crop_and_resize_cpu.cpp nms_cpu.cpp roipool_cpu.cpp)
else()
  add_library(modules_cpu MODULE overlaps_cpu.cpp crop_and_resize_cpu.cpp nms_cpu.cpp)
endif()
#set_target_properties( modules_cpu PROPERTIES SUFFIX ".so" )
