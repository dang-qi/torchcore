cmake_minimum_required(VERSION 2.8)

cuda_add_library(modules_gpu MODULE overlaps_gpu.cu crop_and_resize_gpu.cu)
