#ifndef OVERLAPS_GPU_HPP
#define OVERLAPS_GPU_HPP

#include <iostream>
#include <cfloat>
#include "cuda_tools.hpp"


void overlaps_gpu( float* boxes0, float* labels0, int64_t* batches0, int64_t nboxes0,
               	   float* boxes1, float* labels1, int64_t* batches1, int64_t nboxes1,
                   float* overlaps, bool ignore_labels, bool ignore_batch );

#endif
