#ifndef OVERLAPS_KERNEL_HPP
#define OVERLAPS_KERNEL_HPP

#include<cuda.h>
#include<iostream>
#include<cfloat>

void overlaps_gpu_kernel( const float* rois0, const float* roilabels0, const int* roibatches0, int nrois0,
                          const float* rois1, const float* roilabels1, const int* roibatches1, int nrois1,
                          float* overlaps );

#endif
