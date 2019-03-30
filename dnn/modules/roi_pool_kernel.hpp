#ifndef ROI_POOL_KERNEL_HPP 
#define ROI_POOL_KERNEL_HPP

#include<cuda.h>
#include<iostream>
#include<cfloat>

void roi_pool_forward( const float* data, const float* rois, const int* roibatches, int batch_size, int nchannels, int height, int width,
                       int nrois, int rois_dim, int pool_h, int pool_w, float scale, float* output, int* memory );

void roi_pool_backward( const float* grad_out, const float* rois, const int* roibatches, int batch_size, int nchannels,
                        int height, int width, int nrois, int rois_dim, int pool_h, int pool_w, float* grad_in, const int* memory );
 
#endif
