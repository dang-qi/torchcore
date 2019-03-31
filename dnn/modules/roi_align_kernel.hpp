#ifndef ROI_ALIGN_KERNEL_HPP 
#define ROI_ALIGN_KERNEL_HPP

#include<cuda.h>
#include<iostream>
#include<cfloat>

void roi_align_forward( const float* data, const float* rois, const int* roibatches, int batch_size, int nchannels, int height, int width,
                       int nrois, int rois_dim, int pool_h, int pool_w, float scale, int sampling, float* output );

void roi_align_backward( const float* grad_out, const float* rois, const int* roibatches, int batch_size, int nchannels,
                        int height, int width, int nrois, int rois_dim, int pool_h, int pool_w, float scale, int sampling, float* grad_in );
 
#endif
