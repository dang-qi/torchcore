#ifndef ROIPOOL_GPU_HPP
#define ROIPOOL_GPU_HPP

#include <iostream>
#include <cfloat>
#include "cuda_tools.hpp"

void RoiPoolForwardGpu( const float* data, const int batch_size, const int nchannels, const int height, const int width,
                        const float* rois, const int* roibatches, const int nrois, const float spatial_scale,
                        float* crop, int* argmax, const int crop_height, const int crop_width );

void RoiPoolBackwardGpu( const float* crop_diff, const int* argmax, const int nrois, const float spatial_scale,
                         const int crop_height, const int crop_width,
                         float* data_diff, const int batch_size, const int channels, const int height, const int width,
                         const float* rois, const int* roibatches );

#endif
