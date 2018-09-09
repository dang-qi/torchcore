#ifndef CROP_AND_RESIZE_GPU_HPP
#define CROP_AND_RESIZE_GPU_HPP

#include<cmath>
#include<cstdio>
#include<cstdlib>
#include "cuda_tools.hpp"

using namespace std;

void CropAndResizePerBoxForwardGpu( const float* image_ptr, const float* boxes_ptr, const int* box_ind_ptr,
                                    int num_boxes, int batch, int image_height, int image_width, int crop_height,
                                    int crop_width, int depth, float extrapolation_value, float *crops_ptr );

void CropAndResizePerBoxBackwardGpu( const float *grads_ptr, const float *boxes_ptr, const int *box_ind_ptr, int num_boxes,
                                     int batch, int image_height, int image_width, int crop_height, int crop_width, int depth,
                                     float *grads_image_ptr );

#endif

/*void crop_and_resize_forward(
    THFloatTensor * image,
    THFloatTensor * boxes,      // [y1, x1, y2, x2]
    THIntTensor * box_index,    // range in [0, batch_size)
    const float extrapolation_value,
    const int crop_height,
    const int crop_width,
    THFloatTensor * crops
);

void crop_and_resize_backward(
    THFloatTensor * grads,
    THFloatTensor * boxes,      // [y1, x1, y2, x2]
    THIntTensor * box_index,    // range in [0, batch_size)
    THFloatTensor * grads_image // resize to [bsize, c, hc, wc]
);*/
