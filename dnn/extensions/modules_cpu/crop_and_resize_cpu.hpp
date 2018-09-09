#ifndef CROP_AND_RESIZE_CPU_HPP
#define CROP_AND_RESIZE_CPU_HPP

#include<cmath>
#include<cstdio>
#include<cstdlib>

using namespace std;

void CropAndResizePerBoxForwardCpu( const float* image, int batch_size, int depth, int image_height, int image_width,
                                 const float* boxes_data, const int* box_index_data, const int start_box, const int limit_box,
                                 float* crops_data, const int crop_height, const int crop_width,
                                 const float extrapolation_value );

void CropAndResizePerBoxBackwardCpu( const float* grads_data, const float* boxes_data, const int* box_index_data, float* grads_image_data,
                                  const int batch_size, const int depth, const int image_height, const int image_width,
                                  const int num_boxes, const int crop_height, const int crop_width );

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
