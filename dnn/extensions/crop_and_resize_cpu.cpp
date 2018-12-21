#include<torch/extension.h>
#include<vector>
#include<iostream>
#include"crop_and_resize_cpu.hpp"

using namespace std;

void crop_and_resize_forward_cpu( at::Tensor image, at::Tensor boxes_data, at::Tensor box_index,
                                  const float extrapolation_value, const int crop_height, const int crop_width,
                                  at::Tensor crops )
{
    const float* image_ptr = image.data<float>();
    const int batch_size = image.size(0);
    const int depth = image.size(1);
    const int image_height = image.size(2);
    const int image_width = image.size(3);

    const float* boxes_data_ptr = boxes_data.data<float>();
    const int* box_index_ptr = box_index.data<int>();
    const int num_boxes = boxes_data.size(0);

    float* crops_ptr = crops.data<float>();

    CropAndResizePerBoxForwardCpu( image_ptr, batch_size, depth, image_height, image_width,
                                boxes_data_ptr, box_index_ptr, 0, num_boxes,
                                crops_ptr, crop_height, crop_width, extrapolation_value );
}

void crop_and_resize_backward_cpu( at::Tensor grads, at::Tensor boxes, at::Tensor box_index,
                                   at::Tensor grads_image )
{
    const int batch_size = grads_image.size(0);
    const int depth = grads_image.size(1);
    const int image_height = grads_image.size(2);
    const int image_width = grads_image.size(3);

    const int num_boxes = boxes.size(0);
    const int crop_height = grads.size(2);
    const int crop_width = grads.size(3);

    const float* grads_ptr = grads.data<float>();
    const float* boxes_ptr = boxes.data<float>();
    const int* box_index_ptr = box_index.data<int>();
    float* grads_image_ptr = grads_image.data<float>();

    CropAndResizePerBoxBackwardCpu( grads_ptr, boxes_ptr, box_index_ptr, grads_image_ptr,
                                 batch_size, depth, image_height, image_width,
                                 num_boxes, crop_height, crop_width );

}

PYBIND11_MODULE( TORCH_EXTENSION_NAME, m )
{
    m.def("forward", &crop_and_resize_forward_cpu, "Crop and Resize forward");
    m.def("backward", &crop_and_resize_backward_cpu, "Crop and Resize backward");
}
