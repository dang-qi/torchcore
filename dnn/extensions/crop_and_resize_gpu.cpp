#include<torch/torch.h>
#include<vector>
#include<iostream>
#include"crop_and_resize_gpu.hpp"

using namespace std;

void crop_and_resize_forward_cpu( at::Tensor image, at::Tensor boxes_data, at::Tensor box_index,
                                  const float extrapolation_value, const int crop_height, const int crop_width,
                                  at::Tensor crops )
{
    const float* image_ptr = image.data<float>();
    const int batch = image.size(0);
    const int depth = image.size(1);
    const int image_height = image.size(2);
    const int image_width = image.size(3);

    const float* boxes_ptr = boxes_data.data<float>();
    const int* box_ind_ptr = box_index.data<int>();
    const int num_boxes = boxes_data.size(0);

    float* crops_ptr = crops.data<float>();

    CropAndResizePerBoxForwardGpu( image_ptr,boxes_ptr, box_ind_ptr, num_boxes, batch, image_height, image_width,
                                    crop_height, crop_width, depth, extrapolation_value, crops_ptr );
}

void crop_and_resize_backward_cpu( at::Tensor grads, at::Tensor boxes, at::Tensor box_index,
                                   at::Tensor grads_image )
{
    const int batch = grads_image.size(0);
    const int depth = grads_image.size(1);
    const int image_height = grads_image.size(2);
    const int image_width = grads_image.size(3);

    const int num_boxes = boxes.size(0);
    const int crop_height = grads.size(2);
    const int crop_width = grads.size(3);

    const float* grads_ptr = grads.data<float>();
    const float* boxes_ptr = boxes.data<float>();
    const int* box_ind_ptr = box_index.data<int>();
    float* grads_image_ptr = grads_image.data<float>();

    CropAndResizePerBoxBackwardGpu( grads_ptr, boxes_ptr, box_ind_ptr, num_boxes, batch, image_height, image_width,
                                    crop_height, crop_width, depth, grads_image_ptr );

}

PYBIND11_MODULE( TORCH_EXTENSION_NAME, m )
{
    m.def("forward", &crop_and_resize_forward_cpu, "Crop and Resize forward");
    m.def("backward", &crop_and_resize_backward_cpu, "Crop and Resize backward");
}
