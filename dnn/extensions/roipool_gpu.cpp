#include<torch/extension.h>
#include<vector>
#include<iostream>

#include"roipool_gpu.hpp"

void roipool_forward_gpu( at::Tensor data, at::Tensor rois, at::Tensor roibatches,
                    const float spatial_scale, at::Tensor crop, at::Tensor argmax )
{
  const float* data_ptr = data.data<float>();

  const int batch_size = data.size(0);
  const int nchannels = data.size(1);
  const int data_height = data.size(2);
  const int data_width = data.size(3);

  const float* rois_ptr = rois.data<float>();
  const int* roibatches_ptr = roibatches.data<int>();
  const int nrois = rois.size(0);

  float* crop_ptr = crop.data<float>();
  int* argmax_ptr = argmax.data<int>();

  const int crop_height = crop.size(2);
  const int crop_width = crop.size(3);

  //RoiPoolForwardCpu( data_ptr, batch_size, nchannels, data_height, data_width,
  //                   rois_ptr, roibatches_ptr, nrois, spatial_scale,
  //                   crop_ptr, argmax_ptr, crop_height, crop_width );

}

void roipool_backward_gpu( at::Tensor crop_diff, at::Tensor argmax, at::Tensor rois,
                           at::Tensor roibatches, const float spatial_scale, at::Tensor data_diff )
{
  const float* crop_diff_ptr = crop_diff.data<float>();
  const int* argmax_ptr = argmax.data<int>();
  const int nrois = crop_diff.size(0);
  const int nchannels = crop_diff.size(1);
  const int crop_height = crop_diff.size(2);
  const int crop_width = crop_diff.size(3);

  const float* rois_ptr = rois.data<float>();
  const int* roibatches_ptr = roibatches.data<int>();

  float* data_diff_ptr = data_diff.data<float>();
  const int batch_size = data_diff.size(0);
  const int height = data_diff.size(2);
  const int width = data_diff.size(3);

  //RoiPoolBackwardCpu( crop_diff_ptr, argmax_ptr, nrois, spatial_scale, crop_height, crop_width,
  //                    data_diff_ptr, batch_size, nchannels, height, width, rois_ptr, roibatches_ptr );
}

PYBIND11_MODULE( TORCH_EXTENSION_NAME, m )
{
    m.def("forward", &roipool_forward_gpu, "Roipool Forward");
    m.def("backward", &roipool_backward_gpu, "Roipool Backward");
}
