#include<torch/torch.h>
#include<vector>
#include<iostream>
#include"overlaps_gpu.hpp"

using namespace std;

void overlaps_forward_gpu( at::Tensor boxes0, at::Tensor labels0, at::Tensor batches0,
                       at::Tensor boxes1, at::Tensor labels1, at::Tensor batches1,
                       at::Tensor out, bool ignore_labels, bool ignore_batch )
{
    auto nboxes0 = boxes0.size(0);
    auto nboxes1 = boxes1.size(0);

    float* boxes0_ptr = boxes0.data<float>();
    float* labels0_ptr = labels0.data<float>();
    int64_t* batches0_ptr = batches0.data<int64_t>();

    float* boxes1_ptr = boxes1.data<float>();
    float* labels1_ptr = labels1.data<float>();
    int64_t* batches1_ptr = batches1.data<int64_t>();

    float* out_ptr = out.data<float>();

    overlaps_gpu( boxes0_ptr, labels0_ptr, batches0_ptr, nboxes0,
              	  boxes1_ptr, labels1_ptr, batches1_ptr, nboxes1,
              	  out_ptr, ignore_labels, ignore_batch );
}

PYBIND11_MODULE( TORCH_EXTENSION_NAME, m )
{
    m.def("forward", &overlaps_forward_gpu, "Overlaps forward");
}
