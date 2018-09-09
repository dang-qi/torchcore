#include<torch/torch.h>
#include<vector>
#include<iostream>

#include "nms_cpu.hpp"

using namespace std;

void nms_forward_cpu( at::Tensor keep_out, at::Tensor num_out, at::Tensor boxes, at::Tensor order,
                      at::Tensor areas, float nms_overlap_thresh )
{
    int64_t boxes_num = boxes.size(0);
    int64_t boxes_dim = boxes.size(1);

    int64_t* keep_out_flat = keep_out.data<int64_t>();
    const float* boxes_flat = boxes.data<float>();
    const int64_t* order_flat = order.data<int64_t>();
    const float* areas_flat = areas.data<float>();
    int64_t* num_out_flat = num_out.data<int64_t>();

    auto suppressed = at::zeros_like( order );
    int64_t* suppressed_flat = suppressed.data<int64_t>();

    nms_cpu( keep_out_flat, boxes_flat, order_flat, areas_flat, num_out_flat,
             suppressed_flat, boxes_num, boxes_dim, nms_overlap_thresh );
}

PYBIND11_MODULE( TORCH_EXTENSION_NAME, m )
{
    m.def("forward", &nms_forward_cpu, "NMS forward");
}
