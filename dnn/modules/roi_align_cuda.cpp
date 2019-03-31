#include <torch/extension.h>
#include <iostream>
#include <cfloat>
#include <vector>

#include "roi_align_kernel.hpp"

using namespace std;

void roi_align_forward_gpu(const at::Tensor &input, const at::Tensor &rois, const at::Tensor &roibatches,
                                int64_t pool_h, int64_t pool_w,
                                double scale, int sampling, at::Tensor &output) {
    AT_CHECK(input.ndimension() == 4, "Feature should be BxCxHxW forms");
    AT_CHECK(input.is_contiguous(), "Feature should be contiguous");
    AT_CHECK(rois.ndimension() == 2, "ROI Proposals should be Kx4 forms");
    AT_CHECK(rois.size(1) == 4, "ROI proposals should be Kx4 forms");
    AT_CHECK(roibatches.size(0) == rois.size(0), "Number of rois and roibatches should be equal");
    AT_CHECK(rois.is_contiguous(), "ROI proposals should be contiguous.");
    AT_CHECK(roibatches.is_contiguous(), "ROI batches proposals should be contiguous.");

    //const vector<int64_t> rois_size = {rois.size(0), rois.size(1), pool_h, pool_w};
    //const vector<int64_t> input_size = {input.size(0), input.size(1), input.size(2), input.size(3)};

    //auto output = at::zeros( {rois_size[0], input_size[1], pool_h, pool_w}, input.type() );
    //auto output = input.type().tensor({rois_size[0], input_size[1], pool_h, pool_w});

    int batch_size = input.size(0);
    int nchannels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    int nrois = rois.size(0);
    int rois_dim = rois.size(1);

    roi_align_forward(input.data<float>(), rois.data<float>(), roibatches.data<int>(), batch_size, nchannels, height, width,
                     nrois, rois_dim, pool_h, pool_w,
                     static_cast<float>(scale), sampling, output.data<float>());

}

void roi_align_backward_gpu(const at::Tensor &rois, const at::Tensor &roibatches, const at::Tensor &grad_out,
                                 int64_t pool_h, int64_t pool_w, float scale, int sampling, at::Tensor grad_in) {
    AT_CHECK(grad_out.ndimension() == 4, "Feature should be BxCxHxW forms");
    AT_CHECK(grad_out.is_contiguous(), "Feature should be contiguous");
    AT_CHECK(rois.ndimension() == 2, "ROI Proposals should be Kx4 forms");
    AT_CHECK(rois.size(1) == 4 && rois.is_contiguous(), "ROI proposals should be Kx5 forms and contiguous");
    AT_CHECK(roibatches.size(0) == rois.size(0), "Number of rois and roibatches should be equal");

    //auto grad_in = grad_out.type().tensor({b_size, channel, h, w});
    //auto grad_in = at::zeros({b_size, channel, h, w}, grad_out.type());
    grad_in.zero_();

    int batch_size = grad_in.size(0);
    int nchannels = grad_in.size(1);
    int height = grad_in.size(2);
    int width = grad_in.size(3);

    int nrois = rois.size(0);
    int rois_dim = rois.size(1);

    roi_align_backward( grad_out.data<float>(), rois.data<float>(), roibatches.data<int>(),
                       batch_size, nchannels, height, width, nrois, rois_dim, pool_h, pool_w,
                       scale, sampling, grad_in.data<float>() );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_gpu", &roi_align_forward_gpu, "roi_align_forward_gpu");
    m.def("backward_gpu", &roi_align_backward_gpu, "roi_align_backward_gpu");
}
