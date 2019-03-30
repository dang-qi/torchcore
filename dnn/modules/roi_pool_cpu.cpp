#include <torch/extension.h>
#include <iostream>
#include <cfloat>
#include <vector>

using namespace std;

void roi_pool_forward( const float* data, const float* rois, const int* roibatches, int batch_size, int nchannels, int height, int width,
                       int nrois, int rois_dim, int pool_h, int pool_w, float scale, float* output, int* memory )
{
    //cout << batch_size << "\t" << nchannels << "\t" << height << "\t" << width << endl;
    //cout << nrois << "\t" << rois_dim << "\t" << pool_h << "\t" << pool_w << endl;
    int total = nrois * nchannels * pool_h * pool_w;

    for( int index=0 ; index<total ; index ++ )
    {
        int pw = index % pool_w;
        int ph = (index / pool_w) % pool_h;
        int c = (index / pool_h / pool_w) % nchannels;
        int n = (index / pool_h / pool_w / nchannels );

        int roi_batch_idx = roibatches[n];
        const float* offset_rois = rois + n*rois_dim;

        int roi_start_w = round(offset_rois[0] * scale);
        int roi_start_h = round(offset_rois[1] * scale);
        int roi_end_w = round(offset_rois[2] * scale);
        int roi_end_h = round(offset_rois[3] * scale);

        int roi_w = max( roi_end_w-roi_start_w,1);
        int roi_h = max( roi_end_h-roi_start_h,1);

        float bin_size_h = static_cast<float>(roi_h) / static_cast<float>(pool_h);
        float bin_size_w = static_cast<float>(roi_w) / static_cast<float>(pool_w);

        int hstart = static_cast<int>( floor(static_cast<float>(ph) * bin_size_h ));
        int wstart = static_cast<int>( floor(static_cast<float>(pw) * bin_size_w ));
        int hend = static_cast<int>( ceil(static_cast<float>(ph+1) * bin_size_h ));
        int wend = static_cast<int>( ceil(static_cast<float>(pw+1) * bin_size_w ));

        hstart = min(max(hstart + roi_start_h, 0), height);
        hend = min(max(hend + roi_start_h, 0), height);
        wstart = min(max(wstart + roi_start_w, 0), width);
        wend = min(max(wend + roi_start_w, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        float maxval = is_empty ? 0 : -FLT_MAX;
        int maxidx = -1;
        const float* offset_data = data + ( roi_batch_idx * nchannels + c) * height * width;
        for( int hi=hstart ; hi < hend ; hi++ )
        {
            for( int wi=wstart ; wi < wend ; wi++ )
            {
                int ind = hi * width + wi;
                if( offset_data[ind] > maxval )
                {
                    maxval = offset_data[ind];
                    maxidx = ind;
                }
            }
        }

        if( index < total )
        {
            output[index] = maxval;
            if( memory )
                memory[index] = maxidx;
        }
    }
}

void roi_pool_forward_cpu(const at::Tensor &input, const at::Tensor &rois, const at::Tensor &roibatches,
                                int64_t pool_h, int64_t pool_w,
                                double scale, at::Tensor output, at::Tensor &memory) {
    AT_CHECK(input.ndimension() == 4, "Feature should be BxCxHxW forms");
    AT_CHECK(input.is_contiguous(), "Feature should be contiguous");
    AT_CHECK(rois.ndimension() == 2, "ROI Proposals should be Kx5 forms");
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

    if (memory.data<int>())
        memory.zero_();

    roi_pool_forward(input.data<float>(), rois.data<float>(), roibatches.data<int>(), batch_size, nchannels, height, width,
                     nrois, rois_dim, pool_h, pool_w,
                     static_cast<float>(scale), output.data<float>(), memory.data<int>());
}

void roi_pool_backward( const float* grad_out, const float* rois, const int* roibatches, int batch_size, int nchannels,
                        int height, int width, int nrois, int rois_dim, int pool_h, int pool_w, float* grad_in, const int* memory )
{
    int total = nrois * nchannels * pool_h * pool_w;

    for( int index=0 ; index<total ; index++ )
    {
        int pw = index % pool_w;
        int ph = ( index/pool_w ) % pool_h;
        int c = (index / pool_w / pool_h ) % nchannels;
        int n = index / pool_w / pool_h / nchannels;

        int roi_batch_idx = roibatches[n];
        //const float* offset_rois = rois + n * rois_dim;

        int grad_in_offset = ( roi_batch_idx * nchannels + c ) * height * width;
        int grad_out_offset = ( n * nchannels + c ) * pool_h * pool_w;

        const float* offset_grad_out = grad_out + grad_out_offset;
        float* offset_grad_in = grad_in + grad_in_offset;
        const int* offset_memory = memory + grad_out_offset;

        int argmax = offset_memory[ ph * pool_w + pw ];
        if( argmax != -1 )
            offset_grad_in[ argmax ] += static_cast<float>( offset_grad_out[ ph * pool_w + pw ] );

    }
}

void roi_pool_backward_cpu(const at::Tensor &rois, const at::Tensor &roibatches, const at::Tensor &grad_out,
                                 int64_t pool_h, int64_t pool_w, at::Tensor grad_in, at::Tensor &memory) {
    AT_CHECK(grad_out.ndimension() == 4, "Feature should be BxCxHxW forms");
    AT_CHECK(grad_out.is_contiguous(), "Feature should be contiguous");
    AT_CHECK(rois.ndimension() == 2, "ROI Proposals should be Kx5 forms");
    AT_CHECK(rois.size(1) == 4 && rois.is_contiguous(), "ROI proposals should be Kx5 forms and contiguous");
    AT_CHECK(roibatches.size(0) == rois.size(0), "Number of rois and roibatches should be equal");
    AT_CHECK(memory.is_contiguous(), "Memory should be contiguous.");

    //auto grad_in = grad_out.type().tensor({b_size, channel, h, w});
    //auto grad_in = at::zeros({b_size, channel, h, w}, grad_out.type());
    grad_in.zero_();

    int batch_size = grad_in.size(0);
    int nchannels = grad_in.size(1);
    int height = grad_in.size(2);
    int width = grad_in.size(3);

    int nrois = rois.size(0);
    int rois_dim = rois.size(1);

    roi_pool_backward( grad_out.data<float>(), rois.data<float>(), roibatches.data<int>(),
                       batch_size, nchannels, height, width, nrois, rois_dim, pool_h, pool_w,
                       grad_in.data<float>(), memory.data<int>() );

    //roi_pool_backward(grad_out.numel(), grad_out.data<float>(), rois.data<float>(), channel, h, w, pool_h, pool_w,
    //                  grad_in.data<float>(), memory.data<int>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_cpu", &roi_pool_forward_cpu, "roi_pool_forward_cpu");
    m.def("backward_cpu", &roi_pool_backward_cpu, "roi_pool_backward_cpu");
}
