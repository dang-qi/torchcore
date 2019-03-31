#include "roi_pool_kernel.hpp"
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include "cuda_headers.hpp"

__global__ void roi_pool_forward_kernel( int total, const float* data, const float* rois, const int* roibatches, int batch_size, int nchannels,
	   									 int height, int width, int nrois, int rois_dim, int pool_h, int pool_w, float scale, float* output,
										 int* memory )
{
	CUDA_1D_KERNEL_LOOP(index,total)
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

void roi_pool_forward( const float* data, const float* rois, const int* roibatches, int batch_size, int nchannels, int height, int width,
                       int nrois, int rois_dim, int pool_h, int pool_w, float scale, float* output, int* memory )
{
	int total = nrois*nchannels*pool_h*pool_w;
	int blocksPerGrid = ( total + threadsPerBlock - 1) / threadsPerBlock;

    roi_pool_forward_kernel <<<blocksPerGrid, threadsPerBlock>>> (total, data, rois, roibatches, batch_size, nchannels,
																  height, width, nrois, rois_dim, pool_h, pool_w, scale,
																  output, memory );

    AT_CHECK(cudaGetLastError() == cudaSuccess, "roi_pool_forward_kernel failed");
}

__global__ void roi_pool_backward_kernel( int total, const float* grad_out, const float* rois, const int* roibatches, int batch_size, int nchannels,
										  int height, int width, int nrois, int rois_dim, int pool_h, int pool_w, float* grad_in,
										  const int* memory )
{
	CUDA_1D_KERNEL_LOOP(index,total)
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
			atomicAdd( offset_grad_in + argmax,  static_cast<float>(offset_grad_out[ ph * pool_w + pw ]) );
	}
}

void roi_pool_backward( const float* grad_out, const float* rois, const int* roibatches, int batch_size, int nchannels,
                        int height, int width, int nrois, int rois_dim, int pool_h, int pool_w, float* grad_in, const int* memory )
{
	int total = nrois*nchannels*pool_h*pool_w;
	int blocksPerGrid = ( total + threadsPerBlock - 1) / threadsPerBlock;

	roi_pool_backward_kernel <<<blocksPerGrid, threadsPerBlock>>>( total, grad_out, rois, roibatches, batch_size, nchannels,
																   height, width, nrois, rois_dim, pool_h, pool_w, grad_in,
																   memory );

    AT_CHECK(cudaGetLastError() == cudaSuccess, "roi_pool_backward_kernel failed");
}
