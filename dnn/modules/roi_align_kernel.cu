#include "roi_align_kernel.hpp"
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include "cuda_headers.hpp"

__device__ float bilinear_interpolate( const float* data, const int h, const int w, float y, float x )
{
    if (y < -1.0 || y > h || x < -1.0 || x > w) {
        return 0;
    }

    y = y <= 0 ? 0 : (y >= h - 1 ? h - 1 : y);
    x = x <= 0 ? 0 : (x >= w - 1 ? w - 1 : x);

    int y_low = (int) y;
    int x_low = (int) x;
    int y_high = y_low >= h - 1 ? y_low : y_low + 1;
    int x_high = x_low >= w - 1 ? x_low : x_low + 1;

    float ly = y - y_low;
    float lx = x - x_low;
    float hy = 1. - ly, hx = 1. - lx;
    // do bilinear interpolation
    float v1 = data[y_low * w + x_low];
    float v2 = data[y_low * w + x_high];
    float v3 = data[y_high * w + x_low];
    float v4 = data[y_high * w + x_high];
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

__global__ void roi_align_forward_kernel( int total, const float* data, const float* rois, const int* roibatches, int batch_size, int nchannels,
	   									 int height, int width, int nrois, int rois_dim, int pool_h, int pool_w, float scale, int sampling,
										 float* output )
{
	CUDA_1D_KERNEL_LOOP(index,total)
	{
        int pw = index % pool_w;
        int ph = (index / pool_w) % pool_h;
        int c = (index / pool_h / pool_w) % nchannels;
        int n = (index / pool_h / pool_w / nchannels );

        int roi_batch_idx = roibatches[n];
        const float* offset_rois = rois + n*rois_dim;

        float start_x = offset_rois[0] * scale;
        float start_y = offset_rois[1] * scale;
        float end_x = offset_rois[2] * scale;
        float end_y = offset_rois[3] * scale;

        float roi_w = fmax( end_x-start_x,1.0);
        float roi_h = fmax( end_y-start_y,1.0);

        float bin_size_h = roi_h / static_cast<float>(pool_h);
        float bin_size_w = roi_w / static_cast<float>(pool_w);

        const float* offset_data = data + ( roi_batch_idx * nchannels + c) * height * width;

        // We use roi_bin_grid to sample the grid and mimic integral
        int bin_grid_h = sampling > 0 ? sampling : ceilf(static_cast<float>(roi_h) / pool_h);
        int bin_grid_w = sampling > 0 ? sampling : ceilf(static_cast<float>(roi_w) / pool_w);

        const float count = bin_grid_h * bin_grid_w;

        float output_val = 0;
        for( int iy=0 ; iy<bin_grid_h ; iy++ )
        {
            float y = start_y + ph*bin_size_h + static_cast<float>( iy + 0.5 ) * bin_size_h / static_cast<float>( bin_grid_h );
            for( int ix=0 ; ix<bin_grid_w ; ix++ )
            {
                float x = start_x + pw*bin_size_w + static_cast<float>( ix+0.5 ) * bin_size_w / static_cast<float>( bin_grid_w );
               	float val = bilinear_interpolate( offset_data, height, width, y, x );
                output_val += val;
            }
        }

        output[index] = output_val / count;
	}
}

void roi_align_forward( const float* data, const float* rois, const int* roibatches, int batch_size, int nchannels, int height, int width,
                       int nrois, int rois_dim, int pool_h, int pool_w, float scale, int sampling, float* output )
{
	int total = nrois*nchannels*pool_h*pool_w;
	int blocksPerGrid = ( total + threadsPerBlock - 1) / threadsPerBlock;

    roi_align_forward_kernel <<<blocksPerGrid, threadsPerBlock>>> (total, data, rois, roibatches, batch_size, nchannels,
																  height, width, nrois, rois_dim, pool_h, pool_w, scale,
																  sampling, output );

    AT_CHECK(cudaGetLastError() == cudaSuccess, "roi_pool_forward_kernel failed");
}

__device__ void bilinear_interpolate_gradient(const int h, const int w, float y, float x, float &w1, float &w2, float &w3, float &w4,
                                              int &pos1, int &pos2, int &pos3, int &pos4) {
    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > h || x < -1.0 || x > w) {
        w1 = w2 = w3 = w4 = 0.;
        pos1 = pos2 = pos3 = pos4 = -1;
        return;
    }

    y = y <= 0 ? 0 : (y >= h - 1 ? h - 1 : y);
    x = x <= 0 ? 0 : (x >= w - 1 ? w - 1 : x);

    int y_low = (int) y;
    int x_low = (int) x;
    int y_high = y_low >= h - 1 ? y_low : y_low + 1;
    int x_high = x_low >= w - 1 ? x_low : x_low + 1;

    pos1 = y_low * w + x_low;
    pos2 = y_low * w + x_high;
    pos3 = y_high * w + x_low;
    pos4 = y_high * w + x_high;

    float ly = y - y_low;
    float lx = x - x_low;
    float hy = 1. - ly, hx = 1. - lx;

    w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
}


__global__ void roi_align_backward_kernel( int total, const float* grad_out, const float* rois, const int* roibatches, int batch_size, int nchannels,
										  int height, int width, int nrois, int rois_dim, int pool_h, int pool_w, float scale, int sampling, float* grad_in )
{
	CUDA_1D_KERNEL_LOOP(index,total)
	{
        int pw = index % pool_w;
        int ph = (index / pool_w) % pool_h;
        int c = (index / pool_h / pool_w) % nchannels;
        int n = (index / pool_h / pool_w / nchannels );

        int roi_batch_idx = roibatches[n];
        const float* offset_rois = rois + n*rois_dim;

        float start_x = offset_rois[0] * scale;
        float start_y = offset_rois[1] * scale;
        float end_x = offset_rois[2] * scale;
        float end_y = offset_rois[3] * scale;

        float roi_w = fmax( end_x-start_x,1.0);
        float roi_h = fmax( end_y-start_y,1.0);

        float bin_size_h = roi_h / static_cast<float>(pool_h);
        float bin_size_w = roi_w / static_cast<float>(pool_w);

        float* offset_grad_in = grad_in + ( roi_batch_idx * nchannels + c ) * height * width;
        const float* offset_grad_out = grad_out + ( n*nchannels + c ) * pool_h*pool_w;
        const float grad_out_this_bin = offset_grad_out[ ph*pool_w + pw];

        int roi_bin_grid_h = sampling > 0 ? sampling : ceilf(static_cast<float>(roi_h) / pool_h);
        int roi_bin_grid_w = sampling > 0 ? sampling : ceilf(static_cast<float>(roi_w) / pool_w);

        const float count = roi_bin_grid_h * roi_bin_grid_w;

        float w1, w2, w3, w4;
        int pos1, pos2, pos3, pos4;
        for( int iy=0 ; iy<roi_bin_grid_h ; iy++ )
        {
            float y = start_y + ph*bin_size_h + static_cast<float>( iy + 0.5 ) * bin_size_h / static_cast<float>( roi_bin_grid_h );
            for( int ix=0 ; ix<roi_bin_grid_w ; ix++ )
            {
                float x = start_x + pw*bin_size_w + static_cast<float>( ix+0.5 ) * bin_size_w / static_cast<float>( roi_bin_grid_w );

                bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4, pos1, pos2, pos3, pos4);
                float g1 = grad_out_this_bin * w1 / count;
                float g2 = grad_out_this_bin * w2 / count;
                float g3 = grad_out_this_bin * w3 / count;
                float g4 = grad_out_this_bin * w4 / count;

                if (pos1 >= 0 && pos2 >= 0 && pos3 >= 0 && pos4 >= 0)
                {
					atomicAdd( offset_grad_in + pos1, g1 );
					atomicAdd( offset_grad_in + pos2, g2 );
					atomicAdd( offset_grad_in + pos3, g3 );
					atomicAdd( offset_grad_in + pos4, g4 );
                }
            }
        }
	}
}

void roi_align_backward( const float* grad_out, const float* rois, const int* roibatches, int batch_size, int nchannels,
                        int height, int width, int nrois, int rois_dim, int pool_h, int pool_w, float scale, 
						int sampling, float* grad_in )
{
	int total = nrois*nchannels*pool_h*pool_w;
	int blocksPerGrid = ( total + threadsPerBlock - 1) / threadsPerBlock;

	roi_align_backward_kernel <<<blocksPerGrid, threadsPerBlock>>>( total, grad_out, rois, roibatches, batch_size, nchannels,
																   height, width, nrois, rois_dim, pool_h, pool_w, scale, 
																   sampling, grad_in );

    AT_CHECK(cudaGetLastError() == cudaSuccess, "roi_pool_backward_kernel failed");
}
