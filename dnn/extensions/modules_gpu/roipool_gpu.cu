#include "roipool_gpu.hpp"

__global__ void RoipoolForwardKernel( int64_t nthreads, const float* data, const int batch_size, const int nchannels, const int height, const int width,
                        const float* rois, const int* roibatches, const int nrois, const float spatial_scale,
                        float* crop, int* argmax, const int pooled_height, const int pooled_width )
{
  CUDA_1D_KERNEL_LOOP( index, nthreads )
  {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    int roi_batch_index = roibatches[n];
    const float* roi = rois + n*4;
    int x0 = round( roi[0] * spatial_scale );
    int y0 = round( roi[1] * spatial_scale );
    int x1 = round( roi[2] * spatial_scale );
    int y1 = round( roi[3] * spatial_scale );

    int roi_width = max( x1-x0+1, 1 );
    int roi_height = max( y1-y0+1, 1 );

    float bin_size_h = static_cast<float>(roi_height) / static_cast<float>(pooled_height);
    float bin_size_w = static_cast<float>(roi_width) / static_cast<float>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<float>(ph)*bin_size_h));
    int hend = static_cast<int>(ceil(static_cast<float>(ph+1)*bin_size_h));

    int wstart = static_cast<int>(floor(static_cast<float>(pw)*bin_size_w));
    int wend = static_cast<int>(ceil(static_cast<float>(pw+1)*bin_size_w));

    hstart = min( max(hstart + y0, 0), height );
    hend = min( max(hend + y0, 0), height );
    wstart = min( max(wstart + x0, 0), width );
    wend = min( max(wend + x0, 0), width );

    bool is_empty = (hend <= hstart) || (wend <= wstart);
    float maxval = is_empty ? 0 : -FLT_MAX;

    int maxidx = -1;
    const float* data_ptr = data + (roi_batch_index* channels + c) * height * width;
    for( int h=hstart ; h<hend ; h++ )
    {
      for( int w=wstart ; w<wend ; w++ )
      {
        int data_index = h*width + w;
        if( data_ptr[data_index] > maxval )
        {
          maxval = data_ptr[data_index];
          maxidx = data_index;
        }
      }
    }
    crop[index] = maxval;
    argmax[index] = maxidx;
  }
}

__global__ void RoipoolBackwardKernel( int64_t nthreads, const float* crop_diff, const int* argmax, const int nrois, const float spatial_scale,
                         const int crop_height, const int crop_width,
                         float* data_diff, const int batch_size, const int channels, const int height, const int width,
                         const float* rois, const int* roibatches )
{
  CUDA_1D_KERNEL_LOOP( index, nthreads )
  {
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    float gradient = 0;
    for( int roi_idx=0 ; roi_idx<nrois; roi_idx++ )
    {
      const float* roi = rois + roi_idx*4;
      int roi_batch_index = roibatches[roi_idx];

      if( roi_batch_index != n )
      {
        continue;
      }

      int x0 = round( roi[0] * spatial_scale );
      int y0 = round( roi[1] * spatial_scale );
      int x1 = round( roi[2] * spatial_scale );
      int y1 = round( roi[3] * spatial_scale );

      const bool in_roi = (w>=x0) && ( w<=x1 ) && (h>=y0) && (h<=y1);

      if(!in_roi)
      {
        continue;
      }

      int offset = ( roi_batch_index * channels + c ) * crop_height * crop_width;
      const float* offset_crop_diff = crop_diff + offset;
      const int* offset_argmax = argmax + offset;

      int roi_width = std::max( x1-x0+1, 1 );
      int roi_height = std::max( y1-y0+1, 1 );

      float bin_size_h = static_cast<float>(roi_height) / static_cast<float>(crop_height);
      float bin_size_w = static_cast<float>(roi_width) / static_cast<float>(crop_width);

      int phstart = floor(static_cast<float>(h-y0)/bin_size_h);
      int phend = ceil(static_cast<float>(h-y0+1)/bin_size_h);

      int pwstart = floor(static_cast<float>(w-x0)/bin_size_h);
      int pwend = ceil(static_cast<float>(w-x0+1)/bin_size_h);

      phstart = min(max(phstart, 0), crop_height);
      phend = min(max(phend, 0), crop_height);
      pwstart = min(max(pwstart, 0), crop_width);
      pwend = min(max(pwend, 0), crop_width);

      for( int ph=phstart ; ph<phend ; ph++ )
      {
        for( int pw=pwstart; pw<pwend ; pw++ )
        {
          if( offset_argmax[ph*crop_width+pw] == (h*width+w) )
          {
            gradient += offset_crop_diff[ph*crop_width+pw];
          }
        }
      }
    }
    data_diff[index] = gradient;
  }
}

void RoiPoolForwardGpu( const float* data, const int batch_size, const int nchannels, const int height, const int width,
                        const float* rois, const int* roibatches, const int nrois, const float spatial_scale,
                        float* crop, int* argmax, const int crop_height, const int crop_width )
{
  int total = nrois * channels * pooled_height * pooled_width; //crop_count
  cudaError_t err;
  int total_blocks = (total+kThreadsPerBlock-1)/kThreadsPerBlock;

  RoipoolForwardKernel<<<total_blocks,kThreadsPerBlock>>>( total,
                          data, batch_size, nchannels, height, width,
                          rois, roibatches, nrois, spatial_scale,
                          crop, argmax, crop_height, crop_width );

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
      fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
      exit( -1 );
  }
}

void RoiPoolBackwardGpu( const float* crop_diff, const int* argmax, const int nrois, const float spatial_scale,
                         const int crop_height, const int crop_width,
                         float* data_diff, const int batch_size, const int channels, const int height, const int width,
                         const float* rois, const int* roibatches )
{
  int total = batch_size * channels * height * width; //data_count
  cudaError_t err;
  int total_blocks = (total+kThreadsPerBlock-1)/kThreadsPerBlock;

  RoipoolBackwardKernel<<<total_blocks,kThreadsPerBlock>>>( total,
                          crop_diff, argmax, nrois, spatial_scale, crop_height, crop_width,
                          data_diff, batch_size, channels, height, width, rois,
                          roibatches );

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }
}
