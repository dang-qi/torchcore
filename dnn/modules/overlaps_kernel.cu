#include "overlaps_kernel.hpp"
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include "cuda_headers.hpp"

__global__ void overlaps_forward_kernel( int total, const float* rois0, const float* roilabels0, const int* roibatches0, int nrois0,
                                         const float* rois1, const float* roilabels1, const int* roibatches1, int nrois1,
                                         float* overlaps )
{
    CUDA_1D_KERNEL_LOOP( index, total )
    {
        int index0 = index / nrois1;
        int index1 = index % nrois1;

        bool label_check = (roilabels0[ index0 ] == roilabels1[ index1 ]);
        bool batch_check = (roibatches0[ index0 ] == roibatches1[ index1 ]);

        if( label_check && batch_check )
        {
            const float* roi0 = rois0 + 4*index0;
            const float* roi1 = rois1 + 4*index1;

            float roi0_area = (roi0[2]-roi0[0]+1)*(roi0[3]-roi0[1]+1);
            float roi1_area = (roi1[2]-roi1[0]+1)*(roi1[3]-roi1[1]+1);

            float ix = min( roi1[2], roi0[2] ) - max( roi1[0], roi0[0] );
            float iy = min( roi1[3], roi0[3] ) - max( roi1[1], roi0[1] );
            float intersection = max(0.0,static_cast<double>(ix)+1)*max(0.0,static_cast<double>(iy)+1);

            overlaps[ index ] = intersection / ( roi0_area + roi1_area - intersection );
        }
        else
        {
            overlaps[ index ] = -1;
        }
    }
}

void overlaps_gpu_kernel( const float* rois0, const float* roilabels0, const int* roibatches0, int nrois0,
                          const float* rois1, const float* roilabels1, const int* roibatches1, int nrois1,
                          float* overlaps )
{
    int total = nrois0 * nrois1;
	int blocksPerGrid = ( total + threadsPerBlock - 1) / threadsPerBlock;

    overlaps_forward_kernel <<<blocksPerGrid, threadsPerBlock>>> (total, rois0, roilabels0, roibatches0, nrois0,
                                                                         rois1, roilabels1, roibatches1, nrois1,
                                                                         overlaps );

    AT_CHECK(cudaGetLastError() == cudaSuccess, "roi_pool_forward_kernel failed");
}
