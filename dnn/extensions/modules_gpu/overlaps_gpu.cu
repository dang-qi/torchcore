#include "overlaps_gpu.hpp"

__global__ void BuildOverlaps( int64_t nthreads,
	   						   float* boxes0, float* labels0, int64_t* batches0, int nboxes0,
							   float* boxes1, float* labels1, int64_t* batches1, int nboxes1,
							   float* overlaps, bool ignore_labels, bool ignore_batch )
{
    CUDA_1D_KERNEL_LOOP( index, nthreads )
    {
        int64_t index1 = index % nboxes1;
        int64_t index0 = index / nboxes1;

        bool label_check = (labels0[index0] == labels1[index1]) | ignore_labels;
        bool batch_check = (batches0[index0] == batches1[index1]) | ignore_batch;

        //cout << index0 << "\t" << index1 << "\t" << label_check << "\t" << batch_check
        //     << batches0[index0] << "\t" << batches1[index1] << endl;

        if( label_check && batch_check )
        {
            float* box0 = boxes0 + 4*index0;
            float* box1 = boxes1 + 4*index1;

            float area0 = (box0[2]-box0[0]+1)*(box0[3]-box0[1]+1);
            float area1 = (box1[2]-box1[0]+1)*(box1[3]-box1[1]+1);

            float ix = min( box1[2], box0[2] ) - max( box1[0], box0[0] );
            float iy = min( box1[3], box0[3] ) - max( box1[1], box0[1] );
            float intersection = max(0.0,static_cast<double>(ix)+1)*max(0.0,static_cast<double>(iy)+1);
            overlaps[index] = intersection / ( area0 + area1 - intersection );
        }
        else
        {
            overlaps[index] = -1.0;
        }
    }
}

void overlaps_gpu( float* boxes0, float* labels0, int64_t* batches0, int64_t nboxes0,
               	   float* boxes1, float* labels1, int64_t* batches1, int64_t nboxes1,
                   float* overlaps, bool ignore_labels, bool ignore_batch )
{
	auto total = nboxes0 * nboxes1;

    cudaError_t err;
    int total_blocks = (total+kThreadsPerBlock-1)/kThreadsPerBlock;

    BuildOverlaps<<<total_blocks,kThreadsPerBlock>>>( total,
													  boxes0, labels0, batches0, nboxes0,
													  boxes1, labels1, batches1, nboxes1,
													  overlaps, ignore_labels, ignore_batch );

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

}
