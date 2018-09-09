#include "overlaps_cpu.hpp"

void overlaps_cpu( const float* boxes0, const float* labels0, const int64_t* batches0, int64_t nboxes0,
                   const float* boxes1, const float* labels1, const int64_t* batches1, int64_t nboxes1,
                   float* overlaps, bool ignore_labels, bool ignore_batch )
{
    auto total = nboxes0 * nboxes1;
    for( auto index=0 ; index<total ; index++ )
    {
        int64_t index1 = index % nboxes1;
        int64_t index0 = index / nboxes1;

        bool label_check = (labels0[index0] == labels1[index1]) | ignore_labels;
        bool batch_check = (batches0[index0] == batches1[index1]) | ignore_batch;

        //cout << index0 << "\t" << index1 << "\t" << label_check << "\t" << batch_check
        //     << batches0[index0] << "\t" << batches1[index1] << endl;

        if( label_check && batch_check )
        {
            const float* box0 = boxes0 + 4*index0;
            const float* box1 = boxes1 + 4*index1;

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
