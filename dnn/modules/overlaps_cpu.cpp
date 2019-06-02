#include <torch/extension.h>
#include <iostream>
#include <cfloat>
#include <cmath>
#include <vector>

void overlaps_cpu_kernel( const float* rois0, const float* roilabels0, const float* roibatches0, int nrois0,
                          const float* rois1, const float* roilabels1, const float* roibatches1, int nrois1,
                          float* overlaps )
{
    int total = nrois0 * nrois1;

    for( int index=0 ; index<total ; index++ )
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

            float ix = fmin( roi1[2], roi0[2] ) - fmax( roi1[0], roi0[0] );
            float iy = fmin( roi1[3], roi0[3] ) - fmax( roi1[1], roi0[1] );
            float intersection = fmax(0.0,static_cast<double>(ix)+1)*fmax(0.0,static_cast<double>(iy)+1);

            overlaps[ index ] = intersection / ( roi0_area + roi1_area - intersection );
        }
        else
        {
            overlaps[ index ] = -1;
        }
    }
}

void overlaps_cpu( const at::Tensor rois0, const at::Tensor roilabels0, const at::Tensor roibatches0,
                   const at::Tensor rois1, const at::Tensor roilabels1, const at::Tensor roibatches1,
                   at::Tensor overlaps )
{
    AT_ASSERTM( rois0.size(0) == overlaps.size(0), "Number of rois0 does not match outputs dim1" );
    AT_ASSERTM( rois0.size(1) == 4, "Rois0 should be Kx4");
    AT_ASSERTM( rois0.size(0) == roilabels0.size(0), "The size of rois0 and roilabels0 does not match");
    AT_ASSERTM( rois0.size(0) == roibatches0.size(0), "The size of rois0 and roibatches0 does not match" );

    AT_ASSERTM( rois1.size(0) == overlaps.size(1), "Number of rois1 does not match outputs dim1" );
    AT_ASSERTM( rois1.size(1) == 4, "Rois1 should be Kx4");
    AT_ASSERTM( rois1.size(0) == roilabels1.size(0), "The size of rois1 and roilabels1 does not match");
    AT_ASSERTM( rois1.size(0) == roibatches1.size(0), "The size of rois1 and roibatches1 does not match" );



    int nrois0 = rois0.size(0);
    int nrois1 = rois1.size(0);

    overlaps_cpu_kernel(rois0.data<float>(), roilabels0.data<float>(), roibatches0.data<float>(), nrois0,
                        rois1.data<float>(), roilabels0.data<float>(), roibatches1.data<float>(), nrois1,
                        overlaps.data<float>() );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_cpu", &overlaps_cpu, "overlaps_cpu");
}
