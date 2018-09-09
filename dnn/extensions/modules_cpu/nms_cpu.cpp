#include"nms_cpu.hpp"

void nms_cpu( int64_t* keep_out_flat, const float* boxes_flat, const int64_t* order_flat,
                      const float* areas_flat, int64_t* num_out_flat, int64_t* suppressed_flat, const int64_t boxes_num,
                      const int64_t boxes_dim, const float nms_overlap_thresh )
{
    int64_t num_to_keep = 0;
    for( int64_t i=0 ; i<boxes_num ; i++ )
    {
        int64_t idx_i = order_flat[i];
        if( suppressed_flat[idx_i] == 1 )
        {
            continue;
        }
        keep_out_flat[num_to_keep++] = idx_i;

        float ix1 = boxes_flat[ idx_i*boxes_dim ];
        float iy1 = boxes_flat[ idx_i*boxes_dim+1 ];
        float ix2 = boxes_flat[ idx_i*boxes_dim+2 ];
        float iy2 = boxes_flat[ idx_i*boxes_dim+3 ];
        float iarea = areas_flat[idx_i];

        for( int64_t j=i+1 ; j<boxes_num ; j++ )
        {
            int64_t idx_j = order_flat[j];
            if( suppressed_flat[idx_j] == 1 )
            {
                continue;
            }

            float xx1 = fmaxf(ix1, boxes_flat[idx_j * boxes_dim]);
            float yy1 = fmaxf(iy1, boxes_flat[idx_j * boxes_dim + 1]);
            float xx2 = fminf(ix2, boxes_flat[idx_j * boxes_dim + 2]);
            float yy2 = fminf(iy2, boxes_flat[idx_j * boxes_dim + 3]);
            float w = fmaxf(0.0, xx2 - xx1 + 1);
            float h = fmaxf(0.0, yy2 - yy1 + 1);

            float inter = w * h;
            float ovr = inter / (iarea + areas_flat[idx_j] - inter);
            if (ovr >= nms_overlap_thresh) {
                suppressed_flat[idx_j] = 1;
            }
        }
    }
    *num_out_flat = num_to_keep;
}
