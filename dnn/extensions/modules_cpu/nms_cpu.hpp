#ifndef NMS_CPU_HPP
#define NMS_CPU_HPP

#include<iostream>
#include<cmath>

void nms_cpu( int64_t* keep_out_flat, const float* boxes_flat, const int64_t* order_flat,
                      const float* areas_flat, int64_t* num_out_flat, int64_t* suppressed_flat, const int64_t boxes_num,
                      const int64_t boxes_dim, const float nms_overlap_thresh );

#endif
