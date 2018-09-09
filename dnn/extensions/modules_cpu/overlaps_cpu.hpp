#ifndef OVERLAPS_CPU_HPP
#define OVERLAPS_CPU_HPP

#include<iostream>

using namespace std;

void overlaps_cpu( const float* boxes0, const float* labels0, const int64_t* batches0, int64_t nboxes0,
                   const float* boxes1, const float* labels1, const int64_t* batches1, int64_t nboxes1,
                   float* overlaps, bool ignore_labels, bool ignore_batch );

#endif
