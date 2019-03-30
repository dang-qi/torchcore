#ifndef CUDA_HEADERS_HPP
#define CUDA_HEADERS_HPP

#define CUDA_1D_KERNEL_LOOP(i,n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
#define threadsPerBlock 1024

#endif
