#ifndef CUDA_TOOLS_HPP
#define CUDA_TOOLS_HPP

#define CUDA_1D_KERNEL_LOOP(i,n) for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
#define kThreadsPerBlock 1024

#endif
