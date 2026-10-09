#include "kernels.h"
#include "cuda_utils.h"

__global__ void residual_add_kernel(float* output, const float* a,
                                     const float* b, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = a[idx] + b[idx];
    }
}

void launch_residual_add(float* output, const float* a, const float* b,
                          int num_elements, cudaStream_t stream) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    residual_add_kernel<<<blocks, threads, 0, stream>>>(output, a, b, num_elements);
}
