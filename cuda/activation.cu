#include "kernels.h"
#include <cmath>

__global__ void silu_mul_kernel(float* out, const float* gate, const float* up, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float g = gate[idx];
        float silu = g / (1.0f + expf(-g));
        out[idx] = silu * up[idx];
    }
}

void launch_silu_mul(float* out, const float* gate, const float* up, int num_elements, cudaStream_t stream) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    silu_mul_kernel<<<blocks, threads, 0, stream>>>(out, gate, up, num_elements);
}
