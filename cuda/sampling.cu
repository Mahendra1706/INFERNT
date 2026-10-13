#include "kernels.h"

__global__ void argmax_kernel(int* out_token, const float* logits, int vocab_size) {
    __shared__ float s_max[256];
    __shared__ int s_idx[256];

    int tid = threadIdx.x;
    float max_val = -1e30f;
    int max_idx = 0;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = logits[i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    s_max[tid] = max_val;
    s_idx[tid] = max_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_max[tid + s] > s_max[tid]) {
                s_max[tid] = s_max[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_token = s_idx[0];
    }
}

void launch_argmax(int* out_token, const float* logits, int vocab_size, cudaStream_t stream) {
    argmax_kernel<<<1, 256, 0, stream>>>(out_token, logits, vocab_size);
}
