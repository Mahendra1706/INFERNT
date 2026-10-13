#include "kernels.h"

__global__ void embedding_lookup_kernel(float* output, const float* table, int token_id, int hidden_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < hidden_size) {
        output[idx] = table[token_id * hidden_size + idx];
    }
}

void launch_embedding_lookup(float* output, const float* table, int token_id, int hidden_size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (hidden_size + threads - 1) / threads;
    embedding_lookup_kernel<<<blocks, threads, 0, stream>>>(output, table, token_id, hidden_size);
}
