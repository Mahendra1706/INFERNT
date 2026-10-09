#include "kernels.h"
#include "cuda_utils.h"

__global__ void kv_cache_append_kernel(float* cache, const float* new_data,
                                        int num_heads, int head_dim,
                                        int max_seq_len, int position) {
    int head = blockIdx.x;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    int cache_offset = head * max_seq_len * head_dim + position * head_dim + d;
    int new_offset = head * head_dim + d;
    cache[cache_offset] = new_data[new_offset];
}

void launch_kv_cache_append(float* k_cache, float* v_cache,
                             const float* k_new, const float* v_new,
                             int num_kv_heads, int head_dim,
                             int max_seq_len, int position,
                             cudaStream_t stream) {
    int threads = head_dim;
    kv_cache_append_kernel<<<num_kv_heads, threads, 0, stream>>>(
        k_cache, k_new, num_kv_heads, head_dim, max_seq_len, position);
    kv_cache_append_kernel<<<num_kv_heads, threads, 0, stream>>>(
        v_cache, v_new, num_kv_heads, head_dim, max_seq_len, position);
}
