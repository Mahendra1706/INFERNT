#include "kernels.h"
#include "cuda_utils.h"

__global__ void attention_scores_kernel(float* scores, const float* q,
                                         const float* k_cache,
                                         int num_q_heads, int num_kv_heads,
                                         int head_dim, int seq_len,
                                         int max_seq_len, float scale) {
    int q_head = blockIdx.x;
    int kv_head = q_head / (num_q_heads / num_kv_heads);
    int pos = threadIdx.x;
    if (pos >= seq_len) return;

    const float* q_vec = q + q_head * head_dim;
    const float* k_vec = k_cache + kv_head * max_seq_len * head_dim + pos * head_dim;

    float dot = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        dot += q_vec[d] * k_vec[d];
    }

    scores[q_head * seq_len + pos] = dot * scale;
}

__global__ void attention_values_kernel(float* output, const float* scores,
                                         const float* v_cache,
                                         int num_q_heads, int num_kv_heads,
                                         int head_dim, int seq_len,
                                         int max_seq_len) {
    int q_head = blockIdx.x;
    int kv_head = q_head / (num_q_heads / num_kv_heads);
    int d = threadIdx.x;
    if (d >= head_dim) return;

    const float* s = scores + q_head * seq_len;
    float sum = 0.0f;
    for (int pos = 0; pos < seq_len; pos++) {
        float v_val = v_cache[kv_head * max_seq_len * head_dim + pos * head_dim + d];
        sum += s[pos] * v_val;
    }

    output[q_head * head_dim + d] = sum;
}

void launch_attention_scores(float* scores, const float* q, const float* k_cache,
                              int num_q_heads, int num_kv_heads, int head_dim,
                              int seq_len, int max_seq_len, float scale,
                              cudaStream_t stream) {
    int threads = min(seq_len, 1024);
    attention_scores_kernel<<<num_q_heads, threads, 0, stream>>>(
        scores, q, k_cache, num_q_heads, num_kv_heads, head_dim,
        seq_len, max_seq_len, scale);
}

void launch_attention_values(float* output, const float* scores,
                              const float* v_cache,
                              int num_q_heads, int num_kv_heads, int head_dim,
                              int seq_len, int max_seq_len,
                              cudaStream_t stream) {
    int threads = head_dim;
    attention_values_kernel<<<num_q_heads, threads, 0, stream>>>(
        output, scores, v_cache, num_q_heads, num_kv_heads, head_dim,
        seq_len, max_seq_len);
}
