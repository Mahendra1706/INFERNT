#pragma once

#include <cuda_runtime.h>

void launch_rmsnorm(float* output, const float* input, const float* weight,
                    int batch_size, int hidden_size, float eps,
                    cudaStream_t stream = nullptr);

void launch_residual_add(float* output, const float* a, const float* b,
                         int num_elements, cudaStream_t stream = nullptr);

void launch_rope(float* q, float* k,
                 int batch_size, int num_q_heads, int num_kv_heads,
                 int head_dim, int position, float theta,
                 cudaStream_t stream = nullptr);

void launch_softmax(float* output, const float* input,
                    int batch_size, int seq_len,
                    cudaStream_t stream = nullptr);

void launch_softmax_with_mask(float* output, const float* input,
                              int num_heads, int seq_len, int total_seq_len,
                              cudaStream_t stream = nullptr);

void launch_kv_cache_append(float* k_cache, float* v_cache,
                            const float* k_new, const float* v_new,
                            int num_kv_heads, int head_dim,
                            int max_seq_len, int position,
                            cudaStream_t stream = nullptr);

void launch_attention_scores(float* scores, const float* q, const float* k_cache,
                             int num_q_heads, int num_kv_heads, int head_dim,
                             int seq_len, int max_seq_len, float scale,
                             cudaStream_t stream = nullptr);

void launch_attention_values(float* output, const float* scores, const float* v_cache,
                             int num_q_heads, int num_kv_heads, int head_dim,
                             int seq_len, int max_seq_len,
                             cudaStream_t stream = nullptr);
