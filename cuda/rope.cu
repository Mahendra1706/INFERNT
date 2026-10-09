#include "kernels.h"
#include "cuda_utils.h"
#include <cmath>

__global__ void rope_kernel(float* q, float* k,
                            int num_q_heads, int num_kv_heads,
                            int head_dim, int position, float theta) {
    int head = blockIdx.x;
    int d = threadIdx.x * 2;
    if (d >= head_dim) return;

    float freq = 1.0f / powf(theta, (float)d / (float)head_dim);
    float angle = position * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    if (head < num_q_heads) {
        int offset = head * head_dim + d;
        float q0 = q[offset];
        float q1 = q[offset + 1];
        q[offset]     = q0 * cos_val - q1 * sin_val;
        q[offset + 1] = q0 * sin_val + q1 * cos_val;
    }

    if (head < num_kv_heads) {
        int offset = head * head_dim + d;
        float k0 = k[offset];
        float k1 = k[offset + 1];
        k[offset]     = k0 * cos_val - k1 * sin_val;
        k[offset + 1] = k0 * sin_val + k1 * cos_val;
    }
}

void launch_rope(float* q, float* k,
                 int batch_size, int num_q_heads, int num_kv_heads,
                 int head_dim, int position, float theta,
                 cudaStream_t stream) {
    int max_heads = max(num_q_heads, num_kv_heads);
    int threads = head_dim / 2;
    rope_kernel<<<max_heads, threads, 0, stream>>>(q, k, num_q_heads,
                                                     num_kv_heads, head_dim,
                                                     position, theta);
}
