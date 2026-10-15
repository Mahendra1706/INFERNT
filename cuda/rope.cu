#include "kernels.h"
#include "cuda_utils.h"
#include <cmath>

__global__ void rope_kernel(float* q, float* k,
                            int num_q_heads, int num_kv_heads,
                            int head_dim, int position, float theta) {
    int head = blockIdx.x;
    int d = threadIdx.x;
    int half = head_dim / 2;
    if (d >= half) return;

    float freq = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
    float angle = position * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    if (head < num_q_heads) {
        int offset0 = head * head_dim + d;
        int offset1 = head * head_dim + d + half;
        float q0 = q[offset0];
        float q1 = q[offset1];
        q[offset0] = q0 * cos_val - q1 * sin_val;
        q[offset1] = q1 * cos_val + q0 * sin_val;
    }

    if (head < num_kv_heads) {
        int offset0 = head * head_dim + d;
        int offset1 = head * head_dim + d + half;
        float k0 = k[offset0];
        float k1 = k[offset1];
        k[offset0] = k0 * cos_val - k1 * sin_val;
        k[offset1] = k1 * cos_val + k0 * sin_val;
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
