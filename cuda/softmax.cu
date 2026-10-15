#include "kernels.h"
#include "cuda_utils.h"
#include <cfloat>

__global__ void softmax_kernel(float* output, const float* input,
                                int seq_len) {
    int row = blockIdx.x;
    const float* in_row = input + row * seq_len;
    float* out_row = output + row * seq_len;

    __shared__ float s_val[256];

    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        max_val = fmaxf(max_val, in_row[i]);
    }
    s_val[threadIdx.x] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_val[threadIdx.x] = fmaxf(s_val[threadIdx.x], s_val[threadIdx.x + s]);
        }
        __syncthreads();
    }
    max_val = s_val[0];
    __syncthreads();

    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        sum_exp += expf(in_row[i] - max_val);
    }
    s_val[threadIdx.x] = sum_exp;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_val[threadIdx.x] += s_val[threadIdx.x + s];
        }
        __syncthreads();
    }
    float total_sum = s_val[0];
    __syncthreads();

    float inv_sum = 1.0f / (total_sum > 0.0f ? total_sum : 1.0f);
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        out_row[i] = expf(in_row[i] - max_val) * inv_sum;
    }
}

__global__ void softmax_causal_kernel(float* output, const float* input,
                                       int seq_len, int total_seq_len) {
    int head = blockIdx.x;
    const float* in_row = input + head * total_seq_len;
    float* out_row = output + head * total_seq_len;

    __shared__ float s_val[256];

    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        max_val = fmaxf(max_val, in_row[i]);
    }
    s_val[threadIdx.x] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_val[threadIdx.x] = fmaxf(s_val[threadIdx.x], s_val[threadIdx.x + s]);
        }
        __syncthreads();
    }
    max_val = s_val[0];
    __syncthreads();

    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        sum_exp += expf(in_row[i] - max_val);
    }
    s_val[threadIdx.x] = sum_exp;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_val[threadIdx.x] += s_val[threadIdx.x + s];
        }
        __syncthreads();
    }
    float total_sum = s_val[0];
    __syncthreads();

    float inv_sum = 1.0f / (total_sum > 0.0f ? total_sum : 1.0f);
    for (int i = threadIdx.x; i < total_seq_len; i += blockDim.x) {
        if (i < seq_len) {
            out_row[i] = expf(in_row[i] - max_val) * inv_sum;
        } else {
            out_row[i] = 0.0f;
        }
    }
}

void launch_softmax(float* output, const float* input,
                     int batch_size, int seq_len,
                     cudaStream_t stream) {
    int threads = 256;
    softmax_kernel<<<batch_size, threads, 0, stream>>>(output, input, seq_len);
}

void launch_softmax_with_mask(float* output, const float* input,
                               int num_heads, int seq_len, int total_seq_len,
                               cudaStream_t stream) {
    int threads = 256;
    softmax_causal_kernel<<<num_heads, threads, 0, stream>>>(output, input,
                                                             seq_len,
                                                             total_seq_len);
}
