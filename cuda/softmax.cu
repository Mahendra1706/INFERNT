#include "kernels.h"
#include "cuda_utils.h"
#include <cfloat>

__global__ void softmax_kernel(float* output, const float* input,
                                int seq_len) {
    int row = blockIdx.x;
    const float* in_row = input + row * seq_len;
    float* out_row = output + row * seq_len;

    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        max_val = fmaxf(max_val, in_row[i]);
    }

    __shared__ float shared_max;
    if (threadIdx.x == 0) shared_max = -FLT_MAX;
    __syncthreads();
    atomicMax((int*)&shared_max, __float_as_int(max_val));
    __syncthreads();
    max_val = shared_max;

    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        sum_exp += expf(in_row[i] - max_val);
    }

    __shared__ float shared_sum;
    if (threadIdx.x == 0) shared_sum = 0.0f;
    __syncthreads();
    atomicAdd(&shared_sum, sum_exp);
    __syncthreads();

    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        out_row[i] = expf(in_row[i] - max_val) / shared_sum;
    }
}

__global__ void softmax_causal_kernel(float* output, const float* input,
                                       int seq_len, int total_seq_len) {
    int head = blockIdx.x;
    const float* in_row = input + head * total_seq_len;
    float* out_row = output + head * total_seq_len;

    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        max_val = fmaxf(max_val, in_row[i]);
    }

    __shared__ float shared_max;
    if (threadIdx.x == 0) shared_max = -FLT_MAX;
    __syncthreads();
    atomicMax((int*)&shared_max, __float_as_int(max_val));
    __syncthreads();
    max_val = shared_max;

    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        sum_exp += expf(in_row[i] - max_val);
    }

    __shared__ float shared_sum;
    if (threadIdx.x == 0) shared_sum = 0.0f;
    __syncthreads();
    atomicAdd(&shared_sum, sum_exp);
    __syncthreads();

    for (int i = threadIdx.x; i < total_seq_len; i += blockDim.x) {
        if (i < seq_len) {
            out_row[i] = expf(in_row[i] - max_val) / shared_sum;
        } else {
            out_row[i] = 0.0f;
        }
    }
}

void launch_softmax(float* output, const float* input,
                     int batch_size, int seq_len,
                     cudaStream_t stream) {
    int threads = min(seq_len, 1024);
    softmax_kernel<<<batch_size, threads, 0, stream>>>(output, input, seq_len);
}

void launch_softmax_with_mask(float* output, const float* input,
                               int num_heads, int seq_len, int total_seq_len,
                               cudaStream_t stream) {
    int threads = min(total_seq_len, 1024);
    softmax_causal_kernel<<<num_heads, threads, 0, stream>>>(output, input,
                                                              seq_len,
                                                              total_seq_len);
}
