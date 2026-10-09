#include "kernels.h"
#include "cuda_utils.h"

__global__ void rmsnorm_kernel(float* output, const float* input,
                               const float* weight, int hidden_size, float eps) {
    int row = blockIdx.x;
    const float* x = input + row * hidden_size;
    float* o = output + row * hidden_size;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        sum_sq += x[i] * x[i];
    }

    __shared__ float shared_sum;
    if (threadIdx.x == 0) shared_sum = 0.0f;
    __syncthreads();

    atomicAdd(&shared_sum, sum_sq);
    __syncthreads();

    float rms = rsqrtf(shared_sum / hidden_size + eps);

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        o[i] = x[i] * rms * weight[i];
    }
}

void launch_rmsnorm(float* output, const float* input, const float* weight,
                    int batch_size, int hidden_size, float eps,
                    cudaStream_t stream) {
    int threads = min(hidden_size, 1024);
    rmsnorm_kernel<<<batch_size, threads, 0, stream>>>(output, input, weight,
                                                        hidden_size, eps);
}
