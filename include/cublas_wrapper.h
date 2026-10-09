#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUBLAS_CHECK(err)                                                       \
    do {                                                                        \
        cublasStatus_t s = (err);                                               \
        if (s != CUBLAS_STATUS_SUCCESS) {                                        \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, s); \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

class CublasWrapper {
public:
    CublasWrapper();
    ~CublasWrapper();

    CublasWrapper(const CublasWrapper&) = delete;
    CublasWrapper& operator=(const CublasWrapper&) = delete;

    void gemm(const float* A, const float* B, float* C,
              int M, int N, int K,
              float alpha = 1.0f, float beta = 0.0f,
              bool trans_a = false, bool trans_b = false);

    void batched_gemm(const float* A, const float* B, float* C,
                      int M, int N, int K, int batch_count,
                      int stride_a, int stride_b, int stride_c,
                      float alpha = 1.0f, float beta = 0.0f,
                      bool trans_a = false, bool trans_b = false);

    void set_stream(cudaStream_t stream);
    cublasHandle_t handle() const;

private:
    cublasHandle_t handle_;
};
