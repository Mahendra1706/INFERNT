#include "cublas_wrapper.h"
#include <cstdio>
#include <cstdlib>

CublasWrapper::CublasWrapper() {
    CUBLAS_CHECK(cublasCreate(&handle_));
}

CublasWrapper::~CublasWrapper() {
    cublasDestroy(handle_);
}

void CublasWrapper::set_stream(cudaStream_t stream) {
    CUBLAS_CHECK(cublasSetStream(handle_, stream));
}

cublasHandle_t CublasWrapper::handle() const {
    return handle_;
}

void CublasWrapper::gemm(const float* A, const float* B, float* C,
                          int M, int N, int K,
                          float alpha, float beta,
                          bool trans_a, bool trans_b) {
    cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    int lda = trans_a ? M : K;
    int ldb = trans_b ? K : N;
    int ldc = N;

    CUBLAS_CHECK(cublasSgemm(handle_,
                             op_b, op_a,
                             N, M, K,
                             &alpha,
                             B, ldb,
                             A, lda,
                             &beta,
                             C, ldc));
}

void CublasWrapper::batched_gemm(const float* A, const float* B, float* C,
                                  int M, int N, int K, int batch_count,
                                  int stride_a, int stride_b, int stride_c,
                                  float alpha, float beta,
                                  bool trans_a, bool trans_b) {
    cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    int lda = trans_a ? M : K;
    int ldb = trans_b ? K : N;
    int ldc = N;

    long long stride_a_ll = stride_a;
    long long stride_b_ll = stride_b;
    long long stride_c_ll = stride_c;

    CUBLAS_CHECK(cublasSgemmStridedBatched(handle_,
                                           op_b, op_a,
                                           N, M, K,
                                           &alpha,
                                           B, ldb, stride_b_ll,
                                           A, lda, stride_a_ll,
                                           &beta,
                                           C, ldc, stride_c_ll,
                                           batch_count));
}
