#include "cuda_utils.h"
#include "tensor.h"
#include "config.h"
#include "cublas_wrapper.h"
#include "kernels.h"
#include "kv_cache.h"
#include "weights.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <vector>

bool test_tensor_cpu() {
    printf("\n--- Test: CPU Tensor ---\n");

    auto t = Tensor::zeros({2, 3, 4}, DType::FP32, Device::CPU);
    t.print_info("cpu_tensor");

    if (t.dim() != 3) return false;
    if (t.size(0) != 2 || t.size(1) != 3 || t.size(2) != 4) return false;
    if (t.num_elements() != 24) return false;
    if (t.size_bytes() != 96) return false;
    if (!t.is_cpu()) return false;

    float* data = t.data_as_float();
    for (int i = 0; i < 24; i++) {
        if (data[i] != 0.0f) return false;
    }

    for (int i = 0; i < 24; i++) data[i] = static_cast<float>(i);
    t.print_data(10);

    if (t.strides()[0] != 12 || t.strides()[1] != 4 || t.strides()[2] != 1)
        return false;

    printf("  PASSED\n");
    return true;
}

bool test_tensor_gpu() {
    printf("\n--- Test: GPU Tensor ---\n");

    auto cpu_t = Tensor::zeros({4, 8}, DType::FP32, Device::CPU);
    float* cpu_data = cpu_t.data_as_float();
    for (int i = 0; i < 32; i++) cpu_data[i] = static_cast<float>(i) * 0.1f;

    auto gpu_t = cpu_t.to(Device::CUDA);
    if (!gpu_t.is_cuda()) return false;

    auto cpu_back = gpu_t.to(Device::CPU);
    const float* result = cpu_back.data_as_float();
    for (int i = 0; i < 32; i++) {
        if (std::fabs(result[i] - static_cast<float>(i) * 0.1f) > 1e-6f) return false;
    }

    printf("  PASSED\n");
    return true;
}

bool test_tensor_move() {
    printf("\n--- Test: Tensor Move ---\n");

    auto t1 = Tensor::zeros({3, 3}, DType::FP32, Device::CPU);
    float* d = t1.data_as_float();
    for (int i = 0; i < 9; i++) d[i] = static_cast<float>(i);

    Tensor t2 = std::move(t1);
    if (t1.is_valid()) return false;
    if (!t2.is_valid()) return false;
    if (t2.num_elements() != 9) return false;

    printf("  PASSED\n");
    return true;
}

bool test_config() {
    printf("\n--- Test: ModelConfig ---\n");

    ModelConfig config;
    config.architecture = "LlamaForCausalLM";
    config.vocab_size = 32000;
    config.hidden_size = 2048;
    config.num_layers = 22;
    config.num_attention_heads = 32;
    config.num_key_value_heads = 4;
    config.intermediate_size = 5632;
    config.max_seq_len = 2048;
    config.rms_norm_eps = 1e-5f;
    config.rope_theta = 10000.0f;
    config.derive_computed_fields();
    config.validate();
    config.print();

    if (config.head_dim != 64) return false;
    if (config.num_kv_groups() != 8) return false;

    printf("  PASSED\n");
    return true;
}

bool test_cublas_gemm() {
    printf("\n--- Test: cuBLAS GEMM ---\n");

    int M = 4, N = 3, K = 5;

    auto a_cpu = Tensor::zeros({M, K}, DType::FP32, Device::CPU);
    auto b_cpu = Tensor::zeros({K, N}, DType::FP32, Device::CPU);
    auto c_ref = Tensor::zeros({M, N}, DType::FP32, Device::CPU);

    float* A = a_cpu.data_as_float();
    float* B = b_cpu.data_as_float();
    for (int i = 0; i < M * K; i++) A[i] = static_cast<float>(i + 1) * 0.1f;
    for (int i = 0; i < K * N; i++) B[i] = static_cast<float>(i + 1) * 0.2f;

    float* C_ref = c_ref.data_as_float();
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C_ref[i * N + j] = sum;
        }
    }

    auto a_gpu = a_cpu.to(Device::CUDA);
    auto b_gpu = b_cpu.to(Device::CUDA);
    auto c_gpu = Tensor::zeros({M, N}, DType::FP32, Device::CUDA);

    CublasWrapper cublas;
    cublas.gemm(a_gpu.data_as_float(), b_gpu.data_as_float(),
                c_gpu.data_as_float(), M, N, K);

    auto c_result = c_gpu.to(Device::CPU);
    const float* C_gpu = c_result.data_as_float();

    for (int i = 0; i < M * N; i++) {
        if (std::fabs(C_gpu[i] - C_ref[i]) > 1e-4f) {
            printf("  FAILED at %d: cpu=%.6f gpu=%.6f\n", i, C_ref[i], C_gpu[i]);
            return false;
        }
    }

    printf("  CPU ref: ");
    c_ref.print_data(12);
    printf("  GPU out: ");
    c_result.print_data(12);
    printf("  PASSED\n");
    return true;
}

bool test_rmsnorm() {
    printf("\n--- Test: RMSNorm ---\n");

    int hidden = 8;
    auto input_cpu = Tensor::zeros({1, hidden}, DType::FP32, Device::CPU);
    auto weight_cpu = Tensor::zeros({hidden}, DType::FP32, Device::CPU);

    float* inp = input_cpu.data_as_float();
    float* w = weight_cpu.data_as_float();
    for (int i = 0; i < hidden; i++) {
        inp[i] = static_cast<float>(i + 1) * 0.5f;
        w[i] = 1.0f;
    }

    float eps = 1e-5f;
    float sum_sq = 0.0f;
    for (int i = 0; i < hidden; i++) sum_sq += inp[i] * inp[i];
    float rms = std::sqrt(sum_sq / hidden + eps);
    std::vector<float> expected(hidden);
    for (int i = 0; i < hidden; i++) expected[i] = inp[i] / rms * w[i];

    auto input_gpu = input_cpu.to(Device::CUDA);
    auto weight_gpu = weight_cpu.to(Device::CUDA);
    auto output_gpu = Tensor::zeros({1, hidden}, DType::FP32, Device::CUDA);

    launch_rmsnorm(output_gpu.data_as_float(), input_gpu.data_as_float(),
                   weight_gpu.data_as_float(), 1, hidden, eps);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto output_cpu = output_gpu.to(Device::CPU);
    const float* out = output_cpu.data_as_float();

    for (int i = 0; i < hidden; i++) {
        if (std::fabs(out[i] - expected[i]) > 1e-4f) {
            printf("  FAILED at %d: expected=%.6f got=%.6f\n", i, expected[i], out[i]);
            return false;
        }
    }

    printf("  expected: [");
    for (int i = 0; i < hidden; i++) printf("%.4f%s", expected[i], i < hidden-1 ? ", " : "");
    printf("]\n  got:      [");
    for (int i = 0; i < hidden; i++) printf("%.4f%s", out[i], i < hidden-1 ? ", " : "");
    printf("]\n");
    printf("  PASSED\n");
    return true;
}

bool test_residual() {
    printf("\n--- Test: Residual Add ---\n");

    int n = 16;
    auto a_cpu = Tensor::zeros({n}, DType::FP32, Device::CPU);
    auto b_cpu = Tensor::zeros({n}, DType::FP32, Device::CPU);

    float* a = a_cpu.data_as_float();
    float* b = b_cpu.data_as_float();
    for (int i = 0; i < n; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i) * 2.0f;
    }

    auto a_gpu = a_cpu.to(Device::CUDA);
    auto b_gpu = b_cpu.to(Device::CUDA);
    auto c_gpu = Tensor::zeros({n}, DType::FP32, Device::CUDA);

    launch_residual_add(c_gpu.data_as_float(), a_gpu.data_as_float(),
                        b_gpu.data_as_float(), n);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto c_cpu = c_gpu.to(Device::CPU);
    const float* c = c_cpu.data_as_float();
    for (int i = 0; i < n; i++) {
        float expected = static_cast<float>(i) * 3.0f;
        if (std::fabs(c[i] - expected) > 1e-5f) return false;
    }

    printf("  PASSED\n");
    return true;
}

bool test_rope() {
    printf("\n--- Test: RoPE ---\n");

    int num_heads = 2;
    int head_dim = 4;
    float theta = 10000.0f;
    int position = 3;

    auto q_cpu = Tensor::zeros({num_heads * head_dim}, DType::FP32, Device::CPU);
    auto k_cpu = Tensor::zeros({num_heads * head_dim}, DType::FP32, Device::CPU);

    float* q = q_cpu.data_as_float();
    float* k = k_cpu.data_as_float();
    for (int i = 0; i < num_heads * head_dim; i++) {
        q[i] = 1.0f;
        k[i] = 1.0f;
    }

    std::vector<float> q_expected(num_heads * head_dim);
    std::vector<float> k_expected(num_heads * head_dim);
    for (int h = 0; h < num_heads; h++) {
        for (int d = 0; d < head_dim; d += 2) {
            float freq = 1.0f / std::pow(theta, (float)d / (float)head_dim);
            float angle = position * freq;
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);
            int idx = h * head_dim + d;
            q_expected[idx]   = 1.0f * cos_val - 1.0f * sin_val;
            q_expected[idx+1] = 1.0f * sin_val + 1.0f * cos_val;
            k_expected[idx]   = q_expected[idx];
            k_expected[idx+1] = q_expected[idx+1];
        }
    }

    auto q_gpu = q_cpu.to(Device::CUDA);
    auto k_gpu = k_cpu.to(Device::CUDA);

    launch_rope(q_gpu.data_as_float(), k_gpu.data_as_float(),
                1, num_heads, num_heads, head_dim, position, theta);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto q_result = q_gpu.to(Device::CPU);
    auto k_result = k_gpu.to(Device::CPU);
    const float* qr = q_result.data_as_float();
    const float* kr = k_result.data_as_float();

    for (int i = 0; i < num_heads * head_dim; i++) {
        if (std::fabs(qr[i] - q_expected[i]) > 1e-4f) {
            printf("  FAILED Q at %d: expected=%.6f got=%.6f\n", i, q_expected[i], qr[i]);
            return false;
        }
        if (std::fabs(kr[i] - k_expected[i]) > 1e-4f) {
            printf("  FAILED K at %d: expected=%.6f got=%.6f\n", i, k_expected[i], kr[i]);
            return false;
        }
    }

    printf("  PASSED\n");
    return true;
}

bool test_softmax() {
    printf("\n--- Test: Softmax ---\n");

    int n = 5;
    auto input_cpu = Tensor::zeros({1, n}, DType::FP32, Device::CPU);
    float* inp = input_cpu.data_as_float();
    inp[0] = 1.0f; inp[1] = 2.0f; inp[2] = 3.0f; inp[3] = 4.0f; inp[4] = 5.0f;

    float max_val = 5.0f;
    float sum_exp = 0.0f;
    std::vector<float> expected(n);
    for (int i = 0; i < n; i++) {
        expected[i] = std::exp(inp[i] - max_val);
        sum_exp += expected[i];
    }
    for (int i = 0; i < n; i++) expected[i] /= sum_exp;

    auto input_gpu = input_cpu.to(Device::CUDA);
    auto output_gpu = Tensor::zeros({1, n}, DType::FP32, Device::CUDA);

    launch_softmax(output_gpu.data_as_float(), input_gpu.data_as_float(), 1, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto output_cpu = output_gpu.to(Device::CPU);
    const float* out = output_cpu.data_as_float();

    float total = 0.0f;
    for (int i = 0; i < n; i++) {
        if (std::fabs(out[i] - expected[i]) > 1e-5f) {
            printf("  FAILED at %d: expected=%.6f got=%.6f\n", i, expected[i], out[i]);
            return false;
        }
        total += out[i];
    }

    printf("  sum=%.6f (should be 1.0)\n", total);
    printf("  output: [");
    for (int i = 0; i < n; i++) printf("%.4f%s", out[i], i < n-1 ? ", " : "");
    printf("]\n");
    printf("  PASSED\n");
    return true;
}

bool test_kv_cache() {
    printf("\n--- Test: KV Cache ---\n");

    int num_kv_heads = 2;
    int head_dim = 4;
    int max_seq = 8;

    auto k_cache = Tensor::zeros({num_kv_heads, max_seq, head_dim}, DType::FP32, Device::CUDA);
    auto v_cache = Tensor::zeros({num_kv_heads, max_seq, head_dim}, DType::FP32, Device::CUDA);

    auto k_new_cpu = Tensor::zeros({num_kv_heads * head_dim}, DType::FP32, Device::CPU);
    auto v_new_cpu = Tensor::zeros({num_kv_heads * head_dim}, DType::FP32, Device::CPU);

    float* kn = k_new_cpu.data_as_float();
    float* vn = v_new_cpu.data_as_float();
    for (int i = 0; i < num_kv_heads * head_dim; i++) {
        kn[i] = static_cast<float>(i + 1) * 0.1f;
        vn[i] = static_cast<float>(i + 1) * 0.2f;
    }

    auto k_new_gpu = k_new_cpu.to(Device::CUDA);
    auto v_new_gpu = v_new_cpu.to(Device::CUDA);

    launch_kv_cache_append(k_cache.data_as_float(), v_cache.data_as_float(),
                           k_new_gpu.data_as_float(), v_new_gpu.data_as_float(),
                           num_kv_heads, head_dim, max_seq, 0);

    launch_kv_cache_append(k_cache.data_as_float(), v_cache.data_as_float(),
                           k_new_gpu.data_as_float(), v_new_gpu.data_as_float(),
                           num_kv_heads, head_dim, max_seq, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto k_cpu = k_cache.to(Device::CPU);
    const float* kc = k_cpu.data_as_float();

    for (int h = 0; h < num_kv_heads; h++) {
        float expected = static_cast<float>(h * head_dim + 1) * 0.1f;
        float got = kc[h * max_seq * head_dim + 0 * head_dim + 0];
        if (std::fabs(got - expected) > 1e-5f) {
            printf("  FAILED: head %d pos 0: expected=%.4f got=%.4f\n", h, expected, got);
            return false;
        }
        got = kc[h * max_seq * head_dim + 1 * head_dim + 0];
        if (std::fabs(got - expected) > 1e-5f) {
            printf("  FAILED: head %d pos 1: expected=%.4f got=%.4f\n", h, expected, got);
            return false;
        }
    }

    printf("  PASSED\n");
    return true;
}

bool test_attention_gqa() {
    printf("\n--- Test: Attention with GQA ---\n");

    int num_q_heads = 4;
    int num_kv_heads = 2;
    int head_dim = 4;
    int max_seq = 16;
    int seq_len = 3;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto q_cpu = Tensor::zeros({num_q_heads * head_dim}, DType::FP32, Device::CPU);
    float* q = q_cpu.data_as_float();
    for (int i = 0; i < num_q_heads * head_dim; i++) q[i] = 1.0f;

    auto k_cache_cpu = Tensor::zeros({num_kv_heads, max_seq, head_dim}, DType::FP32, Device::CPU);
    auto v_cache_cpu = Tensor::zeros({num_kv_heads, max_seq, head_dim}, DType::FP32, Device::CPU);
    float* kc = k_cache_cpu.data_as_float();
    float* vc = v_cache_cpu.data_as_float();

    for (int h = 0; h < num_kv_heads; h++) {
        for (int s = 0; s < seq_len; s++) {
            for (int d = 0; d < head_dim; d++) {
                kc[h * max_seq * head_dim + s * head_dim + d] = 1.0f;
                vc[h * max_seq * head_dim + s * head_dim + d] = static_cast<float>(s + 1);
            }
        }
    }

    auto q_gpu = q_cpu.to(Device::CUDA);
    auto k_cache_gpu = k_cache_cpu.to(Device::CUDA);
    auto v_cache_gpu = v_cache_cpu.to(Device::CUDA);
    auto scores_gpu = Tensor::zeros({num_q_heads, seq_len}, DType::FP32, Device::CUDA);
    auto attn_out_gpu = Tensor::zeros({num_q_heads * head_dim}, DType::FP32, Device::CUDA);

    launch_attention_scores(scores_gpu.data_as_float(), q_gpu.data_as_float(),
                            k_cache_gpu.data_as_float(),
                            num_q_heads, num_kv_heads, head_dim,
                            seq_len, max_seq, scale);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto scores_cpu = scores_gpu.to(Device::CPU);
    const float* sc = scores_cpu.data_as_float();
    printf("  raw scores (head 0): [");
    for (int i = 0; i < seq_len; i++) printf("%.4f%s", sc[i], i < seq_len-1 ? ", " : "");
    printf("]\n");

    float expected_score = head_dim * scale;
    for (int h = 0; h < num_q_heads; h++) {
        for (int s = 0; s < seq_len; s++) {
            if (std::fabs(sc[h * seq_len + s] - expected_score) > 1e-4f) {
                printf("  FAILED score at head=%d pos=%d: expected=%.4f got=%.4f\n",
                       h, s, expected_score, sc[h * seq_len + s]);
                return false;
            }
        }
    }

    launch_softmax(scores_gpu.data_as_float(), scores_gpu.data_as_float(),
                   num_q_heads, seq_len);
    CUDA_CHECK(cudaDeviceSynchronize());

    scores_cpu = scores_gpu.to(Device::CPU);
    sc = scores_cpu.data_as_float();
    printf("  softmax (head 0): [");
    for (int i = 0; i < seq_len; i++) printf("%.4f%s", sc[i], i < seq_len-1 ? ", " : "");
    printf("]\n");

    launch_attention_values(attn_out_gpu.data_as_float(), scores_gpu.data_as_float(),
                            v_cache_gpu.data_as_float(),
                            num_q_heads, num_kv_heads, head_dim,
                            seq_len, max_seq);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto attn_out_cpu = attn_out_gpu.to(Device::CPU);
    const float* ao = attn_out_cpu.data_as_float();
    printf("  attn output (head 0): [");
    for (int d = 0; d < head_dim; d++) printf("%.4f%s", ao[d], d < head_dim-1 ? ", " : "");
    printf("]\n");

    float expected_val = (1.0f / 3.0f) * 1.0f + (1.0f / 3.0f) * 2.0f + (1.0f / 3.0f) * 3.0f;
    for (int h = 0; h < num_q_heads; h++) {
        for (int d = 0; d < head_dim; d++) {
            if (std::fabs(ao[h * head_dim + d] - expected_val) > 1e-3f) {
                printf("  FAILED output at head=%d dim=%d: expected=%.4f got=%.4f\n",
                       h, d, expected_val, ao[h * head_dim + d]);
                return false;
            }
        }
    }

    printf("  GQA mapping verified: %d Q heads → %d KV heads (groups of %d)\n",
           num_q_heads, num_kv_heads, num_q_heads / num_kv_heads);
    printf("  PASSED\n");
    return true;
}

int main() {
    printf("========================================\n");
    printf("  INFERNT — CUDA LLM Inference Engine\n");
    printf("  Phases 1-8 Validation\n");
    printf("========================================\n\n");

    cuda_init(0);
    GPUInfo gpu = cuda_get_device_info(0);
    cuda_print_device_info(gpu);

    int passed = 0;
    int failed = 0;

    auto run_test = [&](const char* name, bool (*fn)()) {
        try {
            if (fn()) {
                passed++;
            } else {
                printf("  FAILED: %s\n", name);
                failed++;
            }
        } catch (const std::exception& e) {
            printf("  EXCEPTION in %s: %s\n", name, e.what());
            failed++;
        }
    };

    printf("\n=== Phase 1: Foundation ===\n");
    run_test("CPU Tensor", test_tensor_cpu);
    run_test("GPU Tensor", test_tensor_gpu);
    run_test("Move Semantics", test_tensor_move);
    run_test("ModelConfig", test_config);

    printf("\n=== Phase 2: cuBLAS ===\n");
    run_test("cuBLAS GEMM", test_cublas_gemm);

    printf("\n=== Phase 4: RMSNorm + Residual ===\n");
    run_test("RMSNorm", test_rmsnorm);
    run_test("Residual Add", test_residual);

    printf("\n=== Phase 5: RoPE ===\n");
    run_test("RoPE", test_rope);

    printf("\n=== Phase 6: Softmax ===\n");
    run_test("Softmax", test_softmax);

    printf("\n=== Phase 7: KV Cache ===\n");
    run_test("KV Cache Append", test_kv_cache);

    printf("\n=== Phase 8: Attention + GQA ===\n");
    run_test("Attention GQA", test_attention_gqa);

    printf("\n========================================\n");
    printf("  Results: %d passed, %d failed\n", passed, failed);
    printf("========================================\n");

    return failed > 0 ? 1 : 0;
}
