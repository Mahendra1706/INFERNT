#include "cuda_utils.h"
#include "tensor.h"
#include "config.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <stdexcept>

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

    for (int i = 0; i < 24; i++) {
        data[i] = static_cast<float>(i);
    }
    t.print_data(10);

    printf("  strides: [");
    for (int i = 0; i < t.dim(); i++) {
        if (i > 0) printf(", ");
        printf("%d", t.strides()[i]);
    }
    printf("]\n");

    if (t.strides()[0] != 12 || t.strides()[1] != 4 || t.strides()[2] != 1)
        return false;

    printf("  PASSED\n");
    return true;
}

bool test_tensor_gpu() {
    printf("\n--- Test: GPU Tensor ---\n");

    auto cpu_t = Tensor::zeros({4, 8}, DType::FP32, Device::CPU);
    float* cpu_data = cpu_t.data_as_float();
    for (int i = 0; i < 32; i++) {
        cpu_data[i] = static_cast<float>(i) * 0.1f;
    }
    cpu_t.print_info("cpu_source");
    cpu_t.print_data(10);

    auto gpu_t = cpu_t.to(Device::CUDA);
    gpu_t.print_info("gpu_tensor");

    if (!gpu_t.is_cuda()) return false;
    if (gpu_t.num_elements() != 32) return false;

    auto cpu_back = gpu_t.to(Device::CPU);
    cpu_back.print_info("cpu_roundtrip");
    cpu_back.print_data(10);

    const float* result = cpu_back.data_as_float();
    for (int i = 0; i < 32; i++) {
        float expected = static_cast<float>(i) * 0.1f;
        if (std::fabs(result[i] - expected) > 1e-6f) {
            printf("  FAILED: element %d: expected %.6f, got %.6f\n",
                   i, expected, result[i]);
            return false;
        }
    }

    printf("  PASSED\n");
    return true;
}

bool test_tensor_move() {
    printf("\n--- Test: Tensor Move Semantics ---\n");

    auto t1 = Tensor::zeros({3, 3}, DType::FP32, Device::CPU);
    float* d = t1.data_as_float();
    for (int i = 0; i < 9; i++) d[i] = static_cast<float>(i);

    Tensor t2 = std::move(t1);
    if (t1.is_valid()) return false;
    if (!t2.is_valid()) return false;
    if (t2.num_elements() != 9) return false;

    const float* d2 = t2.data_as_float();
    for (int i = 0; i < 9; i++) {
        if (d2[i] != static_cast<float>(i)) return false;
    }

    printf("  PASSED\n");
    return true;
}

bool test_tensor_reshape() {
    printf("\n--- Test: Tensor Reshape ---\n");

    auto t = Tensor::zeros({2, 3, 4}, DType::FP32, Device::CPU);
    float* d = t.data_as_float();
    for (int i = 0; i < 24; i++) d[i] = static_cast<float>(i);

    auto r = t.reshape({6, 4});
    r.print_info("reshaped");

    if (r.dim() != 2) return false;
    if (r.size(0) != 6 || r.size(1) != 4) return false;
    if (r.num_elements() != 24) return false;

    const float* rd = r.data_as_float();
    for (int i = 0; i < 24; i++) {
        if (rd[i] != static_cast<float>(i)) return false;
    }

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

    if (config.head_dim != 64) {
        printf("  FAILED: head_dim expected 64, got %d\n", config.head_dim);
        return false;
    }
    if (config.num_kv_groups() != 8) {
        printf("  FAILED: num_kv_groups expected 8, got %d\n", config.num_kv_groups());
        return false;
    }

    printf("  PASSED\n");
    return true;
}

int main() {
    printf("========================================\n");
    printf("  INFERNT — CUDA LLM Inference Engine\n");
    printf("  Phase 1: Foundation\n");
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

    run_test("CPU Tensor", test_tensor_cpu);
    run_test("GPU Tensor", test_tensor_gpu);
    run_test("Move Semantics", test_tensor_move);
    run_test("Reshape", test_tensor_reshape);
    run_test("ModelConfig", test_config);

    printf("\n========================================\n");
    printf("  Results: %d passed, %d failed\n", passed, failed);
    printf("========================================\n");

    return failed > 0 ? 1 : 0;
}
