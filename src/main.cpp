#include "cuda_utils.h"
#include "tensor.h"
#include "config.h"
#include "cublas_wrapper.h"
#include "kernels.h"
#include "kv_cache.h"
#include "weights.h"
#include "architecture.h"
#include "tokenizer.h"
#include "checkpoint_loader.h"
#include "engine.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <string>

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
    int half = head_dim / 2;
    for (int h = 0; h < num_heads; h++) {
        for (int d = 0; d < half; d++) {
            float freq = 1.0f / std::pow(theta, (float)(2 * d) / (float)head_dim);
            float angle = position * freq;
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);
            int idx0 = h * head_dim + d;
            int idx1 = h * head_dim + d + half;
            q_expected[idx0] = 1.0f * cos_val - 1.0f * sin_val;
            q_expected[idx1] = 1.0f * cos_val + 1.0f * sin_val;
            k_expected[idx0] = q_expected[idx0];
            k_expected[idx1] = q_expected[idx1];
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

    for (int i = 0; i < n; i++) {
        if (std::fabs(out[i] - expected[i]) > 1e-5f) {
            printf("  FAILED at %d: expected=%.6f got=%.6f\n", i, expected[i], out[i]);
            return false;
        }
    }

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
        if (std::fabs(got - expected) > 1e-5f) return false;
        got = kc[h * max_seq * head_dim + 1 * head_dim + 0];
        if (std::fabs(got - expected) > 1e-5f) return false;
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

    launch_softmax(scores_gpu.data_as_float(), scores_gpu.data_as_float(),
                   num_q_heads, seq_len);
    CUDA_CHECK(cudaDeviceSynchronize());

    launch_attention_values(attn_out_gpu.data_as_float(), scores_gpu.data_as_float(),
                            v_cache_gpu.data_as_float(),
                            num_q_heads, num_kv_heads, head_dim,
                            seq_len, max_seq);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto attn_out_cpu = attn_out_gpu.to(Device::CPU);
    const float* ao = attn_out_cpu.data_as_float();

    float expected_val = (1.0f / 3.0f) * 1.0f + (1.0f / 3.0f) * 2.0f + (1.0f / 3.0f) * 3.0f;
    for (int h = 0; h < num_q_heads; h++) {
        for (int d = 0; d < head_dim; d++) {
            if (std::fabs(ao[h * head_dim + d] - expected_val) > 1e-3f) {
                return false;
            }
        }
    }

    printf("  PASSED\n");
    return true;
}

bool test_mlp_activation() {
    printf("\n--- Test: SiLU MLP Activation ---\n");
    int n = 4;
    auto gate_cpu = Tensor::zeros({n}, DType::FP32, Device::CPU);
    auto up_cpu = Tensor::zeros({n}, DType::FP32, Device::CPU);
    float* g = gate_cpu.data_as_float();
    float* u = up_cpu.data_as_float();
    
    for(int i = 0; i < n; i++) {
        g[i] = static_cast<float>(i - 1);
        u[i] = 2.0f;
    }

    auto gate_gpu = gate_cpu.to(Device::CUDA);
    auto up_gpu = up_cpu.to(Device::CUDA);
    auto out_gpu = Tensor::zeros({n}, DType::FP32, Device::CUDA);

    launch_silu_mul(out_gpu.data_as_float(), gate_gpu.data_as_float(), up_gpu.data_as_float(), n);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto out_cpu = out_gpu.to(Device::CPU);
    const float* o = out_cpu.data_as_float();

    for(int i = 0; i < n; i++) {
        float expected = g[i] / (1.0f + std::exp(-g[i])) * u[i];
        if (std::fabs(o[i] - expected) > 1e-4f) return false;
    }
    printf("  PASSED\n");
    return true;
}

bool test_llama_architecture() {
    printf("\n--- Test: LlamaArchitecture Forward Pass ---\n");
    ModelConfig config;
    config.architecture = "LlamaForCausalLM";
    config.vocab_size = 100;
    config.hidden_size = 32;
    config.num_layers = 2;
    config.num_attention_heads = 4;
    config.num_key_value_heads = 4;
    config.intermediate_size = 64;
    config.max_seq_len = 16;
    config.rms_norm_eps = 1e-5f;
    config.rope_theta = 10000.0f;
    config.derive_computed_fields();

    int q_size = config.num_attention_heads * config.head_dim;
    int kv_size = config.num_key_value_heads * config.head_dim;

    ModelWeights weights;
    weights.final_norm = Tensor::zeros({config.hidden_size}, DType::FP32, Device::CUDA);
    weights.lm_head = Tensor::zeros({config.vocab_size, config.hidden_size}, DType::FP32, Device::CUDA);
    
    for (int i = 0; i < config.num_layers; i++) {
        TransformerLayerWeights lw;
        lw.attn_norm = Tensor::zeros({config.hidden_size}, DType::FP32, Device::CUDA);
        lw.q_proj = Tensor::zeros({q_size, config.hidden_size}, DType::FP32, Device::CUDA);
        lw.k_proj = Tensor::zeros({kv_size, config.hidden_size}, DType::FP32, Device::CUDA);
        lw.v_proj = Tensor::zeros({kv_size, config.hidden_size}, DType::FP32, Device::CUDA);
        lw.o_proj = Tensor::zeros({config.hidden_size, q_size}, DType::FP32, Device::CUDA);
        lw.ffn_norm = Tensor::zeros({config.hidden_size}, DType::FP32, Device::CUDA);
        lw.gate_proj = Tensor::zeros({config.intermediate_size, config.hidden_size}, DType::FP32, Device::CUDA);
        lw.up_proj = Tensor::zeros({config.intermediate_size, config.hidden_size}, DType::FP32, Device::CUDA);
        lw.down_proj = Tensor::zeros({config.hidden_size, config.intermediate_size}, DType::FP32, Device::CUDA);
        weights.layers.push_back(std::move(lw));
    }

    CublasWrapper cublas;
    LlamaArchitecture llama(cublas);
    llama.allocate_activations(config, 1);

    KVCache kv_cache;
    kv_cache.allocate(config);
    kv_cache.reset();

    auto input_emb = Tensor::zeros({1, config.hidden_size}, DType::FP32, Device::CUDA);
    auto logits = Tensor::zeros({1, config.vocab_size}, DType::FP32, Device::CUDA);

    llama.forward(input_emb, weights, kv_cache, 0, logits);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("  PASSED\n");
    return true;
}

bool test_tokenizer() {
    printf("\n--- Test: SentencePiece BPE Tokenizer ---\n");
    Tokenizer tok;
    std::vector<std::string> vocab = {
        "<unk>", "<s>", "</s>", "\xe2\x96\x81", "H", "e", "l", "o", "w", "r", "d", "!",
        "He", "ll", "lo", " \xe2\x96\x81Hello", "\xe2\x96\x81world"
    };
    std::vector<float> scores;
    for (size_t i = 0; i < vocab.size(); i++) {
        scores.push_back(static_cast<float>(i));
    }
    tok.load_from_vocab(vocab, scores);

    std::vector<int> tokens = tok.encode("Hello", true, false);
    if (tokens.empty() || tokens[0] != tok.bos_id()) {
        printf("  FAILED: BOS token missing\n");
        return false;
    }

    std::string decoded = tok.decode(tokens);
    if (decoded.find("Hello") == std::string::npos) {
        printf("  FAILED: decoded string '%s' does not match 'Hello'\n", decoded.c_str());
        return false;
    }

    printf("  Encoded 'Hello' -> %zu tokens, decoded back -> '%s'\n", tokens.size(), decoded.c_str());
    printf("  PASSED\n");
    return true;
}

bool test_checkpoint_loader() {
    printf("\n--- Test: Binary Checkpoint Loader ---\n");
    std::string test_cfg_path = "/tmp/infernt_test_config.json";
    std::string test_bin_path = "/tmp/infernt_test_model.bin";

    {
        std::ofstream f(test_cfg_path);
        f << "{\n"
          << "  \"architecture\": \"LlamaForCausalLM\",\n"
          << "  \"vocab_size\": 16,\n"
          << "  \"hidden_size\": 16,\n"
          << "  \"num_layers\": 1,\n"
          << "  \"num_attention_heads\": 2,\n"
          << "  \"num_key_value_heads\": 2,\n"
          << "  \"intermediate_size\": 32,\n"
          << "  \"max_seq_len\": 32,\n"
          << "  \"rms_norm_eps\": 1e-5,\n"
          << "  \"rope_theta\": 10000.0\n"
          << "}\n";
    }

    ModelConfig cfg = CheckpointLoader::load_config(test_cfg_path);
    if (cfg.vocab_size != 16 || cfg.hidden_size != 16 || cfg.num_layers != 1) {
        printf("  FAILED: Config values mismatch\n");
        return false;
    }

    {
        std::ofstream f(test_bin_path, std::ios::binary);
        uint32_t magic = 0x494E4654;
        uint32_t version = 1;
        uint32_t count = 3;
        f.write(reinterpret_cast<char*>(&magic), 4);
        f.write(reinterpret_cast<char*>(&version), 4);
        f.write(reinterpret_cast<char*>(&count), 4);

        auto write_tensor = [&](const std::string& name, const std::vector<int>& dims) {
            int32_t nl = static_cast<int32_t>(name.length());
            f.write(reinterpret_cast<char*>(&nl), 4);
            f.write(name.data(), nl);
            int32_t nd = static_cast<int32_t>(dims.size());
            f.write(reinterpret_cast<char*>(&nd), 4);
            int64_t total = 1;
            for (int d : dims) {
                int32_t dim_val = d;
                f.write(reinterpret_cast<char*>(&dim_val), 4);
                total *= d;
            }
            f.write(reinterpret_cast<char*>(&total), 8);
            std::vector<float> data(total, 0.5f);
            f.write(reinterpret_cast<char*>(data.data()), total * sizeof(float));
        };

        write_tensor("model.embed_tokens.weight", {16, 16});
        write_tensor("model.norm.weight", {16});
        write_tensor("lm_head.weight", {16, 16});
    }

    ModelWeights weights;
    CheckpointLoader::load_weights(test_bin_path, cfg, weights, Device::CUDA);

    if (!weights.embedding.is_valid() || !weights.final_norm.is_valid() || !weights.lm_head.is_valid()) {
        printf("  FAILED: Tensors not properly populated\n");
        return false;
    }

    printf("  PASSED\n");
    return true;
}

bool test_embedding_and_sampling() {
    printf("\n--- Test: Embedding Lookup & Parallel Argmax ---\n");
    int vocab_size = 10;
    int hidden_size = 8;

    auto table_cpu = Tensor::zeros({vocab_size, hidden_size}, DType::FP32, Device::CPU);
    float* td = table_cpu.data_as_float();
    for (int i = 0; i < vocab_size * hidden_size; i++) td[i] = static_cast<float>(i);

    auto table_gpu = table_cpu.to(Device::CUDA);
    auto out_gpu = Tensor::zeros({hidden_size}, DType::FP32, Device::CUDA);

    int target_token = 3;
    launch_embedding_lookup(out_gpu.data_as_float(), table_gpu.data_as_float(), target_token, hidden_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto out_cpu = out_gpu.to(Device::CPU);
    const float* od = out_cpu.data_as_float();
    for (int i = 0; i < hidden_size; i++) {
        float expected = static_cast<float>(target_token * hidden_size + i);
        if (std::fabs(od[i] - expected) > 1e-5f) {
            printf("  FAILED embedding lookup at %d: expected=%.2f got=%.2f\n", i, expected, od[i]);
            return false;
        }
    }

    int test_vocab = 100;
    auto logits_cpu = Tensor::zeros({test_vocab}, DType::FP32, Device::CPU);
    float* ld = logits_cpu.data_as_float();
    for (int i = 0; i < test_vocab; i++) ld[i] = static_cast<float>(i) * 0.01f;
    ld[42] = 99.0f;

    auto logits_gpu = logits_cpu.to(Device::CUDA);
    auto out_token_gpu = Tensor::zeros({1}, DType::FP32, Device::CUDA);

    launch_argmax(reinterpret_cast<int*>(out_token_gpu.data_as_float()), logits_gpu.data_as_float(), test_vocab);
    CUDA_CHECK(cudaDeviceSynchronize());

    int sampled_idx = 0;
    cudaMemcpy(&sampled_idx, out_token_gpu.data_as_float(), sizeof(int), cudaMemcpyDeviceToHost);

    if (sampled_idx != 42) {
        printf("  FAILED argmax sampling: expected=42 got=%d\n", sampled_idx);
        return false;
    }

    printf("  PASSED\n");
    return true;
}

bool test_autoregressive_generation() {
    printf("\n--- Test: Autoregressive Loop & Latency Profiling ---\n");
    ModelConfig config;
    config.architecture = "LlamaForCausalLM";
    config.vocab_size = 16;
    config.hidden_size = 16;
    config.num_layers = 2;
    config.num_attention_heads = 2;
    config.num_key_value_heads = 2;
    config.intermediate_size = 32;
    config.max_seq_len = 32;
    config.rms_norm_eps = 1e-5f;
    config.rope_theta = 10000.0f;
    config.derive_computed_fields();

    int q_size = config.num_attention_heads * config.head_dim;
    int kv_size = config.num_key_value_heads * config.head_dim;

    ModelWeights weights;
    weights.embedding = Tensor::zeros({config.vocab_size, config.hidden_size}, DType::FP32, Device::CUDA);
    weights.final_norm = Tensor::zeros({config.hidden_size}, DType::FP32, Device::CUDA);
    weights.lm_head = Tensor::zeros({config.vocab_size, config.hidden_size}, DType::FP32, Device::CUDA);

    for (int i = 0; i < config.num_layers; i++) {
        TransformerLayerWeights lw;
        lw.attn_norm = Tensor::zeros({config.hidden_size}, DType::FP32, Device::CUDA);
        lw.q_proj = Tensor::zeros({q_size, config.hidden_size}, DType::FP32, Device::CUDA);
        lw.k_proj = Tensor::zeros({kv_size, config.hidden_size}, DType::FP32, Device::CUDA);
        lw.v_proj = Tensor::zeros({kv_size, config.hidden_size}, DType::FP32, Device::CUDA);
        lw.o_proj = Tensor::zeros({config.hidden_size, q_size}, DType::FP32, Device::CUDA);
        lw.ffn_norm = Tensor::zeros({config.hidden_size}, DType::FP32, Device::CUDA);
        lw.gate_proj = Tensor::zeros({config.intermediate_size, config.hidden_size}, DType::FP32, Device::CUDA);
        lw.up_proj = Tensor::zeros({config.intermediate_size, config.hidden_size}, DType::FP32, Device::CUDA);
        lw.down_proj = Tensor::zeros({config.hidden_size, config.intermediate_size}, DType::FP32, Device::CUDA);
        weights.layers.push_back(std::move(lw));
    }

    Tokenizer tok;
    std::vector<std::string> vocab = {
        "<unk>", "<s>", "</s>", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"
    };
    std::vector<float> scores(16, 1.0f);
    tok.load_from_vocab(vocab, scores);

    InferenceEngine engine;
    engine.initialize(config, std::move(weights), std::move(tok));

    GenerationConfig gen_cfg;
    gen_cfg.max_new_tokens = 5;
    gen_cfg.stream_stdout = false;
    gen_cfg.temperature = 0.0f;

    GenerationStats stats;
    std::string output = engine.generate("abc", gen_cfg, &stats);

    printf("  Generated tokens: %d\n", stats.generated_tokens);
    printf("  Prefill latency: %.2f ms\n", stats.prefill_ms);
    printf("  Decode latency:  %.2f ms\n", stats.decode_ms);
    printf("  Total latency:   %.2f ms\n", stats.total_ms);
    printf("  Throughput:      %.2f tok/s\n", stats.tokens_per_second);

    if (stats.generated_tokens == 0) {
        printf("  FAILED: No tokens generated\n");
        return false;
    }

    printf("  PASSED\n");
    return true;
}

bool test_architecture_generalization() {
    printf("\n--- Test: Architecture Generalization ---\n");
    ModelConfig non_tinyllama_config;
    non_tinyllama_config.architecture = "LlamaForCausalLM";
    non_tinyllama_config.vocab_size = 500;
    non_tinyllama_config.hidden_size = 128;
    non_tinyllama_config.num_layers = 3;
    non_tinyllama_config.num_attention_heads = 8;
    non_tinyllama_config.num_key_value_heads = 2;
    non_tinyllama_config.intermediate_size = 256;
    non_tinyllama_config.max_seq_len = 64;
    non_tinyllama_config.rms_norm_eps = 1e-6f;
    non_tinyllama_config.rope_theta = 50000.0f;
    non_tinyllama_config.derive_computed_fields();
    non_tinyllama_config.validate();

    if (non_tinyllama_config.head_dim != 16) return false;
    if (non_tinyllama_config.num_kv_groups() != 4) return false;

    CublasWrapper cublas;
    LlamaArchitecture arch(cublas);
    arch.allocate_activations(non_tinyllama_config, 1);

    KVCache kv;
    kv.allocate(non_tinyllama_config);
    kv.reset();

    ModelWeights weights;
    weights.final_norm = Tensor::zeros({non_tinyllama_config.hidden_size}, DType::FP32, Device::CUDA);
    weights.lm_head = Tensor::zeros({non_tinyllama_config.vocab_size, non_tinyllama_config.hidden_size}, DType::FP32, Device::CUDA);
    
    int q_size = non_tinyllama_config.num_attention_heads * non_tinyllama_config.head_dim;
    int kv_size = non_tinyllama_config.num_key_value_heads * non_tinyllama_config.head_dim;

    for (int i = 0; i < non_tinyllama_config.num_layers; i++) {
        TransformerLayerWeights lw;
        lw.attn_norm = Tensor::zeros({non_tinyllama_config.hidden_size}, DType::FP32, Device::CUDA);
        lw.q_proj = Tensor::zeros({q_size, non_tinyllama_config.hidden_size}, DType::FP32, Device::CUDA);
        lw.k_proj = Tensor::zeros({kv_size, non_tinyllama_config.hidden_size}, DType::FP32, Device::CUDA);
        lw.v_proj = Tensor::zeros({kv_size, non_tinyllama_config.hidden_size}, DType::FP32, Device::CUDA);
        lw.o_proj = Tensor::zeros({non_tinyllama_config.hidden_size, q_size}, DType::FP32, Device::CUDA);
        lw.ffn_norm = Tensor::zeros({non_tinyllama_config.hidden_size}, DType::FP32, Device::CUDA);
        lw.gate_proj = Tensor::zeros({non_tinyllama_config.intermediate_size, non_tinyllama_config.hidden_size}, DType::FP32, Device::CUDA);
        lw.up_proj = Tensor::zeros({non_tinyllama_config.intermediate_size, non_tinyllama_config.hidden_size}, DType::FP32, Device::CUDA);
        lw.down_proj = Tensor::zeros({non_tinyllama_config.hidden_size, non_tinyllama_config.intermediate_size}, DType::FP32, Device::CUDA);
        weights.layers.push_back(std::move(lw));
    }

    auto input_emb = Tensor::zeros({1, non_tinyllama_config.hidden_size}, DType::FP32, Device::CUDA);
    auto logits = Tensor::zeros({1, non_tinyllama_config.vocab_size}, DType::FP32, Device::CUDA);

    arch.forward(input_emb, weights, kv, 0, logits);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("  Arbitrary dimension model forward pass succeeded\n");
    printf("  PASSED\n");
    return true;
}

int main(int argc, char** argv) {
    std::string model_dir;
    std::string prompt = "Once upon a time";
    int max_tokens = 32;
    float temperature = 0.0f;

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--model" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (std::string(argv[i]) == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::string(argv[i]) == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--temp" && i + 1 < argc) {
            temperature = std::stof(argv[++i]);
        }
    }

    cuda_init(0);
    GPUInfo gpu = cuda_get_device_info(0);

    if (!model_dir.empty()) {
        printf("========================================\n");
        printf("  INFERNT — CUDA LLM Live Inference\n");
        printf("========================================\n");
        cuda_print_device_info(gpu);

        try {
            InferenceEngine engine;
            printf("\nLoading model from: %s\n", model_dir.c_str());
            engine.load_model(model_dir);
            engine.config().print();

            GenerationConfig gen_cfg;
            gen_cfg.max_new_tokens = max_tokens;
            gen_cfg.temperature = temperature;
            gen_cfg.stream_stdout = true;

            GenerationStats stats;
            printf("\nPrompt: %s\nResponse: ", prompt.c_str());
            engine.generate(prompt, gen_cfg, &stats);
            printf("\n\n=== Generation Performance ===\n");
            printf("  Prompt tokens:   %d\n", stats.prompt_tokens);
            printf("  New tokens:      %d\n", stats.generated_tokens);
            printf("  Prefill latency: %.2f ms\n", stats.prefill_ms);
            printf("  Decode latency:  %.2f ms\n", stats.decode_ms);
            printf("  Throughput:      %.2f tokens/sec\n", stats.tokens_per_second);
            printf("==============================\n");
            return 0;
        } catch (const std::exception& e) {
            printf("\nError during inference: %s\n", e.what());
            return 1;
        }
    }

    printf("========================================\n");
    printf("  INFERNT — Engine Verification Suite\n");
    printf("========================================\n\n");
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

    printf("\n=== Module: Core Tensor Infrastructure ===\n");
    run_test("CPU Tensor", test_tensor_cpu);
    run_test("GPU Tensor", test_tensor_gpu);
    run_test("Move Semantics", test_tensor_move);
    run_test("ModelConfig", test_config);

    printf("\n=== Module: Linear Projections & Compute ===\n");
    run_test("cuBLAS GEMM", test_cublas_gemm);

    printf("\n=== Module: Normalization & Residual Connections ===\n");
    run_test("RMSNorm", test_rmsnorm);
    run_test("Residual Add", test_residual);

    printf("\n=== Module: Positional Embeddings ===\n");
    run_test("RoPE", test_rope);

    printf("\n=== Module: Attention & Softmax ===\n");
    run_test("Softmax", test_softmax);

    printf("\n=== Module: KV Cache Subsystem ===\n");
    run_test("KV Cache Append", test_kv_cache);

    printf("\n=== Module: Grouped-Query Attention (GQA) ===\n");
    run_test("Attention GQA", test_attention_gqa);

    printf("\n=== Module: Feed-Forward Network & Transformer Architecture ===\n");
    run_test("SiLU MLP Activation", test_mlp_activation);
    run_test("Llama Forward Pass", test_llama_architecture);

    printf("\n=== Module: Tokenizer Subsystem ===\n");
    run_test("BPE Tokenizer", test_tokenizer);

    printf("\n=== Module: Model Weight Ingestion ===\n");
    run_test("Checkpoint Loader", test_checkpoint_loader);

    printf("\n=== Module: Sampling Subsystem ===\n");
    run_test("Embedding Lookup & Argmax", test_embedding_and_sampling);

    printf("\n=== Module: Autoregressive Pipeline & Profiling ===\n");
    run_test("Autoregressive Generation", test_autoregressive_generation);

    printf("\n=== Module: Cross-Architecture Generalization ===\n");
    run_test("Architecture Generalization", test_architecture_generalization);

    printf("\n========================================\n");
    printf("  Results: %d passed, %d failed\n", passed, failed);
    printf("========================================\n");

    return failed > 0 ? 1 : 0;
}
