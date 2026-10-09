#include "kv_cache.h"
#include "cuda_utils.h"
#include <stdexcept>

void KVCache::allocate(const ModelConfig& config) {
    max_seq_len = config.max_seq_len;
    current_seq_len = 0;

    k_cache.clear();
    v_cache.clear();

    for (int i = 0; i < config.num_layers; i++) {
        k_cache.push_back(Tensor::zeros(
            {config.num_key_value_heads, max_seq_len, config.head_dim},
            DType::FP32, Device::CUDA));
        v_cache.push_back(Tensor::zeros(
            {config.num_key_value_heads, max_seq_len, config.head_dim},
            DType::FP32, Device::CUDA));
    }
}

void KVCache::reset() {
    current_seq_len = 0;
    for (auto& k : k_cache) {
        CUDA_CHECK(cudaMemset(k.data_ptr(), 0, k.size_bytes()));
    }
    for (auto& v : v_cache) {
        CUDA_CHECK(cudaMemset(v.data_ptr(), 0, v.size_bytes()));
    }
}
