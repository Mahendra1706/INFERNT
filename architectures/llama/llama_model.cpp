#include "architecture.h"
#include "kernels.h"
#include <utility>

LlamaArchitecture::LlamaArchitecture(CublasWrapper& cublas) : cublas_(cublas), max_batch_size_(1) {}

void LlamaArchitecture::allocate_activations(const ModelConfig& config, int max_batch_size) {
    config_ = config;
    max_batch_size_ = max_batch_size;

    int hidden_size = config.hidden_size;
    int inter_size = config.intermediate_size;
    
    hidden_states_ = Tensor::zeros({max_batch_size, hidden_size}, DType::FP32, Device::CUDA);
    next_hidden_states_ = Tensor::zeros({max_batch_size, hidden_size}, DType::FP32, Device::CUDA);
    norm_out_ = Tensor::zeros({max_batch_size, hidden_size}, DType::FP32, Device::CUDA);

    int q_size = config.num_attention_heads * config.head_dim;
    int kv_size = config.num_key_value_heads * config.head_dim;

    q_ = Tensor::zeros({max_batch_size, q_size}, DType::FP32, Device::CUDA);
    k_ = Tensor::zeros({max_batch_size, kv_size}, DType::FP32, Device::CUDA);
    v_ = Tensor::zeros({max_batch_size, kv_size}, DType::FP32, Device::CUDA);

    attn_scores_ = Tensor::zeros({max_batch_size * config.num_attention_heads, config.max_seq_len}, DType::FP32, Device::CUDA);
    attn_out_ = Tensor::zeros({max_batch_size, q_size}, DType::FP32, Device::CUDA);

    gate_ = Tensor::zeros({max_batch_size, inter_size}, DType::FP32, Device::CUDA);
    up_ = Tensor::zeros({max_batch_size, inter_size}, DType::FP32, Device::CUDA);
    silu_out_ = Tensor::zeros({max_batch_size, inter_size}, DType::FP32, Device::CUDA);
    mlp_out_ = Tensor::zeros({max_batch_size, hidden_size}, DType::FP32, Device::CUDA);
}

void LlamaArchitecture::forward(const Tensor& input_embeddings,
                                const ModelWeights& weights,
                                KVCache& kv_cache,
                                int position,
                                Tensor& logits) {
    hidden_states_.copy_from(input_embeddings);

    for (int i = 0; i < config_.num_layers; i++) {
        layer_forward(hidden_states_, weights.layers[i], kv_cache, position, i, next_hidden_states_);
        // Swap hidden_states and next_hidden_states (just copy for simplicity or swap data pointers)
        hidden_states_.copy_from(next_hidden_states_);
    }

    launch_rmsnorm(norm_out_.data_as_float(), hidden_states_.data_as_float(),
                   weights.final_norm.data_as_float(), max_batch_size_,
                   config_.hidden_size, config_.rms_norm_eps);

    cublas_.gemm(norm_out_.data_as_float(), weights.lm_head.data_as_float(),
                 logits.data_as_float(),
                 max_batch_size_, config_.vocab_size, config_.hidden_size,
                 1.0f, 0.0f, false, true);
}
