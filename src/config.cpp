#include "config.h"
#include <cstdio>
#include <stdexcept>

const char* activation_name(ActivationType act) {
    switch (act) {
        case ActivationType::SILU: return "SiLU";
        case ActivationType::GELU: return "GELU";
        case ActivationType::RELU: return "ReLU";
    }
    return "UNKNOWN";
}

const char* norm_type_name(NormType norm) {
    switch (norm) {
        case NormType::RMSNORM:   return "RMSNorm";
        case NormType::LAYERNORM: return "LayerNorm";
    }
    return "UNKNOWN";
}

const char* position_embedding_name(PositionEmbeddingType pos) {
    switch (pos) {
        case PositionEmbeddingType::ROPE:     return "RoPE";
        case PositionEmbeddingType::ABSOLUTE: return "Absolute";
        case PositionEmbeddingType::ALIBI:    return "ALiBi";
    }
    return "UNKNOWN";
}

void ModelConfig::derive_computed_fields() {
    if (num_attention_heads > 0 && hidden_size > 0 && head_dim == 0) {
        head_dim = hidden_size / num_attention_heads;
    }
}

int ModelConfig::num_kv_groups() const {
    if (num_key_value_heads == 0) return 1;
    return num_attention_heads / num_key_value_heads;
}

void ModelConfig::validate() const {
    if (vocab_size <= 0)
        throw std::runtime_error("Invalid vocab_size");
    if (hidden_size <= 0)
        throw std::runtime_error("Invalid hidden_size");
    if (num_layers <= 0)
        throw std::runtime_error("Invalid num_layers");
    if (num_attention_heads <= 0)
        throw std::runtime_error("Invalid num_attention_heads");
    if (num_key_value_heads <= 0)
        throw std::runtime_error("Invalid num_key_value_heads");
    if (intermediate_size <= 0)
        throw std::runtime_error("Invalid intermediate_size");
    if (max_seq_len <= 0)
        throw std::runtime_error("Invalid max_seq_len");
    if (head_dim <= 0)
        throw std::runtime_error("Invalid head_dim");
    if (num_attention_heads % num_key_value_heads != 0)
        throw std::runtime_error("num_attention_heads must be divisible by num_key_value_heads");
}

void ModelConfig::print() const {
    printf("=== Model Configuration ===\n");
    printf("  Architecture:        %s\n", architecture.c_str());
    printf("  Vocab size:          %d\n", vocab_size);
    printf("  Hidden size:         %d\n", hidden_size);
    printf("  Num layers:          %d\n", num_layers);
    printf("  Attention heads:     %d\n", num_attention_heads);
    printf("  KV heads:            %d\n", num_key_value_heads);
    printf("  KV groups:           %d\n", num_kv_groups());
    printf("  Intermediate size:   %d\n", intermediate_size);
    printf("  Max seq len:         %d\n", max_seq_len);
    printf("  Head dim:            %d\n", head_dim);
    printf("  RMSNorm eps:         %.1e\n", rms_norm_eps);
    printf("  RoPE theta:          %.1f\n", rope_theta);
    printf("  Activation:          %s\n", activation_name(activation));
    printf("  Normalization:       %s\n", norm_type_name(norm_type));
    printf("  Position embedding:  %s\n", position_embedding_name(position_embedding));
    printf("===========================\n");
}
