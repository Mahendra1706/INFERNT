#pragma once

#include <string>

enum class ActivationType {
    SILU,
    GELU,
    RELU
};

enum class NormType {
    RMSNORM,
    LAYERNORM
};

enum class PositionEmbeddingType {
    ROPE,
    ABSOLUTE,
    ALIBI
};

struct ModelConfig {
    std::string architecture;

    int vocab_size = 0;
    int hidden_size = 0;
    int num_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int intermediate_size = 0;
    int max_seq_len = 0;
    int head_dim = 0;

    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;

    ActivationType activation = ActivationType::SILU;
    NormType norm_type = NormType::RMSNORM;
    PositionEmbeddingType position_embedding = PositionEmbeddingType::ROPE;

    void derive_computed_fields();
    void validate() const;
    void print() const;

    int num_kv_groups() const;
};

const char* activation_name(ActivationType act);
const char* norm_type_name(NormType norm);
const char* position_embedding_name(PositionEmbeddingType pos);
