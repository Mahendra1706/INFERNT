#pragma once

#include "config.h"
#include "weights.h"
#include "architecture.h"
#include "kv_cache.h"
#include "tokenizer.h"
#include "cublas_wrapper.h"
#include <memory>
#include <string>
#include <vector>

struct GenerationConfig {
    int max_new_tokens = 64;
    float temperature = 0.0f;
    int top_k = 1;
    bool stream_stdout = true;
};

struct GenerationStats {
    int prompt_tokens = 0;
    int generated_tokens = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double total_ms = 0.0;
    double tokens_per_second = 0.0;
};

class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine() = default;

    void load_model(const std::string& model_dir);
    void load_model(const ModelConfig& config,
                    const std::string& weights_path,
                    const std::string& tokenizer_path);

    void initialize(const ModelConfig& config,
                    ModelWeights weights,
                    Tokenizer tokenizer);

    std::string generate(const std::string& prompt,
                         const GenerationConfig& gen_config = GenerationConfig(),
                         GenerationStats* stats = nullptr);

    int forward_token(int token_id, int position);
    int sample_next_token(float temperature = 0.0f);

    const ModelConfig& config() const;
    const Tokenizer& tokenizer() const;
    bool is_loaded() const;

private:
    ModelConfig config_;
    ModelWeights weights_;
    Tokenizer tokenizer_;
    std::unique_ptr<CublasWrapper> cublas_;
    std::unique_ptr<ModelArchitecture> architecture_;
    KVCache kv_cache_;

    Tensor input_embedding_;
    Tensor logits_;
    Tensor dev_out_token_;
    bool is_loaded_;
};
