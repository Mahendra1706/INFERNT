#include "engine.h"
#include "checkpoint_loader.h"
#include "kernels.h"
#include "cuda_utils.h"
#include <chrono>
#include <iostream>
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

InferenceEngine::InferenceEngine() : is_loaded_(false) {}

void InferenceEngine::load_model(const std::string& model_dir) {
    std::string config_path = model_dir + "/config.json";
    std::string weights_path = model_dir + "/model.bin";
    std::string tokenizer_path = model_dir + "/tokenizer.bin";
    load_model(CheckpointLoader::load_config(config_path), weights_path, tokenizer_path);
}

void InferenceEngine::load_model(const ModelConfig& config,
                                 const std::string& weights_path,
                                 const std::string& tokenizer_path) {
    config_ = config;
    if (!tokenizer_.load(tokenizer_path)) {
        throw std::runtime_error("Failed to load tokenizer from: " + tokenizer_path);
    }
    ModelWeights loaded_weights;
    CheckpointLoader::load_weights(weights_path, config_, loaded_weights, Device::CUDA);
    initialize(config_, std::move(loaded_weights), std::move(tokenizer_));
}

void InferenceEngine::initialize(const ModelConfig& config,
                                ModelWeights weights,
                                Tokenizer tokenizer) {
    config_ = config;
    weights_ = std::move(weights);
    tokenizer_ = std::move(tokenizer);

    cublas_ = std::make_unique<CublasWrapper>();
    architecture_ = std::make_unique<LlamaArchitecture>(*cublas_);
    architecture_->allocate_activations(config_, 1);

    kv_cache_.allocate(config_);

    input_embedding_ = Tensor::zeros({1, config_.hidden_size}, DType::FP32, Device::CUDA);
    logits_ = Tensor::zeros({1, config_.vocab_size}, DType::FP32, Device::CUDA);
    dev_out_token_ = Tensor::zeros({1}, DType::FP32, Device::CUDA);

    is_loaded_ = true;
}

int InferenceEngine::forward_token(int token_id, int position) {
    if (!is_loaded_) {
        throw std::runtime_error("Engine model is not loaded");
    }

    launch_embedding_lookup(input_embedding_.data_as_float(),
                            weights_.embedding.data_as_float(),
                            token_id, config_.hidden_size);

    architecture_->forward(input_embedding_, weights_, kv_cache_, position, logits_);
    CUDA_CHECK(cudaDeviceSynchronize());

    return sample_next_token(0.0f);
}

int InferenceEngine::sample_next_token(float temperature) {
    if (temperature <= 0.0f) {
        launch_argmax(reinterpret_cast<int*>(dev_out_token_.data_as_float()),
                      logits_.data_as_float(), config_.vocab_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        int sampled_token = 0;
        cudaMemcpy(&sampled_token, dev_out_token_.data_as_float(), sizeof(int), cudaMemcpyDeviceToHost);
        return sampled_token;
    }

    Tensor cpu_logits = logits_.to(Device::CPU);
    const float* logit_data = cpu_logits.data_as_float();

    float max_val = -1e30f;
    for (int i = 0; i < config_.vocab_size; i++) {
        if (logit_data[i] > max_val) max_val = logit_data[i];
    }

    std::vector<float> probs(config_.vocab_size);
    float sum_exp = 0.0f;
    for (int i = 0; i < config_.vocab_size; i++) {
        probs[i] = std::exp((logit_data[i] - max_val) / temperature);
        sum_exp += probs[i];
    }

    static std::mt19937 gen(1337);
    std::uniform_real_distribution<float> dis(0.0f, sum_exp);
    float r = dis(gen);

    float cumulative = 0.0f;
    for (int i = 0; i < config_.vocab_size; i++) {
        cumulative += probs[i];
        if (r <= cumulative) {
            return i;
        }
    }
    return config_.vocab_size - 1;
}

std::string InferenceEngine::generate(const std::string& prompt,
                                      const GenerationConfig& gen_config,
                                      GenerationStats* stats) {
    if (!is_loaded_) {
        throw std::runtime_error("Engine model is not loaded");
    }

    auto start_all = std::chrono::high_resolution_clock::now();

    std::vector<int> prompt_tokens = tokenizer_.encode(prompt, true, false);
    if (prompt_tokens.empty()) {
        prompt_tokens.push_back(tokenizer_.bos_id());
    }

    kv_cache_.reset();

    auto start_prefill = std::chrono::high_resolution_clock::now();
    for (size_t pos = 0; pos < prompt_tokens.size(); pos++) {
        forward_token(prompt_tokens[pos], static_cast<int>(pos));
    }
    auto end_prefill = std::chrono::high_resolution_clock::now();

    int next_token = sample_next_token(gen_config.temperature);

    std::string generated_text;
    auto start_decode = std::chrono::high_resolution_clock::now();

    int generated_count = 0;
    for (int step = 0; step < gen_config.max_new_tokens; step++) {
        if (next_token == tokenizer_.eos_id()) {
            break;
        }

        std::string piece = tokenizer_.decode(next_token);
        generated_text += piece;
        if (gen_config.stream_stdout) {
            std::cout << piece << std::flush;
        }
        generated_count++;

        int current_pos = static_cast<int>(prompt_tokens.size()) + step;
        if (current_pos >= config_.max_seq_len - 1) {
            break;
        }

        forward_token(next_token, current_pos);
        next_token = sample_next_token(gen_config.temperature);
    }
    auto end_decode = std::chrono::high_resolution_clock::now();
    auto end_all = std::chrono::high_resolution_clock::now();

    if (stats) {
        stats->prompt_tokens = static_cast<int>(prompt_tokens.size());
        stats->generated_tokens = generated_count;
        stats->prefill_ms = std::chrono::duration<double, std::milli>(end_prefill - start_prefill).count();
        stats->decode_ms = std::chrono::duration<double, std::milli>(end_decode - start_decode).count();
        stats->total_ms = std::chrono::duration<double, std::milli>(end_all - start_all).count();
        stats->tokens_per_second = (generated_count > 0 && stats->decode_ms > 0) ?
            (generated_count / (stats->decode_ms / 1000.0)) : 0.0;
    }

    return generated_text;
}

const ModelConfig& InferenceEngine::config() const {
    return config_;
}

const Tokenizer& InferenceEngine::tokenizer() const {
    return tokenizer_;
}

bool InferenceEngine::is_loaded() const {
    return is_loaded_;
}
