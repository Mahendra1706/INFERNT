#include "checkpoint_loader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <cstdio>

namespace {
std::string extract_json_string(const std::string& content, const std::string& key) {
    size_t key_pos = content.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return "";
    size_t colon_pos = content.find(':', key_pos);
    if (colon_pos == std::string::npos) return "";
    size_t quote_start = content.find('\"', colon_pos);
    if (quote_start == std::string::npos) return "";
    size_t quote_end = content.find('\"', quote_start + 1);
    if (quote_end == std::string::npos) return "";
    return content.substr(quote_start + 1, quote_end - quote_start - 1);
}

double extract_json_number(const std::string& content, const std::string& key, double default_val = 0.0) {
    size_t key_pos = content.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return default_val;
    size_t colon_pos = content.find(':', key_pos);
    if (colon_pos == std::string::npos) return default_val;
    size_t val_start = content.find_first_of("-0123456789.", colon_pos);
    if (val_start == std::string::npos) return default_val;
    size_t val_end = content.find_first_not_of("-0123456789.eE+", val_start);
    std::string val_str = (val_end == std::string::npos) ?
                          content.substr(val_start) :
                          content.substr(val_start, val_end - val_start);
    try {
        return std::stod(val_str);
    } catch (...) {
        return default_val;
    }
}
}

ModelConfig CheckpointLoader::load_config(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file: " + config_path);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    ModelConfig config;
    config.architecture = extract_json_string(content, "architecture");
    if (config.architecture.empty()) {
        config.architecture = "LlamaForCausalLM";
    }

    config.vocab_size = static_cast<int>(extract_json_number(content, "vocab_size"));
    config.hidden_size = static_cast<int>(extract_json_number(content, "hidden_size"));
    config.num_layers = static_cast<int>(extract_json_number(content, "num_layers"));
    config.num_attention_heads = static_cast<int>(extract_json_number(content, "num_attention_heads"));
    config.num_key_value_heads = static_cast<int>(extract_json_number(content, "num_key_value_heads"));
    config.intermediate_size = static_cast<int>(extract_json_number(content, "intermediate_size"));
    config.max_seq_len = static_cast<int>(extract_json_number(content, "max_seq_len", 2048));
    config.rms_norm_eps = static_cast<float>(extract_json_number(content, "rms_norm_eps", 1e-5));
    config.rope_theta = static_cast<float>(extract_json_number(content, "rope_theta", 10000.0));

    config.derive_computed_fields();
    config.validate();
    return config;
}

void CheckpointLoader::load_weights(const std::string& weights_path,
                                   const ModelConfig& config,
                                   ModelWeights& weights,
                                   Device target_device) {
    std::ifstream file(weights_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open weights file: " + weights_path);
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t tensor_count = 0;

    file.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&tensor_count), sizeof(uint32_t));

    if (magic != 0x494E4654) {
        throw std::runtime_error("Invalid model binary magic number");
    }

    if (version != 1) {
        throw std::runtime_error("Unsupported model binary version");
    }

    weights.layers.resize(config.num_layers);

    for (uint32_t t = 0; t < tensor_count; t++) {
        int32_t name_len = 0;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(int32_t));
        std::string name(name_len, '\0');
        file.read(&name[0], name_len);

        int32_t num_dims = 0;
        file.read(reinterpret_cast<char*>(&num_dims), sizeof(int32_t));
        std::vector<int> shape(num_dims);
        for (int i = 0; i < num_dims; i++) {
            int32_t dim = 0;
            file.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
            shape[i] = dim;
        }

        int64_t num_elements = 0;
        file.read(reinterpret_cast<char*>(&num_elements), sizeof(int64_t));

        Tensor cpu_tensor = Tensor::zeros(shape, DType::FP32, Device::CPU);
        file.read(reinterpret_cast<char*>(cpu_tensor.data_as_float()), num_elements * sizeof(float));

        Tensor dest_tensor = (target_device == Device::CUDA) ? cpu_tensor.to(Device::CUDA) : std::move(cpu_tensor);

        if (name == "model.embed_tokens.weight") {
            weights.embedding = std::move(dest_tensor);
        } else if (name == "model.norm.weight") {
            weights.final_norm = std::move(dest_tensor);
        } else if (name == "lm_head.weight") {
            weights.lm_head = std::move(dest_tensor);
        } else if (name.rfind("model.layers.", 0) == 0) {
            int layer_idx = -1;
            char sub_name[128] = {0};
            if (sscanf(name.c_str(), "model.layers.%d.%s", &layer_idx, sub_name) == 2) {
                if (layer_idx >= 0 && layer_idx < config.num_layers) {
                    std::string s(sub_name);
                    if (s == "input_layernorm.weight") {
                        weights.layers[layer_idx].attn_norm = std::move(dest_tensor);
                    } else if (s == "self_attn.q_proj.weight") {
                        weights.layers[layer_idx].q_proj = std::move(dest_tensor);
                    } else if (s == "self_attn.k_proj.weight") {
                        weights.layers[layer_idx].k_proj = std::move(dest_tensor);
                    } else if (s == "self_attn.v_proj.weight") {
                        weights.layers[layer_idx].v_proj = std::move(dest_tensor);
                    } else if (s == "self_attn.o_proj.weight") {
                        weights.layers[layer_idx].o_proj = std::move(dest_tensor);
                    } else if (s == "post_attention_layernorm.weight") {
                        weights.layers[layer_idx].ffn_norm = std::move(dest_tensor);
                    } else if (s == "mlp.gate_proj.weight") {
                        weights.layers[layer_idx].gate_proj = std::move(dest_tensor);
                    } else if (s == "mlp.up_proj.weight") {
                        weights.layers[layer_idx].up_proj = std::move(dest_tensor);
                    } else if (s == "mlp.down_proj.weight") {
                        weights.layers[layer_idx].down_proj = std::move(dest_tensor);
                    }
                }
            }
        }
    }

    if (!weights.lm_head.is_valid() && weights.embedding.is_valid()) {
        weights.lm_head = weights.embedding.clone();
    }
}
