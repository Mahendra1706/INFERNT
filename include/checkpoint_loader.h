#pragma once

#include "config.h"
#include "weights.h"
#include <string>

class CheckpointLoader {
public:
    static ModelConfig load_config(const std::string& config_path);
    static void load_weights(const std::string& weights_path,
                             const ModelConfig& config,
                             ModelWeights& weights,
                             Device target_device = Device::CUDA);
};
