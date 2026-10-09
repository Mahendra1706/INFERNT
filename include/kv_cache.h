#pragma once

#include "tensor.h"
#include "config.h"
#include <vector>

struct KVCache {
    std::vector<Tensor> k_cache;
    std::vector<Tensor> v_cache;
    int current_seq_len;
    int max_seq_len;

    void allocate(const ModelConfig& config);
    void reset();
};
