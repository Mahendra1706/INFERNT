#pragma once

#include "tensor.h"
#include "config.h"
#include <vector>
#include <memory>

struct TransformerLayerWeights {
    Tensor attn_norm;
    Tensor q_proj;
    Tensor k_proj;
    Tensor v_proj;
    Tensor o_proj;
    Tensor ffn_norm;
    Tensor gate_proj;
    Tensor up_proj;
    Tensor down_proj;
};

struct ModelWeights {
    Tensor embedding;
    std::vector<TransformerLayerWeights> layers;
    Tensor final_norm;
    Tensor lm_head;
};
