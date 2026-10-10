#pragma once

#include "tensor.h"
#include "config.h"
#include "weights.h"
#include "kv_cache.h"
#include "cublas_wrapper.h"

class ModelArchitecture {
public:
    virtual ~ModelArchitecture() = default;
    
    virtual void allocate_activations(const ModelConfig& config, int max_batch_size) = 0;
    
    virtual void forward(const Tensor& input_embeddings,
                         const ModelWeights& weights,
                         KVCache& kv_cache,
                         int position,
                         Tensor& logits) = 0;
};

class LlamaArchitecture : public ModelArchitecture {
public:
    LlamaArchitecture(CublasWrapper& cublas);

    void allocate_activations(const ModelConfig& config, int max_batch_size) override;

    void forward(const Tensor& input_embeddings,
                 const ModelWeights& weights,
                 KVCache& kv_cache,
                 int position,
                 Tensor& logits) override;

private:
    void layer_forward(const Tensor& hidden_states,
                       const TransformerLayerWeights& layer_weights,
                       KVCache& kv_cache,
                       int position,
                       int layer_idx,
                       Tensor& next_hidden_states);

    CublasWrapper& cublas_;
    ModelConfig config_;
    int max_batch_size_;

    // Activation buffers
    Tensor hidden_states_;
    Tensor next_hidden_states_;
    Tensor norm_out_;
    
    // Attention buffers
    Tensor q_;
    Tensor k_;
    Tensor v_;
    Tensor attn_scores_;
    Tensor attn_probs_; // Optional if we apply softmax in-place
    Tensor attn_out_;
    
    // MLP buffers
    Tensor gate_;
    Tensor up_;
    Tensor silu_out_;
    Tensor mlp_out_;
};
