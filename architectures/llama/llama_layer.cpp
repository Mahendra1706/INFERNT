#include "architecture.h"
#include "kernels.h"
#include <cmath>

void LlamaArchitecture::layer_forward(const Tensor& hidden_states,
                                      const TransformerLayerWeights& layer_weights,
                                      KVCache& kv_cache,
                                      int position,
                                      int layer_idx,
                                      Tensor& next_hidden_states) {
    // 1. Attention RMSNorm
    launch_rmsnorm(norm_out_.data_as_float(), hidden_states.data_as_float(),
                   layer_weights.attn_norm.data_as_float(), max_batch_size_,
                   config_.hidden_size, config_.rms_norm_eps);

    // 2. Q/K/V Projections
    int q_size = config_.num_attention_heads * config_.head_dim;
    int kv_size = config_.num_key_value_heads * config_.head_dim;
    
    cublas_.gemm(norm_out_.data_as_float(), layer_weights.q_proj.data_as_float(), q_.data_as_float(),
                 max_batch_size_, q_size, config_.hidden_size, 1.0f, 0.0f, false, true);
    cublas_.gemm(norm_out_.data_as_float(), layer_weights.k_proj.data_as_float(), k_.data_as_float(),
                 max_batch_size_, kv_size, config_.hidden_size, 1.0f, 0.0f, false, true);
    cublas_.gemm(norm_out_.data_as_float(), layer_weights.v_proj.data_as_float(), v_.data_as_float(),
                 max_batch_size_, kv_size, config_.hidden_size, 1.0f, 0.0f, false, true);

    // 3. RoPE
    launch_rope(q_.data_as_float(), k_.data_as_float(),
                max_batch_size_, config_.num_attention_heads, config_.num_key_value_heads,
                config_.head_dim, position, config_.rope_theta);

    // 4. KV Cache Append
    float* k_cache_layer = kv_cache.k_cache[layer_idx].data_as_float();
    float* v_cache_layer = kv_cache.v_cache[layer_idx].data_as_float();
    
    launch_kv_cache_append(k_cache_layer, v_cache_layer,
                           k_.data_as_float(), v_.data_as_float(),
                           config_.num_key_value_heads, config_.head_dim,
                           config_.max_seq_len, position);

    // 5. Attention
    int seq_len = position + 1; // generating token by token
    float scale = 1.0f / std::sqrt((float)config_.head_dim);
    
    launch_attention_scores(attn_scores_.data_as_float(), q_.data_as_float(), k_cache_layer,
                            config_.num_attention_heads, config_.num_key_value_heads,
                            config_.head_dim, seq_len, config_.max_seq_len, scale);
                            
    launch_softmax(attn_scores_.data_as_float(), attn_scores_.data_as_float(),
                   config_.num_attention_heads, seq_len);
                   
    launch_attention_values(attn_out_.data_as_float(), attn_scores_.data_as_float(), v_cache_layer,
                            config_.num_attention_heads, config_.num_key_value_heads,
                            config_.head_dim, seq_len, config_.max_seq_len);

    // 6. O Projection
    cublas_.gemm(attn_out_.data_as_float(), layer_weights.o_proj.data_as_float(), norm_out_.data_as_float(), // reuse norm_out_ for output
                 max_batch_size_, config_.hidden_size, q_size, 1.0f, 0.0f, false, true);

    // 7. Residual 1 (hidden_states + O_proj -> next_hidden_states)
    launch_residual_add(next_hidden_states.data_as_float(), hidden_states.data_as_float(), norm_out_.data_as_float(),
                        max_batch_size_ * config_.hidden_size);

    // 8. FFN RMSNorm
    launch_rmsnorm(norm_out_.data_as_float(), next_hidden_states.data_as_float(),
                   layer_weights.ffn_norm.data_as_float(), max_batch_size_,
                   config_.hidden_size, config_.rms_norm_eps);

    // 9. MLP Projections & SiLU
    cublas_.gemm(norm_out_.data_as_float(), layer_weights.gate_proj.data_as_float(), gate_.data_as_float(),
                 max_batch_size_, config_.intermediate_size, config_.hidden_size, 1.0f, 0.0f, false, true);
    cublas_.gemm(norm_out_.data_as_float(), layer_weights.up_proj.data_as_float(), up_.data_as_float(),
                 max_batch_size_, config_.intermediate_size, config_.hidden_size, 1.0f, 0.0f, false, true);
                 
    launch_silu_mul(silu_out_.data_as_float(), gate_.data_as_float(), up_.data_as_float(), max_batch_size_ * config_.intermediate_size);
    
    cublas_.gemm(silu_out_.data_as_float(), layer_weights.down_proj.data_as_float(), mlp_out_.data_as_float(),
                 max_batch_size_, config_.hidden_size, config_.intermediate_size, 1.0f, 0.0f, false, true);

    // 10. Residual 2 (next_hidden_states + mlp_out -> next_hidden_states)
    launch_residual_add(next_hidden_states.data_as_float(), next_hidden_states.data_as_float(), mlp_out_.data_as_float(),
                        max_batch_size_ * config_.hidden_size);
}
