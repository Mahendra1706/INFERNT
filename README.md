# INFERNT — CUDA LLM Inference Engine

INFERNT is a general-purpose, high-performance CUDA/C++ LLM inference engine built from scratch. It is engineered to expose the direct mathematical and memory-hierarchy relationships between Transformer equations and physical GPU execution.

TinyLlama (1.1B) serves as the primary reference model to validate correctness, but the engine is fully configuration-driven and architecture-general.

---

## 1. Architectural Philosophy

The engine strictly decouples the conceptual layers:

```
┌────────────────────────────────────────────────────────┐
│                         MODEL                          │
│     Configuration (JSON) + Weights (BIN) + Tokenizer   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                     ARCHITECTURE                       │
│    ModelArchitecture abstraction (LlamaArchitecture)   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                        RUNTIME                         │
│     InferenceEngine: Memory, KV Cache, Scheduling      │
└─────────────┬────────────────────────────┬─────────────┘
              │                            │
              ▼                            ▼
┌───────────────────────────┐┌───────────────────────────┐
│          cuBLAS           ││       CUDA KERNELS        │
│   GEMM Linear Projections ││ RMSNorm, RoPE, Softmax,   │
│   Q, K, V, O, Gate, Up,   ││ KV Cache Append, GQA,     │
│   Down, LM Head           ││ SiLU*Mul, Argmax Sampling │
└─────────────┬─────────────┘└─────────────┬─────────────┘
              │                            │
              └─────────────┬──────────────┘
                            ▼
                    NVIDIA GPU Memory
                            │
                            ▼
                     Next Token ID
```

- **MODEL**: Data payload (hyperparameters, vocabulary, weight tensors).
- **ARCHITECTURE**: Computational graph defining how layers and attention blocks are sequenced.
- **RUNTIME**: Memory management, static activation buffers, KV cache lifecycle, and scheduling.
- **GEMM (cuBLAS)**: High-throughput matrix-multiplication operations.
- **CUDA KERNELS**: Memory-bound elementwise and reduction operations (RMSNorm, RoPE, Softmax, SiLU, Attention reduction, Argmax).

---

## 2. Mathematical Pipeline & GPU Mapping

Each mathematical operation maps directly to a tensor and a GPU execution unit:

| Stage | Operation | Tensor Dimensions | GPU Mechanism |
|---|---|---|---|
| **Embedding Lookup** | $x = \text{Embedding}[t]$ | $[1, \text{hidden\_size}]$ | Custom CUDA Kernel (`launch_embedding_lookup`) |
| **Input RMSNorm** | $x_{\text{norm}} = \frac{x}{\text{RMS}(x)} \odot w_{\text{attn\_norm}}$ | $[1, \text{hidden\_size}]$ | Custom CUDA Kernel (`launch_rmsnorm`) |
| **Q/K/V Projections** | $Q = x_{\text{norm}} W_Q^T, K = \dots, V = \dots$ | $Q: [1, N_q \cdot D], K, V: [1, N_{kv} \cdot D]$ | cuBLAS `cublasSgemm` |
| **Rotary Embedding (RoPE)** | $(Q, K) \leftarrow \text{RoPE}(Q, K, \text{pos})$ | $[1, N_q, D], [1, N_{kv}, D]$ | Custom CUDA Kernel (`launch_rope`) |
| **KV Cache Update** | $\text{Cache}_K[\text{pos}] = K, \text{Cache}_V[\text{pos}] = V$ | $[N_{kv}, \text{max\_seq}, D]$ | Custom CUDA Kernel (`launch_kv_cache_append`) |
| **Attention Scores** | $S = \frac{Q \cdot K_{\text{cache}}^T}{\sqrt{D}}$ | $[N_q, \text{seq\_len}]$ | Custom CUDA Kernel with GQA (`launch_attention_scores`) |
| **Causal Softmax** | $P = \text{softmax}(S)$ | $[N_q, \text{seq\_len}]$ | Custom CUDA Kernel (`launch_softmax`) |
| **Attention Values** | $O_{\text{attn}} = P \cdot V_{\text{cache}}$ | $[1, N_q \cdot D]$ | Custom CUDA Kernel with GQA (`launch_attention_values`) |
| **O Projection** | $x_{\text{proj}} = O_{\text{attn}} W_O^T$ | $[1, \text{hidden\_size}]$ | cuBLAS `cublasSgemm` |
| **Residual 1** | $x = x + x_{\text{proj}}$ | $[1, \text{hidden\_size}]$ | Custom CUDA Kernel (`launch_residual_add`) |
| **Post-Attn RMSNorm** | $x_{\text{norm}} = \text{RMSNorm}(x, w_{\text{ffn\_norm}})$ | $[1, \text{hidden\_size}]$ | Custom CUDA Kernel (`launch_rmsnorm`) |
| **Gate / Up Projections** | $\text{gate} = x_{\text{norm}} W_{\text{gate}}^T, \text{up} = x_{\text{norm}} W_{\text{up}}^T$ | $[1, \text{intermediate\_size}]$ | cuBLAS `cublasSgemm` |
| **SwiGLU Activation** | $a = \text{SiLU}(\text{gate}) \odot \text{up}$ | $[1, \text{intermediate\_size}]$ | Custom CUDA Kernel (`launch_silu_mul`) |
| **Down Projection** | $x_{\text{mlp}} = a W_{\text{down}}^T$ | $[1, \text{hidden\_size}]$ | cuBLAS `cublasSgemm` |
| **Residual 2** | $x = x + x_{\text{mlp}}$ | $[1, \text{hidden\_size}]$ | Custom CUDA Kernel (`launch_residual_add`) |
| **Final RMSNorm** | $x_{\text{final}} = \text{RMSNorm}(x, w_{\text{norm}})$ | $[1, \text{hidden\_size}]$ | Custom CUDA Kernel (`launch_rmsnorm`) |
| **LM Head** | $\text{logits} = x_{\text{final}} W_{\text{head}}^T$ | $[1, \text{vocab\_size}]$ | cuBLAS `cublasSgemm` |
| **Token Sampling** | $t_{\text{next}} = \arg\max(\text{logits})$ | Scaled integer ID | Custom CUDA Reduction Kernel (`launch_argmax`) |

---

## 3. Grouped-Query Attention (GQA) & KV Cache

TinyLlama uses 32 Query heads ($N_q = 32$) and 4 Key-Value heads ($N_{kv} = 4$).
The number of query heads per KV group is:
$$\text{num\_kv\_groups} = \frac{N_q}{N_{kv}} = \frac{32}{4} = 8$$

Each KV head is shared across 8 Query heads:
- Query heads $0 \dots 7$ attend to KV head $0$
- Query heads $8 \dots 15$ attend to KV head $1$
- Query heads $16 \dots 23$ attend to KV head $2$
- Query heads $24 \dots 31$ attend to KV head $3$

The KV cache is allocated per layer as a contiguous tensor of shape $[N_{kv}, \text{max\_seq\_len}, \text{head\_dim}]$, avoiding redundant memory allocations during decode.

---

## 4. Checkpoint Conversion Pipeline

Hugging Face models distribute weights in PyTorch / Safetensors formats. INFERNT provides a clean Python conversion tool (`tools/convert_checkpoint.py`) that exports the model into an engine-native flat binary format.

### Exported Artifacts:
1. `config.json`: Model hyperparameters (vocab size, hidden dim, layer counts, eps, theta).
2. `tokenizer.bin`: SentencePiece BPE vocabulary pieces and scores.
3. `model.bin`: Flat binary weight file:
   - Header: Magic `0x494E4654` ('INFT'), version `1`, tensor count.
   - Per-tensor metadata: name string, shape dimensions, element count, and raw FP32 bytes.

---

## 5. Repository Structure

```
INFERNT/
├── CMakeLists.txt              Build configuration (C++17 + CUDA)
├── README.md                   Technical documentation
├── build_and_run.sh            Colab & Linux clean build and execution script
├── infernt_colab.ipynb         Interactive Google Colab notebook
├── master_prompt.md            Engineering specification
│
├── include/                    Header files
│   ├── architecture.h          ModelArchitecture & LlamaArchitecture base classes
│   ├── checkpoint_loader.h     Configuration and binary weight loader
│   ├── config.h                ModelConfig struct and validation
│   ├── cublas_wrapper.h        cuBLAS GEMM wrapper
│   ├── cuda_utils.h            GPU initialization and CUDA_CHECK macros
│   ├── engine.h                InferenceEngine (prefill, decode loop, timing)
│   ├── kernels.h               CUDA kernel declarations
│   ├── kv_cache.h              KVCache management
│   ├── tensor.h                Tensor memory abstraction
│   ├── tokenizer.h             BPE Tokenizer interface
│   └── weights.h               ModelWeights & TransformerLayerWeights structs
│
├── src/                        Runtime implementations
│   ├── config.cpp              Config verification and derived parameters
│   ├── cuda_init.cu            GPU device query and setup
│   ├── engine.cpp              Autoregressive decode loop and latency statistics
│   ├── kv_cache.cpp            KV Cache buffer allocation
│   ├── main.cpp                Full test suite (Phases 1-18) and CLI driver
│   └── tensor.cpp              CPU/GPU tensor allocations and transfers
│
├── architectures/              Model architecture definitions
│   └── llama/
│       ├── llama_layer.cpp     Llama decoder layer forward pass
│       └── llama_model.cpp     Multi-layer sequencing and LM head execution
│
├── cuda/                       Custom CUDA kernels
│   ├── activation.cu           SiLU elementwise multiplication kernel
│   ├── attention.cu            Attention score & value kernels with GQA mapping
│   ├── embedding.cu            Token embedding lookup kernel
│   ├── kv_cache.cu             KV Cache position append kernel
│   ├── residual.cu             Elementwise residual addition kernel
│   ├── rmsnorm.cu              RMSNorm kernel with shared memory reduction
│   ├── rope.cu                 Rotary position embeddings kernel
│   ├── sampling.cu             Parallel argmax token sampling kernel
│   └── softmax.cu              Softmax kernel with numerical stabilization
│
├── gemm/
│   └── cublas_wrapper.cu       cuBLAS GEMM implementation
│
├── loader/
│   └── checkpoint_loader.cpp   Native binary parser and GPU weight loader
│
├── tokenizer/
│   └── tokenizer.cpp           Self-contained SentencePiece BPE tokenizer
│
└── tools/
    └── convert_checkpoint.py   Hugging Face to INFERNT binary converter
```

---

## 6. Build and Run Instructions

### Prerequisites
- Linux OS (Ubuntu 20.04+ recommended)
- NVIDIA GPU with compute capability 7.0+ (T4, V100, A100, RTX 30/40 series)
- CUDA Toolkit 11.8+ or 12.0+
- CMake 3.18+
- Python 3.8+ with PyTorch & Transformers (for checkpoint conversion)

### 1. Build and Run Full Test Suite (Phases 1–18)

```bash
chmod +x build_and_run.sh
./build_and_run.sh
```

This compiles the engine and runs the 14-point validation test suite covering:
1. CPU Tensor memory and indexing
2. GPU Tensor device transfers
3. Move semantics and RAII
4. ModelConfig validation and GQA group calculation
5. cuBLAS GEMM computation against CPU reference
6. RMSNorm shared memory reduction
7. Residual addition kernel
8. RoPE complex frequency transformation
9. Numerically stable Softmax
10. KV Cache append kernel
11. Grouped-Query Attention (GQA) mapping
12. SiLU MLP activation kernel
13. LlamaArchitecture multi-layer forward pass
14. BPE Tokenizer encoding and decoding
15. CheckpointLoader binary parsing
16. Embedding lookup and parallel Argmax
17. Full Autoregressive generation loop and latency profiling
18. Architecture Generalization across non-TinyLlama dimensions

### 2. Convert Pretrained TinyLlama Checkpoint

```bash
pip install transformers torch accelerate sentencepiece
python3 tools/convert_checkpoint.py \
    --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --output_dir ./model_checkpoint
```

### 3. Run Live Inference

```bash
./build/bin/infernt \
    --model ./model_checkpoint \
    --prompt "Once upon a time," \
    --max-tokens 64 \
    --temp 0.0
```

---

## 7. Google Colab Instructions

The easiest way to run INFERNT on an NVIDIA GPU is using the included `infernt_colab.ipynb` notebook:

1. Open [Google Colab](https://colab.research.google.com).
2. Upload `infernt_colab.ipynb`.
3. Select **Runtime → Change runtime type → T4 GPU**.
4. Run all cells:
   - Cell 1: Verifies GPU and `nvcc`.
   - Cell 2: Clones or sets up the workspace.
   - Cell 3: Builds the engine and runs the full Phase 1-18 test suite.
   - Cell 4: Converts TinyLlama 1.1B from Hugging Face into `./model_checkpoint`.
   - Cell 5: Executes live text generation with real-time throughput metrics.

---

## 8. Development Phases Checklist

- [x] **Phase 1**: Project foundation, Tensor abstraction, ModelConfig validation
- [x] **Phase 2**: cuBLAS wrapper and GEMM validation
- [x] **Phase 3**: Memory management & GPU buffer pools
- [x] **Phase 4**: RMSNorm kernel & Residual Addition kernel
- [x] **Phase 5**: Rotary Position Embeddings (RoPE) kernel
- [x] **Phase 6**: Numerically stable Softmax kernel
- [x] **Phase 7**: Contiguous KV Cache allocation and append kernel
- [x] **Phase 8**: Attention computation with GQA head-mapping
- [x] **Phase 9**: SwiGLU / SiLU MLP activation kernel
- [x] **Phase 10**: Single Transformer layer integration
- [x] **Phase 11**: Multi-layer sequencing driven by `config.num_layers`
- [x] **Phase 12**: Final RMSNorm and LM Head projection
- [x] **Phase 13**: Tokenizer integration (SentencePiece BPE in C++)
- [x] **Phase 14**: Checkpoint conversion tool & binary weight loader
- [x] **Phase 15**: Token embedding lookup & parallel argmax sampling
- [x] **Phase 16**: Autoregressive decode loop
- [x] **Phase 17**: Latency profiling (prefill ms, decode ms, tokens/sec)
- [x] **Phase 18**: Architecture generalization validation

---

## 9. Current Limitations & Roadmap

- **Precision**: Full FP32 precision baseline. (Future: FP16 / BF16 mixed-precision).
- **Batch Size**: Batch size = 1 optimized for interactive single-prompt decoding. (Future: Continuous batching).
- **Kernel Fusion**: Kernels execute as discrete CUDA launches. (Future: FlashAttention-2 / fused RMSNorm-GEMM).
