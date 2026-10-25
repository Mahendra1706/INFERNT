# INFERNT — High-Performance CUDA/C++ LLM Inference Engine

INFERNT is a bare-metal, high-performance CUDA/C++ inference runtime for Transformer-based Large Language Models (LLMs) built from scratch. It directly maps Transformer mathematical formulations to physical NVIDIA GPU hardware hierarchies without third-party deep learning framework overhead (no PyTorch, ONNX Runtime, or TensorRT dependencies during inference).

While **TinyLlama-1.1B** serves as the primary reference model for end-to-end linguistic verification, the runtime is completely configuration-driven, supporting arbitrary hidden dimensions, attention head topologies, Grouped-Query Attention (GQA) group ratios, and context lengths.

---

## Architecture & System Design

The runtime enforces strict decoupling across the inference stack:

```
┌────────────────────────────────────────────────────────┐
│                         MODEL                          │
│     Configuration (JSON) + Weights (BIN) + Tokenizer   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                     ARCHITECTURE                       │
│    ModelArchitecture Abstraction (LlamaArchitecture)   │
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
│       cuBLAS GEMM         ││    CUSTOM CUDA KERNELS    │
│  Compute-Bound Projections││  Memory-Bound Reductions  │
│  Q, K, V, O, Gate, Up,    ││  RMSNorm, RoPE, Softmax,  │
│  Down, LM Head            ││  KV Append, GQA, Argmax   │
└─────────────┬─────────────┘└─────────────┬─────────────┘
              │                            │
              └─────────────┬──────────────┘
                            ▼
                    NVIDIA GPU Memory
                            │
                            ▼
                   Next Token Emission
```

### Core Engineering Principles

1. **Static Buffer Pre-Allocation (Zero-Allocation Decode Loop)**:
   All intermediate activation buffers, KV caches, and projection tensors are allocated once during engine initialization. The autoregressive decode loop executes with **zero dynamic memory allocations (`cudaMalloc`)**, eliminating memory fragmentation, page locks, and GPU synchronization overhead during generation.

2. **Compute-Bound vs. Memory-Bound Specialization**:
   - **Compute-Bound Operations**: Large matrix multiplications ($Q, K, V, O, \text{Gate}, \text{Up}, \text{Down}, \text{LM Head}$) are dispatched to optimized cuBLAS FP32 GEMMs (`cublasSgemm`) to achieve peak FLOPS.
   - **Memory-Bandwidth Bound Operations**: Elementwise activations and reductions (RMSNorm, RoPE, Softmax, SwiGLU, Residual additions, Argmax sampling) are executed via custom hand-written CUDA kernels with coalesced memory accesses and warp/block-level shared memory reductions.

3. **Numerically Stable Parallel Softmax**:
   Replaces naive atomic reductions with power-of-2 shared-memory parallel reduction trees. Operates reliably on arbitrary floating-point values—including negative attention dot products—preventing catastrophic cancellation, exponential overflow, and NaN generation.

4. **Contiguous Grouped-Query Attention (GQA)**:
   Key and Value caches are allocated in contiguous layout `[N_kv, max_seq_len, head_dim]` per layer. Query heads dynamically index into shared KV groups via modular arithmetic ($\text{kv\_group} = q\_head / (N_q / N_{kv})$), maximizing L1/L2 cache hit rates.

5. **Split-Half Rotary Position Embeddings (RoPE)**:
   Rotary embeddings strictly follow Hugging Face / LLaMA split-half pairing ($d$ and $d + \frac{\text{head\_dim}}{2}$), enabling direct compatibility with pretrained weights converted from the open-source ecosystem.

6. **Self-Contained C++ Tokenizer Runtime**:
   Features a native, self-contained C++ SentencePiece Byte-Pair Encoding (BPE) implementation with byte fallback support. Live inference requires zero Python runtime libraries.

---

## Mathematical Pipeline & GPU Mapping

Each mathematical operation maps directly to a tensor and an execution unit:

| Stage | Mathematical Formulation | Tensor Dimensions | GPU Execution Mechanism |
|---|---|---|---|
| **Embedding Lookup** | $x = \text{Embedding}[t]$ | $[1, \text{hidden\_size}]$ | Custom CUDA Kernel (`launch_embedding_lookup`) |
| **Input RMSNorm** | $x_{\text{norm}} = \frac{x}{\text{RMS}(x)} \odot w_{\text{attn\_norm}}$ | $[1, \text{hidden\_size}]$ | Custom CUDA Kernel (`launch_rmsnorm`) |
| **Q/K/V Projections** | $Q = x_{\text{norm}} W_Q^T, K = \dots, V = \dots$ | $Q: [1, N_q \cdot D], K, V: [1, N_{kv} \cdot D]$ | cuBLAS `cublasSgemm` |
| **Rotary Embedding (RoPE)** | $(Q, K) \leftarrow \text{RoPE}(Q, K, \text{pos})$ | $[1, N_q, D], [1, N_{kv}, D]$ | Custom CUDA Kernel (`launch_rope`) |
| **KV Cache Update** | $\text{Cache}_K[\text{pos}] = K, \text{Cache}_V[\text{pos}] = V$ | $[N_{kv}, \text{max\_seq}, D]$ | Custom CUDA Kernel (`launch_kv_cache_append`) |
| **Attention Scores** | $S = \frac{Q \cdot K_{\text{cache}}^T}{\sqrt{D}}$ | $[N_q, \text{seq\_len}]$ | Custom CUDA Kernel with GQA (`launch_attention_scores`) |
| **Causal Softmax** | $P = \text{softmax}(S)$ | $[N_q, \text{seq\_len}]$ | Parallel Tree Reduction (`launch_softmax`) |
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
| **Token Sampling** | $t_{\text{next}} = \arg\max(\text{logits})$ or Temperature | Scaled Integer ID | Custom CUDA Reduction / CPU Stochastics |

---

## Workflow & Execution Options

INFERNT supports multiple flexible deployment workflows depending on your environment.

### Prerequisites
* **Operating System**: Linux (Ubuntu 20.04+ recommended)
* **GPU**: NVIDIA GPU with Compute Capability $\ge 7.0$ (T4, V100, A100, H100, RTX 20/30/40 series)
* **Toolchain**: CUDA Toolkit 11.8+ or 12.x, CMake 3.18+, GCC/G++ 9+
* **Model Exporter**: Python 3.8+ with PyTorch & Transformers (only needed once to export model weights)

---

### Option A: Standard Modular Build (Direct CMake Workflow)

This is the recommended workflow for development, profiling, and benchmarking.

#### 1. Compile the Binary
```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

#### 2. Run Comprehensive Verification Suite
Running the compiled binary without parameters executes the engine's built-in mathematical test suite:
```bash
./bin/infernt
```

#### 3. Convert Checkpoint
Export any compatible LLaMA architecture checkpoint (e.g. TinyLlama-1.1B) into INFERNT's flat binary format:
```bash
pip install -q transformers torch accelerate sentencepiece protobuf
python3 ../tools/convert_checkpoint.py \
    --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --output_dir ../model_checkpoint
```

#### 4. Run Live Model Inference
```bash
./bin/infernt \
    --model ../model_checkpoint \
    --prompt "Once upon a time in a high-performance CUDA world," \
    --max-tokens 64 \
    --temp 0.0
```

CLI Parameters:
* `--model <dir>`: Path to converted directory containing `config.json`, `tokenizer.bin`, and `model.bin`.
* `--prompt "<text>"`: Input text prompt.
* `--max-tokens <int>`: Maximum tokens to generate (default: 32).
* `--temp <float>`: Sampling temperature (0.0 for greedy argmax, >0.0 for stochastic sampling).

---

### Option B: Automated Build & Execution Script (`build_and_run.sh`)

For convenience, `build_and_run.sh` wraps environment inspection, dependency checks, incremental building, and argument dispatch into a single script:

```bash
chmod +x build_and_run.sh

# Run the test suite:
./build_and_run.sh

# Or pass inference arguments directly:
./build_and_run.sh --model ./model_checkpoint --prompt "Explain CUDA shared memory in one sentence:" --max-tokens 48 --temp 0.0
```

---

### Option C: Cloud Execution (Google Colab / Headless GPU Server)

If you do not have a local NVIDIA GPU, INFERNT runs seamlessly on free Google Colab T4 GPU instances via `infernt_colab.ipynb` or via terminal:

1. Open [Google Colab](https://colab.research.google.com) and set the runtime to **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`).
2. Run the shell cells to clone, build, convert, and execute:
```bash
!git clone https://github.com/Mahendra1706/INFERNT.git /content/INFERNT
%cd /content/INFERNT

# Build
!mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# Convert
!pip install -q transformers torch accelerate sentencepiece protobuf
!python3 tools/convert_checkpoint.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --output_dir /content/model_checkpoint

# Infer
!/content/INFERNT/build/bin/infernt \
    --model /content/model_checkpoint \
    --prompt "Once upon a time," \
    --max-tokens 64 \
    --temp 0.0
```

---

## Checkpoint Ingestion Pipeline

Hugging Face distributes models across fragmented PyTorch/Safetensors checkpoints. INFERNT standardizes checkpoint loading via `tools/convert_checkpoint.py`:

```
Hugging Face Repo ───► convert_checkpoint.py ───► [model_checkpoint/]
                                                     ├── config.json
                                                     ├── tokenizer.bin
                                                     └── model.bin
```

### Binary Layout Specification (`model.bin`)
* **Header** (12 bytes):
  - Magic Number: `0x494E4654` (`"INFT"`)
  - Format Version: `uint32_t` (`1`)
  - Tensor Count: `uint32_t`
* **Tensor Descriptors**:
  - Name Length (`int32_t`) + UTF-8 Name String
  - Dimension Count (`int32_t`) + Dimension Array (`int32_t[]`)
  - Total Elements (`int64_t`)
  - Contiguous IEEE 754 FP32 payload bytes (`elements * sizeof(float)`)

---

## Engine Verification Suite

The repository includes a comprehensive 15-module mathematical and architectural test harness. It verifies each component against analytical CPU ground truth prior to deployment:

```
=== Module: Core Tensor Infrastructure ===
  CPU Tensor: Memory strides, layout, and allocation
  GPU Tensor: Host-to-Device and Device-to-Host transfers
  Move Semantics: RAII lifecycle and zero-copy transfers
  ModelConfig: Hyperparameter derivation and validation

=== Module: Linear Projections & Compute ===
  cuBLAS GEMM: GPU matrix multiplication vs. triple-nested CPU loop (tol <= 1e-4)

=== Module: Normalization & Residual Connections ===
  RMSNorm: Shared-memory RMS normalization vs. analytical CPU reference
  Residual Add: Fused vector addition kernel

=== Module: Positional Embeddings ===
  RoPE: Split-half complex rotary frequency projection

=== Module: Attention & Softmax ===
  Softmax: Numerically stabilized parallel reduction over negative & positive logits

=== Module: KV Cache Subsystem ===
  KV Cache Append: Strided ring-buffer updates across layers and heads

=== Module: Grouped-Query Attention (GQA) ===
  GQA Attention: Multi-head Query to shared Key/Value projection and reduction

=== Module: Feed-Forward Network & Transformer Architecture ===
  SiLU MLP Activation: SwiGLU non-linear gating kernel
  Llama Forward Pass: Multi-layer Transformer decoder block integration

=== Module: Tokenizer Subsystem ===
  BPE Tokenizer: Vocabulary lookup, byte fallback, and BPE merge sequencing

=== Module: Model Weight Ingestion ===
  Checkpoint Loader: Binary header validation and tensor parsing

=== Module: Sampling Subsystem ===
  Embedding Lookup & Argmax: Row selection and parallel reduction token sampling

=== Module: Autoregressive Pipeline & Profiling ===
  Autoregressive Generation: Multi-step decode loop with latency profiler

=== Module: Cross-Architecture Generalization ===
  Architecture Generalization: Arbitrary head dimensions, KV ratios, and context limits
```

---

## Repository Structure

```
INFERNT/
├── CMakeLists.txt              CMake build configuration (C++17 + CUDA)
├── README.md                   Technical documentation & architecture spec
├── build_and_run.sh            Automated build and execution script
├── infernt_colab.ipynb         Interactive Google Colab notebook
│
├── include/                    Public header interfaces
│   ├── architecture.h          ModelArchitecture & LlamaArchitecture interfaces
│   ├── checkpoint_loader.h     Binary parser & tensor loader
│   ├── config.h                ModelConfig struct & parameter validation
│   ├── cublas_wrapper.h        High-performance cuBLAS GEMM wrapper
│   ├── cuda_utils.h            GPU initialization, device queries, CUDA_CHECK
│   ├── engine.h                InferenceEngine (prefill, decode loop, profiler)
│   ├── kernels.h               Custom CUDA kernel declarations
│   ├── kv_cache.h              KVCache allocation and memory layout
│   ├── tensor.h                Tensor memory abstraction & device transfers
│   ├── tokenizer.h             SentencePiece BPE tokenizer interface
│   └── weights.h               ModelWeights & layer parameter structs
│
├── src/                        Core runtime implementation
│   ├── config.cpp              Config parsing and derived parameter calculation
│   ├── cuda_init.cu            GPU device query and environment initialization
│   ├── engine.cpp              Prefill, autoregressive decode loop, performance metrics
│   ├── kv_cache.cpp            Per-layer contiguous KV buffer allocation
│   ├── main.cpp                CLI argument parser and engine verification suite
│   └── tensor.cpp              Host/device memory allocation and copy semantics
│
├── architectures/              Transformer architectures
│   └── llama/
│       ├── llama_layer.cpp     Decoder layer execution (Attention + SwiGLU MLP)
│       └── llama_model.cpp     Multi-layer sequencing and LM head execution
│
├── cuda/                       Custom CUDA kernel implementations
│   ├── activation.cu           SwiGLU (SiLU elementwise product) kernel
│   ├── attention.cu            GQA attention score & value kernels
│   ├── embedding.cu            Embedding table lookup kernel
│   ├── kv_cache.cu             KV cache position append kernel
│   ├── residual.cu             Elementwise residual addition kernel
│   ├── rmsnorm.cu              RMSNorm with shared-memory reduction
│   ├── rope.cu                 Hugging Face split-half RoPE kernel
│   ├── sampling.cu             Parallel argmax reduction sampling kernel
│   └── softmax.cu              Numerically stable power-of-2 tree reduction softmax
│
├── gemm/
│   └── cublas_wrapper.cu       cuBLAS handle management and GEMM execution
│
├── loader/
│   └── checkpoint_loader.cpp   Streaming binary file parser & GPU buffer ingestion
│
├── tokenizer/
│   └── tokenizer.cpp           Native C++ SentencePiece BPE tokenizer
│
└── tools/
    └── convert_checkpoint.py   Hugging Face to INFERNT binary converter
```

---

## Technical Performance & Metrics

When running inference, `InferenceEngine` measures and outputs low-level profiling telemetry:

```
========================================
  INFERNT — CUDA LLM Live Inference
========================================
GPU Device 0: Tesla T4
Compute Capability: 7.5
Total Global Memory: 15102 MB

Loading model from: ./model_checkpoint
[Config] Architecture: LlamaForCausalLM, Hidden: 2048, Layers: 22, Heads: 32 (KV: 4), Intermediate: 5632

Prompt: Once upon a time,
Response:  there was a little girl who lived in a village near the forest.

=== Generation Performance ===
  Prompt tokens:   7
  New tokens:      64
  Prefill latency: 158.70 ms
  Decode latency:  1205.26 ms
  Throughput:      53.10 tokens/sec
==============================
```

### Comparative Benchmark: INFERNT vs. llama.cpp

The table below compares **INFERNT** against the battle-tested **llama.cpp** engine, evaluated on the **identical hardware and model target**:
* **Hardware**: NVIDIA Tesla T4 GPU (16 GB GDDR6, Turing Architecture, sm_75)
* **Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (22 layers, hidden dimension 2048, 32 Q heads, 4 KV groups)

| Metric | INFERNT (Custom CUDA/C++ from Scratch) | llama.cpp (Optimized C++ Reference Engine) | Architecture Delta & Notes |
|---|---|---|---|
| **Prefill Speed** | ~44.1 tokens/sec | **3,088.72 ± 78.11 tokens/sec** | Batched Prompt GEMM ($M = N_{\text{prompt}}$) vs. unbatched sequential prefill |
| **Decode Speed** | 53.10 tokens/sec | **138.17 ± 0.73 tokens/sec** | Pure FP32 custom CUDA kernels vs. Tensor Core accelerated FP16/GEMV |
| **End-to-End Speed** | ~46.9 tokens/sec | **135.13 ± 0.75 tokens/sec** | Overall generation rate including prompt ingestion and decode |
| **Time to First Token (TTFT)** | 158.70 ms (7 tokens) | **24.74 ± 0.40 ms** | Reflects prompt evaluation phase latency |
| **Per-Token Latency (Mean)** | 18.83 ms | **7.24 ms** | Single-token forward pass time in autoregressive decode loop |

#### Performance & Architectural Analysis

1. **Autoregressive Decode Throughput (53.10 vs. 138.17 tok/s)**:
   * **INFERNT** executes an unquantized FP32 pipeline with custom coalesced CUDA kernels (RMSNorm, RoPE, Causal Softmax, GQA, SwiGLU) and cuBLAS `cublasSgemm` projections. With **zero dynamic runtime memory allocations** (`cudaMalloc`) in the generation loop, INFERNT achieves a per-token latency of **18.83 ms** (53.10 tok/s) from scratch.
   * **llama.cpp** achieves 138.17 tok/s (7.24 ms/token) through FP16 precision, hardware-specialized matrix-vector GEMV kernels, and Tensor Core utilization.

2. **Prefill Throughput & Time to First Token (TTFT)**:
   * **llama.cpp** batches all prompt tokens into a single parallel 2D GEMM invocation ($M = \text{prompt\_length}$), saturating GPU compute units to achieve **3,088.72 tokens/sec** and a sub-25 ms TTFT.
   * **INFERNT** currently processes prompt tokens sequentially token-by-token through the forward loop. Transitioning to a batched prefill GEMM will eliminate the 158.70 ms prefill bottleneck and bring TTFT on par with llama.cpp.

3. **Roadmap to Parity**:
   The benchmark directly validates the core engine design while outlining the path toward matching and exceeding production baselines:
   * **Batched Prefill GEMM**: Process entire prompt matrices in parallel rather than token-by-token.
   * **FP16 / BF16 Tensor Cores**: Leverage NVIDIA Tensor Cores (`wmma` / `mma.sync`) to halve memory bandwidth pressure and boost arithmetic throughput.
   * **Fused Attention**: Implement fused FlashAttention-style decode kernels to minimize global memory roundtrips for KV reads.

---

## possible next steps to touch baseline

- [ ] **Batched Prefill GEMM**: Matrix-parallel prompt processing ($M = N_{\text{tokens}}$) to close TTFT and prefill throughput gap.
- [ ] **Precision**: FP16 / BF16 mixed-precision kernels utilizing Tensor Cores (`wmma` / `mma.sync`).
- [ ] **Attention Optimization**: Fused FlashAttention-2 decode kernel to minimize global memory roundtrips.
- [ ] **PagedAttention & KV Paging**: Virtual memory paging for KV cache to support dynamic sequence lengths without maximum length pre-allocation.
- [ ] **Batching**: Continuous request batching with dynamic scheduling.
