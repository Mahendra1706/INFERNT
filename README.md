# INFERNT — CUDA LLM Inference Engine

A general-purpose CUDA-based LLM inference engine built from scratch for learning and understanding how Transformer inference works at the GPU level.

## Architecture

The engine separates these concepts:

```
MODEL         = configuration + weights + tokenizer
ARCHITECTURE  = computation recipe (e.g., Llama decoder)
RUNTIME       = memory management + execution + scheduling
KERNELS       = custom CUDA implementations (RMSNorm, RoPE, softmax, ...)
GEMM          = cuBLAS black-box matrix multiplication
```

The first supported architecture is **Llama-style decoder-only Transformer**.

TinyLlama/TinyLlama_v1.1 is the first reference model — it validates the engine but does not define it.

## Project Structure

```
infernt/
├── CMakeLists.txt
├── README.md
├── include/           Headers
│   ├── tensor.h       Tensor abstraction
│   ├── config.h       Model configuration
│   └── cuda_utils.h   CUDA error checking, GPU info
├── src/               Implementations
│   ├── tensor.cpp     Tensor (CPU/GPU memory, shapes, strides)
│   ├── config.cpp     ModelConfig (validation, derived fields)
│   ├── cuda_init.cu   GPU detection and initialization
│   └── main.cpp       Entry point and smoke tests
├── architectures/     Architecture-specific backends
│   └── llama/         Llama decoder implementation (future)
├── cuda/              Custom CUDA kernels (future)
├── gemm/              cuBLAS wrapper (future)
├── loader/            Checkpoint loading (future)
├── tokenizer/         Tokenizer integration (future)
├── tools/             Conversion scripts (future)
└── tests/             Test suite (future)
```

## Building

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Running

```bash
./build/bin/infernt
```

This prints GPU device information and runs validation tests for the Tensor and ModelConfig subsystems.

## Tensor Abstraction

The `Tensor` class tracks:
- **data pointer** — raw memory on CPU or GPU
- **shape** — vector of dimension sizes
- **strides** — row-major stride computation
- **dtype** — FP32, FP16, BF16, INT32, INT64, UINT8
- **device** — CPU or CUDA
- **num_elements** — total element count

Supports: allocation, zeroing, CPU↔GPU transfer, move semantics, reshape (view), debug printing.

## Model Configuration

`ModelConfig` holds architecture-relevant parameters:

| Field | Description |
|-------|-------------|
| `vocab_size` | Vocabulary size |
| `hidden_size` | Model hidden dimension |
| `num_layers` | Number of Transformer layers |
| `num_attention_heads` | Query head count |
| `num_key_value_heads` | KV head count (GQA support) |
| `intermediate_size` | MLP intermediate dimension |
| `max_seq_len` | Maximum sequence length |
| `head_dim` | Per-head dimension (derived) |
| `rms_norm_eps` | RMSNorm epsilon |
| `rope_theta` | RoPE base frequency |
| `activation` | Activation function (SiLU, GELU, ReLU) |
| `norm_type` | Normalization type (RMSNorm, LayerNorm) |
| `position_embedding` | Position embedding type (RoPE, Absolute, ALiBi) |

All dimensions are configuration-driven. No model-specific constants exist in the engine code.

## Development Phases

- [x] Phase 1: Project skeleton, CMake, CUDA init, GPU detection, Tensor, Config
- [ ] Phase 2: cuBLAS wrapper, GEMM test
- [ ] Phase 3: Memory management, model structures
- [ ] Phase 4: RMSNorm, residual kernels
- [ ] Phase 5: RoPE
- [ ] Phase 6: Softmax
- [ ] Phase 7: KV cache
- [ ] Phase 8: Attention (single query → GQA)
- [ ] Phase 9: Llama MLP
- [ ] Phase 10: Single Transformer layer
- [ ] Phase 11: Multi-layer execution
- [ ] Phase 12: Final RMSNorm + LM head
- [ ] Phase 13: Tokenizer integration
- [ ] Phase 14: Checkpoint conversion + weight loading
- [ ] Phase 15: Full TinyLlama inference
- [ ] Phase 16: Autoregressive generation
- [ ] Phase 17: Cleanup and profiling
- [ ] Phase 18: Architecture generalization

## Current Limitations

- No model loading yet
- No CUDA kernels yet (Phase 4+)
- No cuBLAS integration yet (Phase 2)
- FP32 only (precision optimization deferred)
- Single batch, single GPU
