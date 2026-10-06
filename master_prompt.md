# Master Prompt: Build a General CUDA LLM Inference Engine

You are the primary coding agent for an educational but real CUDA-based LLM inference engine.

Your job is to build the complete project incrementally, from the runtime foundation to executing a real pretrained open-source Transformer model on an NVIDIA GPU in Google Colab.

The engine must be designed as a GENERAL inference runtime, not as a TinyLlama-specific implementation.

TinyLlama is only the FIRST REFERENCE MODEL used to validate the engine.

The long-term goal is:

General Inference Runtime
→ support multiple model configurations
→ support multiple Transformer architectures
→ reuse the same CUDA kernels and runtime infrastructure whenever possible.

Do not optimize prematurely. Correctness, explicit tensor flow, clean architecture, and understandability are the first priorities.

---

# 1. PRIMARY GOAL

Build a small inference engine capable of:

1. Loading model configuration.
2. Loading pretrained model weights.
3. Loading the required tokenizer.
4. Allocating GPU memory.
5. Executing a decoder-only Transformer layer by layer.
6. Using custom CUDA kernels for operations we are learning.
7. Using cuBLAS directly as a black-box GEMM implementation.
8. Maintaining a KV cache during autoregressive generation.
9. Producing logits.
10. Sampling or selecting the next token.
11. Repeating generation until termination.

The final system must run a real pretrained model on an NVIDIA GPU in Google Colab.

---

# 2. FIRST REFERENCE MODEL

Use:

TinyLlama/TinyLlama_v1.1

Hugging Face repository:

https://huggingface.co/TinyLlama/TinyLlama_v1.1

This model is a Llama-family decoder Transformer and should be used as the first validation target.

IMPORTANT:

Do not hardcode TinyLlama's layer count, hidden size, number of heads, vocabulary size, or intermediate size into the engine.

Read the actual configuration from the model's configuration file during model loading.

The code must work from configuration values.

For example, do NOT write:

for (int i = 0; i < 22; i++)

Write:

for (int i = 0; i < config.num_layers; i++)

Likewise, do not hardcode:

2048
32
4
5632
32000

inside kernels or engine logic.

Those values must come from ModelConfig.

---

# 3. MOST IMPORTANT ARCHITECTURAL PRINCIPLE

Separate these concepts:

MODEL
ARCHITECTURE
RUNTIME
KERNELS
WEIGHTS
TOKENIZER

The architecture must not be treated as the same thing as the model.

Conceptually:

Model
→ configuration + weights + tokenizer

Architecture
→ computation recipe

Runtime
→ executes architecture using GPU resources

Kernels
→ low-level implementations of operations

Example:

TinyLlama
→ uses Llama architecture

Our engine
→ contains a Llama architecture backend

Therefore:

TinyLlama ≠ engine

TinyLlama is only one model that the engine can execute.

---

# 4. GENERAL ENGINE DESIGN

Design the engine around configuration and architecture abstractions.

Conceptually:

                 Inference Engine
                        |
                        v
               Model Architecture
                  /          \
                 /            \
                v              v
          Llama Backend    Future Backend
                |
        -----------------
        |       |       |
     Model A Model B Model C

The first supported architecture is Llama-style.

The architecture layer must be isolated enough that another architecture can later be added without rewriting the core runtime.

Do not create a giant file containing model-specific conditionals such as:

if (is_llama)
if (is_gpt)
if (is_qwen)

inside every operation.

Instead, create architecture-specific implementations behind a clean interface.

---

# 5. MODEL CONFIGURATION

Create a ModelConfig structure that contains architecture-relevant information.

At minimum support concepts such as:

vocab_size
hidden_size
num_layers
num_attention_heads
num_key_value_heads
intermediate_size
max_seq_len
head_dim
rms_norm_eps
rope_theta
activation
normalization type
position embedding type
architecture identifier

The exact fields may be extended when required by supported architectures.

The config must be loaded dynamically from the model files.

The runtime must derive dimensions such as head_dim when appropriate instead of requiring redundant hardcoded values.

---

# 6. TENSOR ABSTRACTION

Create a minimal Tensor abstraction.

It should contain at least:

- data pointer
- shape
- number of elements
- datatype
- device/location information
- stride information where required

The Tensor abstraction must support CPU and GPU tensors.

Make tensor layouts explicit.

Do not silently transpose or reshape data without making the operation understandable.

The goal is that a developer can inspect a tensor and understand:

shape
→ datatype
→ memory layout
→ where it lives

---

# 7. MEMORY ARCHITECTURE

Separate persistent model weights from temporary activations.

At initialization:

CPU
→ load model
→ allocate GPU memory
→ copy weights to GPU

During inference:

reuse the GPU-resident weights.

Do not reload weights for every generated token.

Allocate reusable activation buffers whenever practical.

Do not perform repeated cudaMalloc/cudaFree operations inside the token-generation loop.

Create persistent KV-cache storage.

Use a CUDA stream for inference.

Do not put cudaDeviceSynchronize() after every kernel launch.

Operations launched into the same CUDA stream should naturally execute in order unless synchronization is actually required.

---

# 8. MODEL WEIGHT REPRESENTATION

Create:

ModelWeights
TransformerLayerWeights

A Model should conceptually contain:

embedding weights
layer weights
final normalization weights
LM head weights

Each TransformerLayerWeights object should contain the learned tensors needed by one instance of the architecture's Transformer block.

There must be one weight set per layer.

Example:

model.layers[0]
model.layers[1]
...
model.layers[N-1]

But there must be only ONE implementation of the Transformer block.

The same implementation must execute repeatedly with different layer weight objects.

---

# 9. GENERAL TRANSFORMER EXECUTION MODEL

The runtime should conceptually perform:

token ids
→ embedding
→ Transformer block × num_layers
→ final normalization
→ LM head
→ logits
→ sampling
→ next token

Do not create separate code implementations for every layer.

Use:

for (layer = 0; layer < config.num_layers; ++layer)

and execute the same architecture-specific block using:

model.layers[layer]

---

# 10. FIRST ARCHITECTURE BACKEND: LLAMA

Implement a Llama-style decoder-only Transformer backend.

The implementation must derive all dimensions from ModelConfig.

The Llama backend should support the following conceptual flow:

input
→ RMSNorm
→ Q/K/V projections
→ reshape into attention heads
→ RoPE
→ KV-cache append
→ grouped-query attention
→ output projection
→ residual
→ RMSNorm
→ Llama-style MLP
→ residual

Then proceed to the next layer.

Do not replace the architecture with a simplified generic Transformer.

The implementation must correspond to the actual architecture used by the reference model.

---

# 11. GEMM POLICY

Use cuBLAS directly for matrix multiplication.

Do not implement custom GEMM initially.

Treat cuBLAS as a black box.

Create a small wrapper around the required cuBLAS calls so the rest of the engine does not depend on raw cuBLAS API details everywhere.

The wrapper should support the matrix operations required by:

Q projection
K projection
V projection
QKV projection if the chosen implementation uses a combined projection
attention matrix multiplication
attention output projection
MLP projections
LM head

Do not spend time implementing the internals of cuBLAS.

The engine must use cuBLAS as an external high-performance GEMM backend.

---

# 12. CUSTOM CUDA KERNELS

Operations that are useful for learning and are not better treated as GEMM should be implemented as custom CUDA kernels.

Initially implement kernels for:

RMSNorm
RoPE
softmax
residual addition
KV-cache append
attention support operations
sampling/argmax

The exact kernel decomposition can evolve after correctness testing.

Do not fuse everything initially.

Make the computation path explicit.

---

# 13. RMSNORM

Implement RMSNorm as a custom CUDA kernel.

The mathematical operation is:

RMSNorm(x) =
x / sqrt(mean(x^2) + eps) * weight

Use the epsilon and normalization behavior specified by ModelConfig.

Do not hardcode model-specific values inside the kernel.

---

# 14. ROPE

Implement rotary positional embeddings as a custom CUDA kernel.

The implementation must derive:

head dimension
position
RoPE base/theta
pairing behavior

from the model configuration and runtime position.

Apply RoPE correctly to Q and K.

Do not use a framework's RoPE implementation in the actual inference path.

---

# 15. KV CACHE

Implement a persistent KV cache for autoregressive decoding.

There must be separate cache storage for each Transformer layer.

Conceptually:

KV Cache
|
+-- Layer 0
|    +-- K
|    +-- V
|
+-- Layer 1
|    +-- K
|    +-- V
|
+-- ...
|
+-- Layer N-1
     +-- K
     +-- V

The cache must support appending the newly computed K and V at the current sequence position.

The layout must be explicitly defined and documented.

The layout should be chosen with future GPU access patterns in mind, but correctness is more important than optimization for the first version.

---

# 16. GROUPED-QUERY ATTENTION

The first reference model uses fewer KV heads than query heads.

The engine must not assume:

num_attention_heads == num_key_value_heads

Instead support:

num_attention_heads
num_key_value_heads

as independent configuration values.

Implement the query-head to KV-head mapping correctly.

For example, if the architecture specifies 32 query heads and 4 KV heads, the runtime must map multiple query heads to the same KV head.

Do not hardcode this mapping.

Derive it from configuration.

---

# 17. ATTENTION

Initially implement the straightforward attention algorithm.

For the current query:

S = Q K^T / sqrt(head_dim)

then:

P = softmax(S)

then:

O = P V

For the first implementation it is acceptable to materialize attention scores.

Do NOT implement FlashAttention initially.

Do NOT optimize attention memory traffic initially.

First prove that:

Q
→ scores
→ softmax
→ weighted V
→ output

is numerically correct.

Later, attention can be optimized or fused.

---

# 18. MLP

Implement the actual MLP structure required by the Llama-style architecture.

Do not substitute an arbitrary MLP.

Read the activation and intermediate dimensions from ModelConfig.

Use cuBLAS for the matrix multiplications.

Implement activation as a custom CUDA operation when appropriate.

Structure the implementation so that another architecture with a different MLP could later be supported without rewriting the engine core.

---

# 19. RESIDUAL CONNECTIONS

Implement residual addition as a custom CUDA kernel.

Conceptually:

output = input + branch_output

The kernel must operate on arbitrary supported tensor sizes.

---

# 20. FINAL NORMALIZATION + LM HEAD

After all Transformer layers:

hidden
→ final normalization
→ LM head

LM head computes vocabulary logits:

logits = hidden × lm_head_weights

Use cuBLAS.

Do not confuse:

attention scores

with:

language-model logits

Attention scores determine how the current token attends to earlier tokens.

LM logits determine the score of each possible next vocabulary token.

---

# 21. SAMPLING

Start with deterministic greedy decoding.

Given logits:

next_token = argmax(logits)

Implement argmax either on GPU or through a simple host path depending on the first implementation.

Sampling should be a separate subsystem so temperature, top-k, top-p, etc. can be added later.

Do not implement every sampling strategy initially.

---

# 22. TOKENIZER

The engine must be able to convert:

text
→ token ids

and:

token ids
→ text

Use the tokenizer associated with the loaded model.

It is acceptable for tokenizer handling to use a small external dependency or a conversion step.

Do not use a framework model to perform the actual forward pass.

Tokenizer functionality is separate from neural-network execution.

---

# 23. CHECKPOINT CONVERSION

Do not force the runtime to understand every detail of Hugging Face checkpoint storage.

Create a conversion pipeline.

Conceptually:

Hugging Face checkpoint
→ conversion tool
→ engine-native checkpoint
→ C++/CUDA runtime

The conversion tool may be written in Python.

The runtime itself should remain a C++/CUDA executable.

The conversion layer should:

- read the model configuration
- read pretrained tensors
- map source tensor names to engine tensor names
- preserve shapes
- preserve or explicitly convert datatype
- produce a predictable engine-native format

Do not silently change model semantics while converting.

---

# 24. DTYPE STRATEGY

Prioritize simplicity first.

The initial correctness implementation may use FP32 where practical.

After the complete model works:

add FP16/BF16 support where useful.

Do not start with quantization.

Do not start with INT4.

Do not start with complex mixed-precision optimization.

The goal is:

correct FP32 baseline
→ optimized precision later

---

# 25. GENERALITY REQUIREMENTS

The engine must NOT contain TinyLlama-specific constants in the execution code.

Bad:

if (num_layers == 22)
hardcoded hidden size 2048
hardcoded vocabulary 32000
hardcoded 32 heads
hardcoded 4 KV heads

Good:

config.num_layers
config.hidden_size
config.vocab_size
config.num_attention_heads
config.num_key_value_heads

The CUDA kernels must receive dimensions and relevant metadata rather than assuming a particular model.

TinyLlama-specific information belongs only in:

- model files
- configuration
- checkpoint conversion
- model-specific tests

---

# 26. ARCHITECTURE ABSTRACTION

Create an architecture interface or equivalent clean abstraction.

For example, conceptually:

ModelArchitecture
    → prepare model
    → execute transformer layer
    → final processing

First implementation:

LlamaArchitecture

Future possibilities:

GPTArchitecture
QwenArchitecture
other compatible decoder architectures

The architecture abstraction must not become unnecessarily complicated.

Prefer simple composition over a huge inheritance hierarchy.

---

# 27. PROJECT STRUCTURE

Use a clean repository approximately like:

mini_llm/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── tensor.h
│   ├── config.h
│   ├── model.h
│   ├── weights.h
│   ├── kv_cache.h
│   ├── architecture.h
│   └── engine.h
├── src/
│   ├── tensor.cpp
│   ├── config.cpp
│   ├── model.cpp
│   ├── loader.cpp
│   ├── engine.cpp
│   └── main.cpp
├── architectures/
│   └── llama/
│       ├── llama_model.cpp
│       ├── llama_layer.cpp
│       └── llama_attention.cpp
├── cuda/
│   ├── rmsnorm.cu
│   ├── rope.cu
│   ├── softmax.cu
│   ├── residual.cu
│   ├── kv_cache.cu
│   ├── attention.cu
│   └── sampling.cu
├── gemm/
│   └── cublas_wrapper.cu
├── loader/
│   └── checkpoint_loader.cpp
├── tokenizer/
├── tools/
│   └── convert_checkpoint.py
└── tests/

The exact structure may be adjusted when necessary, but keep the architecture understandable.

---

# 28. GOOGLE COLAB REQUIREMENT

The project must run in Google Colab on an available NVIDIA GPU.

The repository will be cloned from GitHub.

The README must provide a clean workflow such as:

git clone <repository>
cd <repository>

install/build dependencies

download model

convert checkpoint

build engine

run inference

The code must detect the available CUDA GPU at startup and print useful device information.

Do not assume a specific GPU model.

Do not assume a fixed amount of VRAM.

Handle allocation failures gracefully.

---

# 29. TESTING STRATEGY

The testing strategy is critical.

Do not attempt the entire model first.

Validate incrementally.

Create tests for:

Tensor allocation
Tensor shape handling
CPU/GPU copies
cuBLAS GEMM
RMSNorm
RoPE
softmax
residual addition
KV cache append
GQA mapping
attention
MLP
single Transformer block
multiple Transformer layers
final normalization
LM head
full model logits
greedy generation

Where possible compare GPU results against a trusted CPU/reference implementation.

For selected test tensors, compare values numerically within reasonable tolerance.

---

# 30. REFERENCE IMPLEMENTATION

A Python/PyTorch reference implementation may be used ONLY for correctness validation.

For example:

reference model
→ same input tokens
→ capture intermediate tensors/logits
→ compare against our CUDA engine

The reference implementation must never replace our C++/CUDA inference path.

Our engine must perform the actual inference.

---

# 31. DEBUG MODE

Create a debug mode capable of printing:

model configuration
tensor shapes
layer index
current sequence position
GPU memory information
selected intermediate tensor statistics

Do not dump enormous tensors by default.

Useful debug information should be opt-in.

---

# 32. ERROR HANDLING

Check CUDA errors.

Check cuBLAS errors.

Validate tensor shapes before operations where practical.

Validate that checkpoint tensors match expected shapes.

Produce useful error messages.

Do not silently continue after a failed allocation, failed kernel launch, or incompatible tensor.

---

# 33. CODE STYLE

IMPORTANT:

DO NOT WRITE COMMENTS INSIDE SOURCE CODE.

Keep the source code clean.

Do not add long explanatory comments above functions.

Do not fill CUDA files with comments explaining basic CUDA concepts.

Put explanations in README.md and architecture documentation instead.

Use clear variable and function names so the code remains readable without comments.

Do not create unnecessary boilerplate.

Do not create giant source files.

Keep components focused.

---

# 34. DOCUMENTATION

Although source code must not contain comments, README.md must explain:

- project purpose
- architecture
- model vs architecture distinction
- model loading pipeline
- tensor shapes
- memory layout
- CUDA kernel responsibilities
- cuBLAS responsibilities
- KV cache layout
- attention flow
- generation loop
- build process
- Colab setup
- checkpoint conversion
- supported architecture
- current limitations
- future optimization stages

The README should contain diagrams where useful.

---

# 35. DEVELOPMENT PHILOSOPHY

This is a learning-first systems project.

Do not optimize everything immediately.

Do not implement:

FlashAttention
custom GEMM
quantization
continuous batching
speculative decoding
CUDA graphs
Tensor parallelism
pipeline parallelism
kernel fusion everywhere
complex memory pools

until the baseline model actually runs.

The first successful milestone is:

one GPU
one model
one batch
autoregressive decoding
correct output

Once correctness exists, optimize one subsystem at a time.

---

# 36. REQUIRED DEVELOPMENT PHASES

Follow this progression.

PHASE 1
Project skeleton
CMake
CUDA initialization
GPU detection
Tensor abstraction

PHASE 2
cuBLAS wrapper
GEMM test
CPU vs GPU GEMM verification

PHASE 3
Memory management
model configuration
basic model structures

PHASE 4
RMSNorm
residual kernels

PHASE 5
RoPE

PHASE 6
softmax

PHASE 7
KV cache

PHASE 8
attention
single query
then GQA

PHASE 9
Llama MLP

PHASE 10
single Transformer layer

PHASE 11
multiple layers driven by config.num_layers

PHASE 12
final RMSNorm
LM head

PHASE 13
tokenizer integration

PHASE 14
checkpoint conversion and pretrained weight loading

PHASE 15
full TinyLlama inference

PHASE 16
autoregressive generation loop

PHASE 17
cleanup and profiling

PHASE 18
architecture-generalization work

After the first model runs, prove that model dimensions are configuration-driven by testing another compatible model if practical.

---

# 37. IMPORTANT IMPLEMENTATION RULE

At every stage ask:

"What mathematical operation am I implementing?"

Then:

"What tensor shape represents it?"

Then:

"Where does that tensor live?"

Then:

"Which CUDA kernel or cuBLAS operation executes it?"

The implementation should preserve this chain:

Mathematics
→ tensor
→ memory
→ kernel/GEMM
→ output tensor

Avoid abstractions that hide this completely.

This engine is intended to make the relationship between Transformer mathematics and GPU execution visible.

---

# 38. FINAL RUNTIME FLOW

The final runtime should conceptually look like:

Load ModelConfig
        ↓
Load tokenizer
        ↓
Load pretrained weights
        ↓
Convert/load engine tensors
        ↓
Allocate GPU model weights
        ↓
Allocate KV cache
        ↓
Receive prompt
        ↓
Tokenize
        ↓
Embedding
        ↓
for layer = 0 ... config.num_layers - 1
        ↓
    Llama Transformer Block
        ↓
Final RMSNorm
        ↓
LM Head via cuBLAS
        ↓
Logits
        ↓
Argmax
        ↓
Next token
        ↓
Append token
        ↓
Repeat

---

# 39. FIRST IMPLEMENTATION PRIORITY

Do not generate the entire finished project in one giant step.

Start with the project foundation.

Implement PHASE 1 first.

After PHASE 1:

1. Show the files created.
2. Show the relevant tensor/config interfaces.
3. Show the build commands.
4. Build it.
5. Run it.
6. Report the result.
7. Then proceed to PHASE 2.

Continue incrementally.

Never skip validation just to move faster.

---

# 40. FINAL DESIGN PRINCIPLE

The project should end up conceptually like this:

                 MODEL
        config + weights + tokenizer
                    |
                    v
              ARCHITECTURE
            Llama / future ...
                    |
                    v
                RUNTIME
       memory + execution + scheduling
                    |
          +---------+---------+
          |                   |
          v                   v
       cuBLAS             CUDA Kernels
        GEMM        RMSNorm / RoPE / Softmax
                    |
                    v
                 GPU
                    |
                    v
               logits
                    |
                    v
              next token

TinyLlama is simply the first real model running through this stack.

Build the engine so that adding another compatible model primarily means loading a different configuration and weight set.

Adding a fundamentally different architecture should require a new architecture backend, not a rewrite of the entire runtime.

Begin with PHASE 1.