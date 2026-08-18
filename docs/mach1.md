# The mach1 codec

This fork runs Mach-1 checkpoints: models whose weights ship as packed
trellis code streams and are decoded on the fly inside the compute
kernels. The codec is additive over a qwen35moe-topology MoE - a gated
delta-net / attention hybrid with 256 routed experts plus a shared expert.
This repo contains the decode-only runtime: the ops, kernels and model
graph needed to run a released checkpoint. The exporter that produces the
code streams is not part of this repo.

The codec is not a `ggml` quantization type. Codec tensors are raw code
streams stored under stock ggml types, consumed by fork-provided ops. That
is why stock llama.cpp cannot load these checkpoints, why `llama-quantize`
refuses them, and why this fork's `ggml-*` libraries must never be mixed
with stock builds (the op enum differs - an ABI break, not a graceful
failure).

## The ops

Eight ops carry the fork. Full contracts (shapes, bit layouts, exact op
order) are in the comments in `ggml/include/ggml.h`; one line each:

| Op | What it computes |
|---|---|
| `GGML_OP_MACH1_NE_MM` | dense matmul, weights walked out of an L=12 bitshift trellis through a per-chunk LUT with group-128 scales |
| `GGML_OP_MACH1_EMBED_ROWS` | embedding row gather from int3 asymmetric grouped codes |
| `GGML_OP_MACH1_EXP_MM` | routed-expert matmul: per-expert weights materialized from a tail-biting L=16 trellis via a hashed LUT and Hadamard rotations, kept and demoted expert tiers |
| `GGML_OP_MACH1_EXP_BASIS` | shared low-rank residual for demoted experts, y = B (c_e * (A x)), with an optional fused accumulate |
| `GGML_OP_MACH1_RT_MM` | dense matmul with the same trellis/LUT/Hadamard machinery as `EXP_MM` but a single dense tensor |
| `GGML_OP_MACH1_HEAD_MM` | lm_head matmul from int5 group-64 codes |
| `GGML_OP_MACH1_EMBED_GATHER` | lossless nibble-LUT embedding gather (reproduces the source bf16 bit patterns exactly) |
| `GGML_OP_ARGMAX_MASKED` | argmax along rows excluding up to 15 fixed column indices; the backend half of masked-greedy sampling |

Decode is deterministic by construction: the kernels reproduce the release
decoder's exact fp32 op order, so a given payload decodes to the same
weights on every conforming backend.

## Backend support

| Backend | Coverage |
|---|---|
| CUDA | all eight ops, plus graph fusion: multi-op regions of the decode graph (`gdn_full`, `shexp_gu`, `qkvzb`, `exp_mega`) are matched and replaced by fused kernels. This is the performance path. |
| Vulkan | all seven `MACH1_*` ops via 12 compute shaders (the expert and dense trellis matmuls are multi-pass: walk / u / out). No fusion. `ARGMAX_MASKED` has no Vulkan kernel; the scheduler places it elsewhere. |
| CPU | vectorised reference implementations of all eight ops (AVX-512, AVX2, NEON), used both stand-alone and as the test oracle. |

Certified bit-exact greedy streams are CUDA and CPU only. Vulkan output is
correct to op-test tolerances but is not certified stream-identical.

The CPU path is usable but slow relative to llama.cpp's hand-tuned quant
kernels - roughly 3x behind Q4_K_M decode on AVX2 hosts and ~5x on AVX-512
(see README.md). A GPU is strongly recommended.

## Verifying a build

Three checks, cheapest first:

1. **Op tests.** `test-backend-ops -o <op>` for each `MACH1_*` op (and
   `ARGMAX_MASKED`) compares the backend against the CPU reference on
   real checkpoint shapes.
2. **Codec unit test.** `test-mach1-codec <fixtures.gguf>` decodes fixture
   tensors sliced from a released checkpoint and checks the results,
   exercising the full payload path without loading a whole model.
3. **Fusion census.** Run decode with `GGML_MACH1_TIME=1` and check the
   kernel census for `gdn_full`, `shexp_gu`, `qkvzb`, `exp_mega`. If they
   are missing, the fused regions are not engaging and decode will be much
   slower than it should be (see REBASE.md).

## Environment variables

Three are user-relevant:

- `GGML_MACH1_TIME=1` - per-kernel timing and a kernel census on stderr.
  The first thing to reach for when performance looks wrong.
- `GGML_MACH1_DEBUG=1` - verbose codec diagnostics on stderr.
- `GGML_MACH1_CPU_SCALAR=1` - forces the CPU path to the scalar reference
  walk instead of the vectorised one. An escape hatch for suspected
  vectorisation bugs and an honest A/B within one process.

Other `GGML_MACH1_*` variables exist. They are engineering and debug
toggles: every default is the certified configuration, and they are not
supported interfaces - flipping them can produce slow, wrong or
non-reproducible output.
