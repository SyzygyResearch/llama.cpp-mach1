#include "common.cuh"

void ggml_cuda_op_mach1_ne_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_mach1_embed_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_mach1_exp_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_mach1_exp_basis(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// consumer nodes folded into the rt out stage's store (GGML_MACH1_RT_TAIL,
// see ggml_cuda_mach1_rt_tail_fuse): out = ((y*sv)*gate) + add + add2, each
// member optional, all [nt, m] row-major except gate which is [nt]
struct mach1_rt_tail {
    const float * add   = nullptr;
    const float * add2  = nullptr;
    const float * gate  = nullptr;
    ggml_tensor * dst   = nullptr;   // the region's output node
};

// x_data_override / xperm: the XPERM input-reorder fold reads x from another
// tensor's data through a row map (see ggml_cuda_mach1_xperm_fuse); defaults
// keep the ordinary path.
void ggml_cuda_op_mach1_rt_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                              const void * x_data_override = nullptr, const int * xperm = nullptr,
                              const mach1_rt_tail * tail = nullptr);
void ggml_cuda_op_mach1_head_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_mach1_embed_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_mach1_supported(const ggml_tensor * op);

// skinny stock-op GEMM lane (GGML_MACH1_SKINNY_MM=1): F32xF32 mul_mats with
// <= 256 output rows at nt >= mach1_pp_min() (256; 128 under PPLOW, never
// below 64) run as one-time-f16-mirrored weight x per-call f16 activations
// through the per-shape-handle GemmEx path instead of the sgemm fallback
// (prefill2k census: 673 us vs q4_K's 128 for the same shape). Prefill-only by
// the nt gate, so decode numerics are untouched. Returns false to fall through.
bool ggml_cuda_mach1_skinny_mm(ggml_backend_cuda_context & ctx,
                               const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// whole-layer expert FFN fusion (GGML_MACH1_EXP_FFN=1): matcher + executor
// for the expert subgraph starting at a MACH1_EXP_MM node, returns the number
// of nodes to skip or 0
int ggml_cuda_mach1_exp_ffn_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);

// measurement only: source line of the guard that rejected the last
// ggml_cuda_mach1_exp_ffn_fuse attempt, 0 if it has never rejected.  A fused
// region is skipped before it is timed, so a matcher that never engages is
// invisible to the per-op census; this names the guard that is firing.
int ggml_cuda_mach1_exp_ffn_reject_line();

// shared-expert branch fusion (GGML_MACH1_SHEXP_FFN=1): matcher + executor
// for the shexp chain starting at its gate MACH1_RT_MM node, returns the
// number of nodes to skip or 0
int ggml_cuda_mach1_shexp_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);

// v_tiled fold (GGML_MACH1_VTILED=1): matcher + executor for the GDN
// grouped->tiled row-reorder glue after a MACH1_RT_MM node, returns the
// number of nodes to skip or 0
int ggml_cuda_mach1_vtiled_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);
// XPERM fold: the tiled->grouped input reorder ahead of ssm_out runs as the
// walk's u-prologue gather (GGML_MACH1_XPERM=1); entered at the CONT node
int ggml_cuda_mach1_xperm_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);

// qkv trio batching (GGML_MACH1_QKV_BATCH=1): matcher + executor for three
// adjacent independent same-input MACH1_RT_MM nodes (attention wq/wk/wv) as
// one launch, returns the number of nodes to skip or 0
int ggml_cuda_mach1_qkv_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);

// rt tail fold (GGML_MACH1_RT_TAIL=1): matcher + executor for the elementwise
// consumers of a standalone MACH1_RT_MM node - the attention/GDN residual ADD
// and the shared-expert sigmoid-gate MUL + moe/residual ADDs - folded into the
// out transform's store, returns the number of nodes to skip or 0
int ggml_cuda_mach1_rt_tail_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);

// GDN-core fusion (GGML_MACH1_GDN_FUSE=1): matcher + executor for the
// per-head norm/gate chain around GGML_OP_GATED_DELTA_NET, starting at its
// q L2_NORM node, returns the number of nodes to skip or 0
int ggml_cuda_mach1_gdn_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);

// batched-decode GDN prep fusion (GGML_MACH1_GDN_PREP=1, ns >= 2): the six
// tiny pre-chain nodes (l2 norms, beta sigmoid, alpha gate chain) ahead of
// GGML_OP_GATED_DELTA_NET as one launch, entered at the q L2_NORM; the
// delta-rule node itself stays on its own launch
int ggml_cuda_mach1_gdn_prep_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);

// full GDN glue fusion (GGML_MACH1_GDN_FULL != 0): the same region extended
// back over the recurrent-state get_rows and the fused conv+silu, entered at
// the get_rows, returns the number of nodes to skip or 0
int ggml_cuda_mach1_gdn_full_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);
// GDN_FULL level 3: the region extended backwards over the beta/alpha
// projections and the conv-state round trip; entered at the beta mul_mat
int ggml_cuda_mach1_gdn_proj_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);
