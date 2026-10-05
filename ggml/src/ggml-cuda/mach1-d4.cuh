#include "common.cuh"

bool ggml_cuda_mach1_d4_supported(const ggml_tensor * op);
void ggml_cuda_op_mach1_d4_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

int ggml_cuda_mach1_d4_ffn_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);

int ggml_cuda_mach1_d4_pair_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);
