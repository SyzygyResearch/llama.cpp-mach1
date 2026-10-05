#include "common.cuh"

int ggml_cuda_mach1_hc_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx, int max_skip);
