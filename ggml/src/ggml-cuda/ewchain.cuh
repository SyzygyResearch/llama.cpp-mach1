#pragma once

#include "common.cuh"

int ggml_cuda_ewchain_fuse(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int node_idx);
