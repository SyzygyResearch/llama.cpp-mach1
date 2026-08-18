#include <algorithm>
#include <cstdint>
#include <cstring>

#include "argmax.cuh"
#include "common.cuh"
#include "sum.cuh"

static __global__ void argmax_f32(const float * __restrict__ x, int32_t * __restrict__ dst, const int64_t ncols) {
    const int64_t row = blockIdx.x;

    float maxval = -FLT_MAX;
    int   argmax = -1;
    const float * rowx = x + row * ncols;

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float val = rowx[col];
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }

        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
            }
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
                const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
                if (val > maxval) {
                    maxval = val;
                    argmax = col;
                }
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        dst[row] = argmax;
    }
}

struct argmax_mask_params {
    int32_t n;
    int32_t ids[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1];
};

static_assert(sizeof(argmax_mask_params) == GGML_MAX_OP_PARAMS);

// mirrors argmax_f32 semantics: a NaN candidate never wins, so NaN is never
// selected; equal finite values tie-break to the lowest column
static __device__ __forceinline__ bool argmax_masked_better(
        const float candidate_value,
        const int candidate_col,
        const float current_value,
        const int current_col) {
    if (candidate_col < 0 || isnan(candidate_value)) {
        return false;
    }
    if (current_col < 0 || isnan(current_value)) {
        return true;
    }
    return candidate_value > current_value ||
            (candidate_value == current_value && candidate_col < current_col);
}

static __global__ void argmax_masked_f32(
        const float * __restrict__ x,
        int32_t * __restrict__ dst,
        const int64_t ncols,
        const argmax_mask_params mask) {
    const int64_t row = blockIdx.x;

    float maxval = -INFINITY;
    int argmax = -1;
    const float * rowx = x + row*ncols;

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        bool is_masked = false;
        for (int i = 0; i < mask.n; ++i) {
            if (col == mask.ids[i]) {
                is_masked = true;
                break;
            }
        }
        const float val = rowx[col];
        if (!is_masked && argmax_masked_better(val, col, maxval, argmax)) {
            maxval = val;
            argmax = col;
        }
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        if (argmax_masked_better(val, col, maxval, argmax)) {
            maxval = val;
            argmax = col;
        }
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }

        __syncthreads();

        if (warp_id == 0) {
            maxval = lane_id < n_warps ? shared_maxval[lane_id] : -INFINITY;
            argmax = lane_id < n_warps ? shared_argmax[lane_id] : -1;
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
                const int col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
                if (argmax_masked_better(val, col, maxval, argmax)) {
                    maxval = val;
                    argmax = col;
                }
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        dst[row] = argmax;
    }
}

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    int32_t     * dst_d  = (int32_t     *) dst->data;

    cudaStream_t stream = ctx.stream();

    const int64_t num_blocks = nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    if (dst->op == GGML_OP_ARGMAX_MASKED) {
        argmax_mask_params mask = {};
        std::memcpy(&mask, dst->op_params, sizeof(mask));
        GGML_ASSERT(mask.n >= 0 && mask.n <= (int32_t) (GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1));
        argmax_masked_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, ne00, mask);
    } else {
        argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, ne00);
    }
}
