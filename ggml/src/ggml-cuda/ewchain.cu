#ifndef EWC_EMU
#include "ewchain.cuh"
#include "unary.cuh"
#endif

#include <cstdlib>

#define EWC_MAX_OPS 8

enum ewc_code : int {
    EWC_IDENT = 0,
    EWC_SCALE,
    EWC_CLAMP,
    EWC_SIGMOID,
    EWC_SILU,
    EWC_TANH,
    EWC_RELU,
    EWC_NEG,
    EWC_EXP,
    EWC_SOFTPLUS,
    EWC_GELU,
    EWC_ADD,
    EWC_SUB,
    EWC_MUL,
    EWC_DIV,
};

struct ewc_op {
    int          code;
    int          other_first;
    float        p0, p1;
    const char * other;
    int64_t      ne[4];
    int64_t      one[4];
    int64_t      onb[4];
};

struct ewc_args {
    int          n_ops;
    const char * x;
    int64_t      xne[4];
    int64_t      xnb[4];
    int64_t      ne[4];
    ewc_op       ops[EWC_MAX_OPS];
};

static __device__ __forceinline__ int64_t ewc_offset(const int64_t i, const int64_t * ne, const int64_t * sne, const int64_t * snb) {
    const int64_t i0 = i % ne[0];
    int64_t r = i / ne[0];
    const int64_t i1 = r % ne[1];
    r /= ne[1];
    const int64_t i2 = r % ne[2];
    const int64_t i3 = r / ne[2];
    return (i0 % sne[0])*snb[0] + (i1 % sne[1])*snb[1] + (i2 % sne[2])*snb[2] + (i3 % sne[3])*snb[3];
}

static __global__ void ewchain_kernel(const ewc_args a, float * dst, const int64_t n) {
    const int64_t i = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float v = *(const float *) (a.x + ewc_offset(i, a.ne, a.xne, a.xnb));
    for (int k = 0; k < a.n_ops; ++k) {
        const ewc_op & op = a.ops[k];
        float o = 0.0f;
        if (op.code >= EWC_ADD) {
            o = *(const float *) (op.other + ewc_offset(i, op.ne, op.one, op.onb));
        }
        switch (op.code) {
            case EWC_IDENT:    break;
            case EWC_SCALE:    v = op.p0 * v + op.p1; break;
            case EWC_CLAMP:    v = fminf(fmaxf(v, op.p0), op.p1); break;
            case EWC_SIGMOID:  v = 1.0f / (1.0f + expf(-v)); break;
            case EWC_SILU:     v = ggml_cuda_op_silu_single(v); break;
            case EWC_TANH:     v = tanhf(v); break;
            case EWC_RELU:     v = fmaxf(v, 0); break;
            case EWC_NEG:      v = -v; break;
            case EWC_EXP:      v = expf(v); break;
            case EWC_SOFTPLUS: v = (v > 20.0f) ? v : logf(1.0f + expf(v)); break;
            case EWC_GELU:     v = ggml_cuda_op_gelu_single(v); break;
            case EWC_ADD:      v = op.other_first ? o + v : v + o; break;
            case EWC_SUB:      v = v - o; break;
            case EWC_MUL:      v = op.other_first ? o * v : v * o; break;
            case EWC_DIV:      v = v / o; break;
        }
    }
    dst[i] = v;
}

static bool ewc_is_view(const ggml_tensor * t) {
    return t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE ||
           t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_NONE;
}

static int ewc_unary_code(const ggml_tensor * t) {
    switch (ggml_get_unary_op(t)) {
        case GGML_UNARY_OP_SIGMOID:  return EWC_SIGMOID;
        case GGML_UNARY_OP_SILU:     return EWC_SILU;
        case GGML_UNARY_OP_TANH:     return EWC_TANH;
        case GGML_UNARY_OP_RELU:     return EWC_RELU;
        case GGML_UNARY_OP_NEG:      return EWC_NEG;
        case GGML_UNARY_OP_EXP:      return EWC_EXP;
        case GGML_UNARY_OP_SOFTPLUS: return EWC_SOFTPLUS;
        case GGML_UNARY_OP_GELU:     return EWC_GELU;
        default:                     return -1;
    }
}

static bool ewc_ranges_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

int ggml_cuda_ewchain_fuse(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int node_idx) {
    static const bool on = getenv("GGML_CUDA_EWCHAIN") != nullptr && atoi(getenv("GGML_CUDA_EWCHAIN")) != 0;
    if (!on) {
        return 0;
    }

    ewc_args a;
    memset(&a, 0, sizeof(a));
    int                 idxs[32];
    enum ggml_op        ops[32];
    int                 n_idx = 0;
    const ggml_tensor * ext[EWC_MAX_OPS + 1];
    int                 n_ext = 0;
    const ggml_tensor * cur   = nullptr;
    int                 last  = -1;

    int cur_idx = -1;
    for (int k = node_idx; k < cgraph->n_nodes && n_idx < 31; ++k) {
        const ggml_tensor * t = cgraph->nodes[k];
        if (cur != nullptr && (cur->flags & GGML_TENSOR_FLAG_OUTPUT || ggml_node_get_use_count(cgraph, cur_idx) != 1)) {
            break;
        }
        if (cur != nullptr && ewc_is_view(t)) {
            if (t->src[0] != cur) {
                continue;
            }
            if (t->op != GGML_OP_RESHAPE || !ggml_is_contiguous(t) || t->data != cur->data) {
                break;
            }
            idxs[n_idx] = k; ops[n_idx] = t->op; n_idx++;
            cur = t;
            cur_idx = k;
            continue;
        }
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || a.n_ops == EWC_MAX_OPS) {
            break;
        }
        ewc_op op;
        memset(&op, 0, sizeof(op));
        const ggml_tensor * other = nullptr;
        switch (t->op) {
            case GGML_OP_REPEAT:
                if (cur != nullptr) {
                    break;
                }
                op.code = EWC_IDENT;
                break;
            case GGML_OP_CONT:
                op.code = cur != nullptr && t->src[0] == cur ? EWC_IDENT : -1;
                break;
            case GGML_OP_SCALE:
                op.code = EWC_SCALE;
                memcpy(&op.p0, (const float *) t->op_params + 0, sizeof(float));
                memcpy(&op.p1, (const float *) t->op_params + 1, sizeof(float));
                break;
            case GGML_OP_CLAMP:
                op.code = EWC_CLAMP;
                memcpy(&op.p0, (const float *) t->op_params + 0, sizeof(float));
                memcpy(&op.p1, (const float *) t->op_params + 1, sizeof(float));
                break;
            case GGML_OP_UNARY:
                op.code = ewc_unary_code(t);
                break;
            case GGML_OP_ADD:
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV:
                op.code = t->op == GGML_OP_ADD ? EWC_ADD : t->op == GGML_OP_SUB ? EWC_SUB :
                          t->op == GGML_OP_MUL ? EWC_MUL : EWC_DIV;
                if (cur == nullptr || t->src[0] == cur) {
                    other = t->src[1];
                } else if (t->src[1] == cur && (t->op == GGML_OP_ADD || t->op == GGML_OP_MUL)) {
                    other = t->src[0];
                    op.other_first = 1;
                } else {
                    op.code = -1;
                }
                if (other != nullptr && (other->type != GGML_TYPE_F32 || !ggml_can_repeat(other, t))) {
                    op.code = -1;
                }
                break;
            default:
                op.code = -1;
                break;
        }
        if (t->op == GGML_OP_REPEAT && cur != nullptr) {
            break;
        }
        if (op.code < 0) {
            break;
        }
        if (other == nullptr && t->op != GGML_OP_REPEAT) {
            if (cur != nullptr ? t->src[0] != cur : (t->src[0]->type != GGML_TYPE_F32 ||
                                                     ggml_nelements(t->src[0]) != ggml_nelements(t))) {
                break;
            }
        }
        if (cur != nullptr && ggml_nelements(t) != ggml_nelements(cur)) {
            break;
        }
        if (cur == nullptr) {
            const ggml_tensor * x = t->src[0];
            if (x->type != GGML_TYPE_F32 || !ggml_can_repeat(x, t)) {
                break;
            }
            a.x = (const char *) x->data;
            for (int d = 0; d < 4; ++d) {
                a.xne[d] = x->ne[d];
                a.xnb[d] = x->nb[d];
                a.ne[d]  = t->ne[d];
            }
            ext[n_ext++] = x;
        }
        if (other != nullptr) {
            for (int j = 0; j < n_idx; ++j) {
                if (cgraph->nodes[idxs[j]] == other) {
                    other = nullptr;
                    break;
                }
            }
            if (other == nullptr) {
                break;
            }
            op.other = (const char *) other->data;
            for (int d = 0; d < 4; ++d) {
                op.ne[d]  = t->ne[d];
                op.one[d] = other->ne[d];
                op.onb[d] = other->nb[d];
            }
            ext[n_ext++] = other;
        }
        a.ops[a.n_ops++] = op;
        idxs[n_idx] = k; ops[n_idx] = t->op; n_idx++;
        cur     = t;
        cur_idx = k;
        last    = k;
    }

    if (a.n_ops < 2) {
        return 0;
    }
    while (n_idx > 0 && idxs[n_idx - 1] != last) {
        n_idx--;
    }
    const ggml_tensor * out = cgraph->nodes[last];
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs, n_idx, ops, &last, 1)) {
        return 0;
    }
    for (int j = 0; j < n_ext; ++j) {
        const ggml_tensor * e = ext[j];
        if (!ewc_ranges_overlap(e, out)) {
            continue;
        }
        if (!(e->data == out->data && ggml_is_contiguous(e) && ggml_nelements(e) == ggml_nelements(out))) {
            return 0;
        }
    }

    const int64_t n  = ggml_nelements(out);
#ifndef EWC_EMU
    const int     bs = 256;
    ewchain_kernel<<<(unsigned) ((n + bs - 1)/bs), bs, 0, ctx.stream()>>>(a, (float *) out->data, n);
    CUDA_CHECK(cudaGetLastError());
#else
    ewc_emu_run(ctx, a, out, n);
#endif
    return last - node_idx;
}
