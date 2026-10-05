#include "mach1-hc.cuh"
#include "mmvq.cuh"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#define M1HC_HC 4
#define M1HC_NC 4

static bool m1hc_on() {
    static const bool v = [](){
        const char * s = getenv("GGML_MACH1_HC");
        return s == nullptr || atoi(s) != 0;
    }();
    return v;
}

static bool m1hc_q81_on() {
    static const bool v = [](){
        const char * s = getenv("GGML_MACH1_HC_Q81");
        return s == nullptr || atoi(s) != 0;
    }();
    return v;
}

static bool m1hc_mb_on() {
    static const bool v = [](){
        const char * s = getenv("GGML_MACH1_HC_MB");
        return s == nullptr || atoi(s) != 0;
    }();
    return v;
}

static __device__ __forceinline__ void m1hc_q8_1(block_q8_1 * y, const int64_t ib, const float xi) {
    float amax = fabsf(xi);
    float sum  = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    const int iqs = threadIdx.x % QK8_1;
    y[ib].qs[iqs] = q;
    if (iqs == 0) {
        y[ib].ds = make_half2(d, sum);
    }
}

struct m1hc_inj_args {
    const nv_bfloat16 * w;
    float *             out;
    float               s1, b1, s2, b2;
};

template <int BS, bool INJ>
static __global__ void __launch_bounds__(BS) m1hc_post_norm_kernel(
        const float * oa, const float * ob, const float * gv, const float * h, const float * w,
        float * hp, float * hn, const int ncols, const float eps, const m1hc_inj_args ia, block_q8_1 * yq) {
    static_assert(!INJ || BS == M1HC_HC*256, "INJ maps one 256-thread group per gate");
    ggml_cuda_pdl_lc();
    const int tid = threadIdx.x;
    __shared__ float s_sum[M1HC_HC][32];
    __shared__ float s_inj[M1HC_HC][32];
    if constexpr (INJ) {
        if (tid < M1HC_HC*32) {
            s_inj[tid/32][tid%32] = 0.0f;
        }
    }

    float xv[M1HC_HC][M1HC_NC];
    float tmp[M1HC_HC];
    ggml_cuda_pdl_sync();
    float o[M1HC_NC];
#pragma unroll
    for (int k = 0; k < M1HC_NC; ++k) {
        const int col = tid + k*BS;
        o[k] = col < ncols ? (ob != nullptr ? __fadd_rn(oa[col], ob[col]) : oa[col]) : 0.0f;
    }
#pragma unroll
    for (int s = 0; s < M1HC_HC; ++s) {
        const float g = gv[s];
        tmp[s] = 0.0f;
#pragma unroll
        for (int k = 0; k < M1HC_NC; ++k) {
            const int col = tid + k*BS;
            if (col < ncols) {
                const float xi = __fadd_rn(h[s*ncols + col], __fmul_rn(o[k], g));
                xv[s][k] = xi;
                tmp[s] += xi * xi;
            }
        }
    }
    float scale[M1HC_HC];
    if (hn != nullptr) {
#pragma unroll
        for (int s = 0; s < M1HC_HC; ++s) {
            tmp[s] = block_reduce<block_reduce_method::SUM, BS>(tmp[s], s_sum[s]);
            const float mean = tmp[s] / ncols;
            scale[s] = rsqrtf(mean + eps);
        }
    } else {
        __syncthreads();
    }
#pragma unroll
    for (int s = 0; s < M1HC_HC; ++s) {
#pragma unroll
        for (int k = 0; k < M1HC_NC; ++k) {
            const int col = tid + k*BS;
            if (col < ncols) {
                hp[s*ncols + col] = xv[s][k];
                if (hn != nullptr) {
                    const float v = __fmul_rn(__fmul_rn(scale[s], xv[s][k]), w[s*ncols + col]);
                    hn[s*ncols + col] = v;
                    if (yq != nullptr) {
                        m1hc_q8_1(yq, (s*ncols + col)/QK8_1, v);
                    }
                }
            }
        }
    }
    if constexpr (INJ) {
        __syncthreads();
        const int r = tid / 256;
        const int t = tid % 256;
        const nv_bfloat162 * w2 = (const nv_bfloat162 *) (ia.w + (int64_t) r*M1HC_HC*ncols);
        const float2 *       y2 = (const float2 *) hn;
        float sum = 0.0f;
        for (int c2 = t; c2 < M1HC_HC*ncols/2; c2 += 256) {
            const nv_bfloat162 wv = w2[c2];
            const float2       yv = y2[c2];
            sum = __fmaf_rn(__bfloat162float(wv.x), yv.x, sum);
            sum = __fmaf_rn(__bfloat162float(wv.y), yv.y, sum);
        }
        sum = warp_reduce_sum(sum);
        s_inj[r][t/32] = sum;
        __syncthreads();
        if (t < 32) {
            sum = warp_reduce_sum(s_inj[r][t]);
            if (t == 0) {
                const float v = __fmaf_rn(ia.s1, sum, ia.b1);
                ia.out[r] = __fmaf_rn(ia.s2, 1.0f / (1.0f + expf(-v)), ia.b2);
            }
        }
    }
}

#define M1HC_INJ_U 10

template <bool INJ, bool BAR>
static __global__ void __launch_bounds__(1024) m1hc_post_norm_mb_kernel(
        const float * oa, const float * ob, const float * gv, const float * h, const float * w,
        float * hp, float * hn, const int ncols, const float eps, const m1hc_inj_args ia, block_q8_1 * yq,
        unsigned int * done, unsigned int * bar) {
    constexpr int BS = 1024;
    ggml_cuda_pdl_lc();
    const int tid = threadIdx.x;
    const int s   = blockIdx.x;
    __shared__ float s_sum[32];
    __shared__ float s_inj[M1HC_HC][32];
    __shared__ bool  s_last;
    if constexpr (INJ) {
        if (tid < M1HC_HC*32) {
            s_inj[tid/32][tid%32] = 0.0f;
        }
    }

    ggml_cuda_pdl_sync();
    const float g = gv[s];
    float o[M1HC_NC];
    float hv[M1HC_NC];
    float wv[M1HC_NC];
#pragma unroll
    for (int k = 0; k < M1HC_NC; ++k) {
        const int col = tid + k*BS;
        o[k]  = col < ncols ? (ob != nullptr ? __fadd_rn(oa[col], ob[col]) : oa[col]) : 0.0f;
        hv[k] = col < ncols ? h[s*ncols + col] : 0.0f;
        wv[k] = col < ncols && hn != nullptr ? w[s*ncols + col] : 0.0f;
    }
    float xv[M1HC_NC] = {};
    float tmp = 0.0f;
#pragma unroll
    for (int k = 0; k < M1HC_NC; ++k) {
        const int col = tid + k*BS;
        if (col < ncols) {
            const float xi = __fadd_rn(hv[k], __fmul_rn(o[k], g));
            xv[k] = xi;
            tmp += xi * xi;
        }
    }
    if constexpr (BAR) {
#pragma unroll
        for (int k = 0; k < M1HC_NC; ++k) {
            asm volatile("" :: "f"(xv[k]), "f"(wv[k]));
        }
        __syncthreads();
        if (tid == 0) {
            const unsigned int old    = atomicAdd(bar, 1u);
            const unsigned int target = old - old % gridDim.x + gridDim.x;
            while ((int) (*(volatile unsigned int *) bar - target) < 0) {
            }
        }
        __syncthreads();
    }
    float scale = 0.0f;
    if (hn != nullptr) {
        tmp = block_reduce<block_reduce_method::SUM, BS>(tmp, s_sum);
        const float mean = tmp / ncols;
        scale = rsqrtf(mean + eps);
    }
#pragma unroll
    for (int k = 0; k < M1HC_NC; ++k) {
        const int col = tid + k*BS;
        if (col < ncols) {
            hp[s*ncols + col] = xv[k];
            if (hn != nullptr) {
                const float v = __fmul_rn(__fmul_rn(scale, xv[k]), wv[k]);
                hn[s*ncols + col] = v;
                if (yq != nullptr) {
                    m1hc_q8_1(yq, (s*ncols + col)/QK8_1, v);
                }
            }
        }
    }
    if constexpr (INJ) {
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            s_last = atomicAdd(done, 1u) == gridDim.x - 1;
        }
        __syncthreads();
        if (!s_last) {
            return;
        }
        if (tid == 0) {
            *done = 0;
        }
        const int r  = tid / 256;
        const int t  = tid % 256;
        const int n2 = M1HC_HC*ncols/2;
        const nv_bfloat162 * w2 = (const nv_bfloat162 *) (ia.w + (int64_t) r*M1HC_HC*ncols);
        const float2 *       y2 = (const float2 *) hn;
        float sum = 0.0f;
        for (int c0 = t; c0 < n2; c0 += 256*M1HC_INJ_U) {
            nv_bfloat162 wr[M1HC_INJ_U];
            float2       yr[M1HC_INJ_U];
#pragma unroll
            for (int u = 0; u < M1HC_INJ_U; ++u) {
                const int c2 = c0 + u*256;
                if (c2 < n2) {
                    wr[u] = w2[c2];
                    yr[u] = __ldcg(y2 + c2);
                }
            }
#pragma unroll
            for (int u = 0; u < M1HC_INJ_U; ++u) {
                if (c0 + u*256 < n2) {
                    sum = __fmaf_rn(__bfloat162float(wr[u].x), yr[u].x, sum);
                    sum = __fmaf_rn(__bfloat162float(wr[u].y), yr[u].y, sum);
                }
            }
        }
        sum = warp_reduce_sum(sum);
        s_inj[r][t/32] = sum;
        __syncthreads();
        if (t < 32) {
            sum = warp_reduce_sum(s_inj[r][t]);
            if (t == 0) {
                const float v = __fmaf_rn(ia.s1, sum, ia.b1);
                ia.out[r] = __fmaf_rn(ia.s2, 1.0f / (1.0f + expf(-v)), ia.b2);
            }
        }
    }
}

template <int BS>
static __global__ void __launch_bounds__(BS) m1hc_mix_kernel(
        const float * up, const float * hn, float * x, const int n, const float scale, const float bias,
        const int has_scale) {
    ggml_cuda_pdl_lc();
    const int tid = threadIdx.x;
    float r[M1HC_NC];
    ggml_cuda_pdl_sync();
#pragma unroll
    for (int k = 0; k < M1HC_NC; ++k) {
        const int col = tid + k*BS;
        r[k] = 0.0f;
        if (col < n) {
#pragma unroll
            for (int s = 0; s < M1HC_HC; ++s) {
                const float m = __fmul_rn(1.0f / (1.0f + expf(-up[s*n + col])), hn[s*n + col]);
                r[k] = s == 0 ? m : __fadd_rn(r[k], m);
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < M1HC_NC; ++k) {
        const int col = tid + k*BS;
        if (col < n) {
            x[col] = has_scale ? __fmaf_rn(scale, r[k], bias) : r[k];
        }
    }
}

template <int BS>
static __global__ void __launch_bounds__(BS) m1hc_mix_mb_kernel(
        const float * up, const float * hn, float * x, const int n, const float scale, const float bias,
        const int has_scale) {
    ggml_cuda_pdl_lc();
    const int col = blockIdx.x*BS + threadIdx.x;
    ggml_cuda_pdl_sync();
    if (col >= n) {
        return;
    }
    float r = 0.0f;
#pragma unroll
    for (int s = 0; s < M1HC_HC; ++s) {
        const float m = __fmul_rn(1.0f / (1.0f + expf(-up[s*n + col])), hn[s*n + col]);
        r = s == 0 ? m : __fadd_rn(r, m);
    }
    x[col] = has_scale ? __fmaf_rn(scale, r, bias) : r;
}

template <int BS, bool silu, bool scale2>
static __global__ void __launch_bounds__(BS) m1hc_act_kernel(
        const float * x, float * y, const int n, const float s1, const float b1, const float s2, const float b2,
        block_q8_1 * yq, const int npad) {
    ggml_cuda_pdl_lc();
    const int tid = threadIdx.x;
    float r[M1HC_NC];
    ggml_cuda_pdl_sync();
#pragma unroll
    for (int k = 0; k < M1HC_NC; ++k) {
        const int col = tid + k*BS;
        r[k] = 0.0f;
        if (col < n) {
            const float v = __fmaf_rn(s1, x[col], b1);
            float a = silu ? v / (1.0f + expf(-v)) : 1.0f / (1.0f + expf(-v));
            if constexpr (scale2) {
                a = __fmaf_rn(s2, a, b2);
            }
            r[k] = a;
        }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < M1HC_NC; ++k) {
        const int col = tid + k*BS;
        if (col < n) {
            y[col] = r[k];
        }
        if (yq != nullptr && col < npad) {
            m1hc_q8_1(yq, col/QK8_1, r[k]);
        }
    }
}

static bool m1hc_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW ||
           t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
}

static const ggml_tensor * m1hc_base(const ggml_tensor * t) {
    return t->view_src != nullptr ? t->view_src : t;
}

static bool m1hc_f32c(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
}

static int m1hc_next(const ggml_cgraph * g, int j, int lim) {
    for (++j; j < lim; ++j) {
        if (!m1hc_noop(g->nodes[j])) {
            return j;
        }
    }
    return -1;
}

static bool m1hc_legal(const ggml_cgraph * g, const int * pat, int np, const int * outs, int nout) {
    int     idx[64];
    ggml_op ops[64];
    int     n = 0;
    for (int j = pat[0]; j <= pat[np - 1]; ++j) {
        const ggml_tensor * t = g->nodes[j];
        bool in = std::find(pat, pat + np, j) != pat + np;
        for (int k = 0; k < n && !in && t->view_src != nullptr; ++k) {
            in = g->nodes[idx[k]] == t->view_src;
        }
        if (!in) {
            continue;
        }
        if (n == 64) {
            return false;
        }
        idx[n] = j;
        ops[n] = t->op;
        ++n;
    }
    return ggml_can_fuse_subgraph_ext(g, idx, n, ops, outs, nout);
}

struct m1hc_stats {
    int post_inj = 0, post_norm = 0, post = 0, mix = 0, act = 0, q81 = 0, mb = 0, mb_bar = 0;
    int last      = -1;
    int printed   = 0;
};

static void m1hc_count(m1hc_stats & st, int node_idx, int * which) {
    if (node_idx <= st.last && st.printed < 2 && st.post_inj + st.post_norm + st.post + st.mix + st.act > 0) {
        fprintf(stderr, "mach1 hc: post+norm+gate %d, post+norm %d, post %d, mix %d, act %d, q8_1 inputs %d, multi-block %d (%d behind a load barrier) per graph\n",
                st.post_inj, st.post_norm, st.post, st.mix, st.act, st.q81, st.mb, st.mb_bar);
        st.printed++;
    }
    if (node_idx <= st.last) {
        st.post_inj = st.post_norm = st.post = st.mix = st.act = st.q81 = st.mb = st.mb_bar = 0;
    }
    st.last = node_idx;
    (*which)++;
}

#define M1HC_Q81_UP  (32u*1024u)
#define M1HC_Q81_CAP (36u*1024u)
#define M1HC_BUF_CAP (M1HC_Q81_CAP + 256u)

static char * m1hc_buf(ggml_backend_cuda_context & ctx) {
    static std::mutex mtx;
    static std::unordered_map<const ggml_backend_cuda_context *, char *> bufs;
    std::lock_guard<std::mutex> lk(mtx);
    char *& b = bufs[&ctx];
    if (b == nullptr) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(ctx.stream(), &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) {
            (void) cudaGetLastError();
            return nullptr;
        }
        CUDA_CHECK(cudaMalloc((void **) &b, M1HC_BUF_CAP));
        CUDA_CHECK(cudaMemset(b + M1HC_Q81_CAP, 0, M1HC_BUF_CAP - M1HC_Q81_CAP));
    }
    return b;
}

static bool m1hc_overlap(const void * a, size_t na, const void * b, size_t nb) {
    const char * pa = (const char *) a;
    const char * pb = (const char *) b;
    return pa < pb + nb && pb < pa + na;
}

static bool m1hc_post_mb_ok(const float * oa, const float * ob, const float * gv, const float * h, const float * w,
                            const float * hp, const float * hn, int64_t n) {
    const size_t row = n*sizeof(float), all = M1HC_HC*row;
    for (const float * o : { hp, hn }) {
        if (o == nullptr) {
            continue;
        }
        if (m1hc_overlap(o, all, oa, row) || (ob != nullptr && m1hc_overlap(o, all, ob, row)) ||
            m1hc_overlap(o, all, gv, M1HC_HC*sizeof(float)) || (o != h && m1hc_overlap(o, all, h, all)) ||
            (w != nullptr && o != w && m1hc_overlap(o, all, w, all))) {
            return false;
        }
    }
    return true;
}

static bool m1hc_mix_mb_ok(const float * up, const float * hn, const float * x, int64_t n) {
    const size_t row = n*sizeof(float), all = M1HC_HC*row;
    for (const float * in : { up, hn }) {
        if (m1hc_overlap(x, row, in, all) && ((const char *) x - (const char *) in) % (ptrdiff_t) row != 0) {
            return false;
        }
    }
    return true;
}

static bool m1hc_mmvq_ok(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_tensor * x_base,
                         int64_t nx, size_t nbytes) {
    if (!m1hc_q81_on() || mm->op != GGML_OP_MUL_MAT || !m1hc_f32c(mm) || ggml_nrows(mm) != 1 ||
        ggml_get_op_params_i32(mm, 1) != 0) {
        return false;
    }
    const ggml_tensor * w = mm->src[0];
    const ggml_tensor * x = mm->src[1];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    return ggml_is_quantized(w->type) && ggml_cuda_should_use_mmvq(w->type, cc, 1) &&
           ggml_cuda_info().devices[ctx.device].warp_size == QK8_1 && w->ne[0] == nx && mm->ne[0] == w->ne[1] &&
           w->ne[2] == 1 && w->ne[3] == 1 && w->nb[0] == ggml_type_size(w->type) && w->buffer != nullptr &&
           ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
           m1hc_f32c(x) && m1hc_base(x) == x_base && ggml_nelements(x) == nx && ggml_nrows(x) == 1 &&
           (size_t) GGML_PAD(nx, MATRIX_ROW_PADDING)/QK8_1*sizeof(block_q8_1) <= nbytes;
}

static int m1hc_mmvf_bs(int64_t ncols) {
    int64_t best = 32;
    int64_t niter_best = (ncols + 2*32 - 1) / (2*32);
    for (int64_t bs = 64; bs <= 256; bs += 32) {
        const int64_t niter = (ncols + 2*bs - 1) / (2*bs);
        if (niter < niter_best) {
            niter_best = niter;
            best       = bs;
        }
    }
    return (int) best;
}

static int m1hc_try_post(ggml_backend_cuda_context & ctx, const ggml_cgraph * g, int i, int lim, m1hc_stats & st) {
    int pat[12];
    int np = 0;
    const ggml_tensor * add0 = nullptr;
    int j = i;
    if (g->nodes[i]->op == GGML_OP_ADD) {
        add0 = g->nodes[i];
        if (!m1hc_f32c(add0) || !m1hc_f32c(add0->src[0]) || !m1hc_f32c(add0->src[1]) ||
            !ggml_are_same_shape(add0, add0->src[0]) || !ggml_are_same_shape(add0, add0->src[1])) {
            return 0;
        }
        pat[np++] = i;
        if ((j = m1hc_next(g, j, lim)) < 0) {
            return 0;
        }
    }
    const ggml_tensor * rep = g->nodes[j];
    if (rep->op != GGML_OP_REPEAT || !m1hc_f32c(rep) || rep->ne[1] != M1HC_HC || rep->ne[2] != 1 ||
        rep->ne[3] != 1) {
        return 0;
    }
    const int64_t       n  = rep->ne[0];
    const ggml_tensor * ro = rep->src[0];
    if (!m1hc_f32c(ro) || ggml_nelements(ro) != n || (add0 != nullptr && m1hc_base(ro) != add0)) {
        return 0;
    }
    pat[np++] = j;
    if ((j = m1hc_next(g, j, lim)) < 0) {
        return 0;
    }
    const ggml_tensor * mul = g->nodes[j];
    if (mul->op != GGML_OP_MUL || !m1hc_f32c(mul) || !ggml_are_same_shape(mul, rep) ||
        (mul->src[0] != rep && mul->src[1] != rep) || mul->src[0] == mul->src[1]) {
        return 0;
    }
    const ggml_tensor * gv = mul->src[0] == rep ? mul->src[1] : mul->src[0];
    if (!m1hc_f32c(gv) || gv->ne[0] != 1 || gv->ne[1] != M1HC_HC || ggml_nelements(gv) != M1HC_HC) {
        return 0;
    }
    pat[np++] = j;
    if ((j = m1hc_next(g, j, lim)) < 0) {
        return 0;
    }
    const ggml_tensor * add = g->nodes[j];
    if (add->op != GGML_OP_ADD || !m1hc_f32c(add) || ggml_nelements(add) != n*M1HC_HC) {
        return 0;
    }
    const int s_mul = m1hc_base(add->src[0]) == mul ? 0 : m1hc_base(add->src[1]) == mul ? 1 : -1;
    if (s_mul < 0) {
        return 0;
    }
    const ggml_tensor * h = add->src[1 - s_mul];
    if (!m1hc_f32c(add->src[s_mul]) || !m1hc_f32c(h) || !ggml_are_same_shape(h, add) || m1hc_base(h) == mul) {
        return 0;
    }
    pat[np++] = j;
    const int j_add = j;

    const ggml_tensor * rms = nullptr;
    const ggml_tensor * hn  = nullptr;
    const ggml_tensor * w   = nullptr;
    int j_hn = -1;
    const int jr = m1hc_next(g, j, lim);
    const int jm = jr < 0 ? -1 : m1hc_next(g, jr, lim);
    if (jm >= 0 && g->nodes[jr]->op == GGML_OP_RMS_NORM && g->nodes[jm]->op == GGML_OP_MUL) {
        rms = g->nodes[jr];
        hn  = g->nodes[jm];
        w   = hn->src[0] == rms ? hn->src[1] : hn->src[1] == rms ? hn->src[0] : nullptr;
        if (!m1hc_f32c(rms) || m1hc_base(rms->src[0]) != add || !m1hc_f32c(rms->src[0]) ||
            rms->src[0]->ne[0] != n || !ggml_are_same_shape(rms, rep) || !m1hc_f32c(hn) ||
            !ggml_are_same_shape(hn, rms) || w == nullptr || !m1hc_f32c(w) || !ggml_are_same_shape(w, rms) ||
            m1hc_base(w) == rms) {
            rms = nullptr;
        }
    }
    if (rms != nullptr) {
        pat[np++] = jr;
        pat[np++] = jm;
        j_hn = jm;
    }
    const int bs = n < 1024 ? 256 : 1024;
    if (n > (int64_t) bs*M1HC_NC) {
        return 0;
    }
    int outs[6] = { j_add, j_hn };
    int nout = rms != nullptr ? 2 : 1;
    auto add_row_view = [&](const ggml_tensor * row, int j_end) {
        for (int k = jm + 1; k < j_end; ++k) {
            if (g->nodes[k] == row && std::find(outs, outs + nout, k) == outs + nout) {
                outs[nout++] = k;
            }
        }
    };

    m1hc_inj_args ia = {};
    if (rms != nullptr && bs == M1HC_HC*256) {
        const int ji  = m1hc_next(g, jm, lim);
        const int js1 = ji  < 0 ? -1 : m1hc_next(g, ji, lim);
        const int jsg = js1 < 0 ? -1 : m1hc_next(g, js1, lim);
        const int js2 = jsg < 0 ? -1 : m1hc_next(g, jsg, lim);
        if (js2 >= 0) {
            const ggml_tensor * mm  = g->nodes[ji];
            const ggml_tensor * s1  = g->nodes[js1];
            const ggml_tensor * sg  = g->nodes[jsg];
            const ggml_tensor * s2  = g->nodes[js2];
            const ggml_tensor * wi  = mm->src[0];
            const ggml_tensor * hn2 = mm->src[1];
            if (mm->op == GGML_OP_MUL_MAT && wi->type == GGML_TYPE_BF16 && ggml_is_contiguous(wi) &&
                wi->ne[0] == n*M1HC_HC && wi->ne[1] == M1HC_HC && wi->ne[2] == 1 && wi->ne[3] == 1 &&
                m1hc_f32c(hn2) && m1hc_base(hn2) == hn && hn2->ne[0] == n*M1HC_HC && ggml_nrows(hn2) == 1 &&
                m1hc_f32c(mm) && ggml_nelements(mm) == M1HC_HC && ggml_cuda_info().devices[ctx.device].warp_size == 32 &&
                GGML_CUDA_CC_IS_NVIDIA(ggml_cuda_info().devices[ctx.device].cc) && m1hc_mmvf_bs(n*M1HC_HC) == 256 &&
                s1->op == GGML_OP_SCALE && s1->src[0] == mm && m1hc_f32c(s1) &&
                sg->op == GGML_OP_UNARY && ggml_get_unary_op(sg) == GGML_UNARY_OP_SIGMOID && sg->src[0] == s1 &&
                m1hc_f32c(sg) && s2->op == GGML_OP_SCALE && s2->src[0] == sg && m1hc_f32c(s2)) {
                const int np0 = np, nout0 = nout;
                pat[np++] = ji;
                pat[np++] = js1;
                pat[np++] = jsg;
                pat[np++] = js2;
                outs[nout++] = js2;
                add_row_view(hn2, ji);
                if (m1hc_legal(g, pat, np, outs, nout)) {
                    ia.w   = (const nv_bfloat16 *) wi->data;
                    ia.out = (float *) s2->data;
                    memcpy(&ia.s1, (const float *) s1->op_params + 0, sizeof(float));
                    memcpy(&ia.b1, (const float *) s1->op_params + 1, sizeof(float));
                    memcpy(&ia.s2, (const float *) s2->op_params + 0, sizeof(float));
                    memcpy(&ia.b2, (const float *) s2->op_params + 1, sizeof(float));
                } else {
                    np   = np0;
                    nout = nout0;
                }
            }
        }
    }

    const ggml_tensor * mmd = nullptr;
    block_q8_1 *        yq  = nullptr;
    if (rms != nullptr && n % QK8_1 == 0 && (n*M1HC_HC) % MATRIX_ROW_PADDING == 0) {
        const int jd = m1hc_next(g, pat[np - 1], lim);
        if (jd >= 0 && m1hc_mmvq_ok(ctx, g->nodes[jd], hn, n*M1HC_HC, M1HC_Q81_UP)) {
            const int np0 = np, nout0 = nout;
            pat[np++] = jd;
            outs[nout++] = jd;
            add_row_view(g->nodes[jd]->src[1], jd);
            char * qb = m1hc_legal(g, pat, np, outs, nout) ? m1hc_buf(ctx) : nullptr;
            if (qb != nullptr) {
                mmd = g->nodes[jd];
                yq  = (block_q8_1 *) qb;
            } else {
                np   = np0;
                nout = nout0;
            }
        }
    }
    if (ia.out == nullptr && mmd == nullptr && !m1hc_legal(g, pat, np, outs, nout)) {
        return 0;
    }
    float eps = 0.0f;
    if (rms != nullptr) {
        memcpy(&eps, rms->op_params, sizeof(float));
    }
    const float * oa = (const float *) (add0 != nullptr ? add0->src[0]->data : ro->data);
    const float * ob = add0 != nullptr ? (const float *) add0->src[1]->data : nullptr;
    float * hn_d = rms != nullptr ? (float *) hn->data : nullptr;
    const float * w_d = rms != nullptr ? (const float *) w->data : nullptr;
    const float * gv_d = (const float *) gv->data;
    const float * h_d  = (const float *) h->data;
    float *       hp_d = (float *) add->data;
    unsigned int * done = nullptr;
    unsigned int * bar  = nullptr;
    bool mb = false;
    bool mb_free = false;
    if (bs == 1024 && m1hc_mb_on()) {
        char * b = m1hc_buf(ctx);
        done = ia.out != nullptr && b != nullptr &&
            !m1hc_overlap(ia.out, M1HC_HC*sizeof(float), hn_d, M1HC_HC*n*sizeof(float)) ?
            (unsigned int *) (b + M1HC_Q81_CAP) : nullptr;
        mb_free = m1hc_post_mb_ok(oa, ob, gv_d, h_d, w_d, hp_d, hn_d, n);
        bar = !mb_free && b != nullptr ? (unsigned int *) (b + M1HC_Q81_CAP + 64) : nullptr;
        mb = (ia.out == nullptr || done != nullptr) && (mb_free || bar != nullptr);
    }
    if (mb) {
        const ggml_cuda_kernel_launch_params lpm(dim3(M1HC_HC), dim3(1024), 0, ctx.stream());
        auto kern = ia.out != nullptr ? (mb_free ? m1hc_post_norm_mb_kernel<true, false>  : m1hc_post_norm_mb_kernel<true, true>)
                                      : (mb_free ? m1hc_post_norm_mb_kernel<false, false> : m1hc_post_norm_mb_kernel<false, true>);
        ggml_cuda_kernel_launch(kern, lpm, oa, ob, gv_d, h_d, w_d, hp_d, hn_d, (int) n, eps, ia, yq, done, bar);
    } else if (ia.out != nullptr) {
        const ggml_cuda_kernel_launch_params lp(dim3(1), dim3(bs), 0, ctx.stream());
        ggml_cuda_kernel_launch(m1hc_post_norm_kernel<1024, true>, lp, oa, ob, (const float *) gv->data,
                                (const float *) h->data, w_d, (float *) add->data, hn_d, (int) n, eps, ia, yq);
    } else if (bs == 1024) {
        const ggml_cuda_kernel_launch_params lp(dim3(1), dim3(bs), 0, ctx.stream());
        ggml_cuda_kernel_launch(m1hc_post_norm_kernel<1024, false>, lp, oa, ob, (const float *) gv->data,
                                (const float *) h->data, w_d, (float *) add->data, hn_d, (int) n, eps, ia, yq);
    } else {
        const ggml_cuda_kernel_launch_params lp(dim3(1), dim3(bs), 0, ctx.stream());
        ggml_cuda_kernel_launch(m1hc_post_norm_kernel<256, false>, lp, oa, ob, (const float *) gv->data,
                                (const float *) h->data, w_d, (float *) add->data, hn_d, (int) n, eps, ia, yq);
    }
    if (mmd != nullptr) {
        ggml_cuda_mul_mat_vec_q_q8_1(ctx, mmd->src[0], yq, const_cast<ggml_tensor *>(mmd));
    }
    m1hc_count(st, i, rms == nullptr ? &st.post : ia.out != nullptr ? &st.post_inj : &st.post_norm);
    st.q81    += mmd != nullptr;
    st.mb     += mb;
    st.mb_bar += mb && !mb_free;
    return pat[np - 1] - i;
}

static int m1hc_try_mix(ggml_backend_cuda_context & ctx, const ggml_cgraph * g, int i, int lim, m1hc_stats & st) {
    const ggml_tensor * sg = g->nodes[i];
    if (!m1hc_f32c(sg) || !m1hc_f32c(sg->src[0]) || ggml_nrows(sg) != 1) {
        return 0;
    }
    int pat[M1HC_HC + 3];
    int np = 0;
    pat[np++] = i;
    int j = m1hc_next(g, i, lim);
    if (j < 0) {
        return 0;
    }
    const ggml_tensor * mul = g->nodes[j];
    if (mul->op != GGML_OP_MUL || !m1hc_f32c(mul) || mul->ne[1] != M1HC_HC || mul->ne[2] != 1 || mul->ne[3] != 1 ||
        ggml_nelements(mul) != ggml_nelements(sg)) {
        return 0;
    }
    const int s_sg = m1hc_base(mul->src[0]) == sg ? 0 : m1hc_base(mul->src[1]) == sg ? 1 : -1;
    if (s_sg < 0) {
        return 0;
    }
    const ggml_tensor * hn = mul->src[1 - s_sg];
    if (!m1hc_f32c(mul->src[s_sg]) || !m1hc_f32c(hn) || !ggml_are_same_shape(hn, mul) || m1hc_base(hn) == sg) {
        return 0;
    }
    pat[np++] = j;
    const int64_t n = mul->ne[0];
    const ggml_tensor * prev = nullptr;
    for (int s = 1; s < M1HC_HC; ++s) {
        if ((j = m1hc_next(g, j, lim)) < 0) {
            return 0;
        }
        const ggml_tensor * add = g->nodes[j];
        if (add->op != GGML_OP_ADD || !m1hc_f32c(add) || ggml_nelements(add) != n) {
            return 0;
        }
        const ggml_tensor * a = add->src[0];
        const ggml_tensor * b = add->src[1];
        auto is_view = [&](const ggml_tensor * v, int k) {
            return v->view_src == mul && v->view_offs == (size_t) k*mul->nb[1] && ggml_nelements(v) == n &&
                   ggml_is_contiguous(v);
        };
        if (s == 1 ? !(is_view(a, 0) && is_view(b, 1)) : !(a == prev && is_view(b, s))) {
            return 0;
        }
        prev = add;
        pat[np++] = j;
    }
    float scale = 1.0f;
    float bias  = 0.0f;
    int has_scale = 0;
    const int js = m1hc_next(g, j, lim);
    if (js >= 0 && g->nodes[js]->op == GGML_OP_SCALE && g->nodes[js]->src[0] == prev && m1hc_f32c(g->nodes[js])) {
        memcpy(&scale, (const float *) g->nodes[js]->op_params + 0, sizeof(float));
        memcpy(&bias,  (const float *) g->nodes[js]->op_params + 1, sizeof(float));
        has_scale = 1;
        pat[np++] = js;
    }
    if (n > 1024*M1HC_NC) {
        return 0;
    }
    const int out = pat[np - 1];
    if (!m1hc_legal(g, pat, np, &out, 1)) {
        return 0;
    }
    const float * up_d = (const float *) sg->src[0]->data;
    const float * hn_d = (const float *) hn->data;
    float *       x_d  = (float *) g->nodes[out]->data;
    const bool mb = m1hc_mb_on() && m1hc_mix_mb_ok(up_d, hn_d, x_d, n);
    if (mb) {
        const ggml_cuda_kernel_launch_params lp(dim3((n + 255)/256), dim3(256), 0, ctx.stream());
        ggml_cuda_kernel_launch(m1hc_mix_mb_kernel<256>, lp, up_d, hn_d, x_d, (int) n, scale, bias, has_scale);
    } else {
        const ggml_cuda_kernel_launch_params lp(dim3(1), dim3(1024), 0, ctx.stream());
        ggml_cuda_kernel_launch(m1hc_mix_kernel<1024>, lp, up_d, hn_d, x_d, (int) n, scale, bias, has_scale);
    }
    m1hc_count(st, i, &st.mix);
    st.mb += mb;
    return out - i;
}

static int m1hc_try_act(ggml_backend_cuda_context & ctx, const ggml_cgraph * g, int i, int lim, m1hc_stats & st) {
    const ggml_tensor * sc = g->nodes[i];
    if (!m1hc_f32c(sc) || !m1hc_f32c(sc->src[0]) || ggml_nrows(sc) != 1) {
        return 0;
    }
    int pat[4];
    int np = 0;
    pat[np++] = i;
    const int j = m1hc_next(g, i, lim);
    if (j < 0) {
        return 0;
    }
    const ggml_tensor * u = g->nodes[j];
    if (u->op != GGML_OP_UNARY || u->src[0] != sc || !m1hc_f32c(u) ||
        (ggml_get_unary_op(u) != GGML_UNARY_OP_SILU && ggml_get_unary_op(u) != GGML_UNARY_OP_SIGMOID)) {
        return 0;
    }
    const bool silu = ggml_get_unary_op(u) == GGML_UNARY_OP_SILU;
    pat[np++] = j;
    const int j2 = m1hc_next(g, j, lim);
    const bool scale2 = j2 >= 0 && g->nodes[j2]->op == GGML_OP_SCALE && g->nodes[j2]->src[0] == u &&
                        m1hc_f32c(g->nodes[j2]);
    if (scale2) {
        pat[np++] = j2;
    }
    const int64_t n = ggml_nelements(sc);
    if (n > 256*M1HC_NC) {
        return 0;
    }
    int outs[2] = { pat[np - 1], -1 };
    if (!m1hc_legal(g, pat, np, outs, 1)) {
        return 0;
    }
    const int out = outs[0];
    const ggml_tensor * mmu  = nullptr;
    block_q8_1 *        yq   = nullptr;
    const int           npad = (int) GGML_PAD(n, MATRIX_ROW_PADDING);
    const int           ju   = m1hc_next(g, out, lim);
    if (ju >= 0 && npad <= 256*M1HC_NC &&
        m1hc_mmvq_ok(ctx, g->nodes[ju], g->nodes[out], n, M1HC_Q81_CAP - M1HC_Q81_UP)) {
        pat[np++] = ju;
        outs[1]   = ju;
        char * qb = m1hc_legal(g, pat, np, outs, 2) ? m1hc_buf(ctx) : nullptr;
        if (qb != nullptr) {
            mmu = g->nodes[ju];
            yq  = (block_q8_1 *) (qb + M1HC_Q81_UP);
        } else {
            np--;
        }
    }
    float s1, b1, s2 = 1.0f, b2 = 0.0f;
    memcpy(&s1, (const float *) sc->op_params + 0, sizeof(float));
    memcpy(&b1, (const float *) sc->op_params + 1, sizeof(float));
    if (scale2) {
        memcpy(&s2, (const float *) g->nodes[j2]->op_params + 0, sizeof(float));
        memcpy(&b2, (const float *) g->nodes[j2]->op_params + 1, sizeof(float));
    }
    const float * x = (const float *) sc->src[0]->data;
    float *       y = (float *) g->nodes[out]->data;
    const ggml_cuda_kernel_launch_params lp(dim3(1), dim3(256), 0, ctx.stream());
    if (silu && scale2) {
        ggml_cuda_kernel_launch(m1hc_act_kernel<256, true, true>, lp, x, y, (int) n, s1, b1, s2, b2, yq, npad);
    } else if (silu) {
        ggml_cuda_kernel_launch(m1hc_act_kernel<256, true, false>, lp, x, y, (int) n, s1, b1, s2, b2, yq, npad);
    } else if (scale2) {
        ggml_cuda_kernel_launch(m1hc_act_kernel<256, false, true>, lp, x, y, (int) n, s1, b1, s2, b2, yq, npad);
    } else {
        ggml_cuda_kernel_launch(m1hc_act_kernel<256, false, false>, lp, x, y, (int) n, s1, b1, s2, b2, yq, npad);
    }
    if (mmu != nullptr) {
        ggml_cuda_mul_mat_vec_q_q8_1(ctx, mmu->src[0], yq, const_cast<ggml_tensor *>(mmu));
    }
    m1hc_count(st, i, &st.act);
    st.q81 += mmu != nullptr;
    return pat[np - 1] - i;
}

int ggml_cuda_mach1_hc_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx, int max_skip) {
    if (!m1hc_on() || max_skip <= 0) {
        return 0;
    }
    static m1hc_stats st;
    const ggml_tensor * t   = cgraph->nodes[node_idx];
    const int           lim = std::min(cgraph->n_nodes, node_idx + max_skip + 1);
    switch (t->op) {
        case GGML_OP_ADD:
        case GGML_OP_REPEAT:
            return m1hc_try_post(ctx, cgraph, node_idx, lim, st);
        case GGML_OP_UNARY:
            return ggml_get_unary_op(t) == GGML_UNARY_OP_SIGMOID ? m1hc_try_mix(ctx, cgraph, node_idx, lim, st) : 0;
        case GGML_OP_SCALE:
            return m1hc_try_act(ctx, cgraph, node_idx, lim, st);
        default:
            return 0;
    }
}

#else

int ggml_cuda_mach1_hc_fuse(ggml_backend_cuda_context &, const ggml_cgraph *, int, int) {
    return 0;
}

#endif
