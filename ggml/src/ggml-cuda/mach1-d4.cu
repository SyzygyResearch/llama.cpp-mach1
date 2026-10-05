
#ifndef MACH1_D4_EMU
#include "mach1-d4.cuh"
#include "cp-async.cuh"
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <mma.h>
#define MACH1_D4_HAVE_WMMA 1
#endif
#define MACH1_D4_DYN_SMEM(T, name) extern __shared__ T name[]
#define MACH1_D4_LDG(p) __ldg(p)
#else
#define MACH1_D4_LDG(p) (*(p))
#endif

#if defined(MACH1_D4_EMU) || (!defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA))

#include "mach1-pdl.cuh"

#define MACH1_D4_TT 8
#define MACH1_D4_WG 256

__constant__ int8_t mach1_d4_h12[144] = {
     1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
    -1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
     1,  1,  1, -1,  1,  1, -1, -1, -1, -1,  1,  1,
     1, -1, -1, -1,  1, -1, -1,  1, -1,  1,  1, -1,
     1,  1,  1,  1,  1, -1,  1,  1, -1, -1, -1, -1,
     1, -1,  1, -1, -1, -1,  1, -1, -1,  1, -1,  1,
     1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1,
     1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,
     1,  1, -1, -1, -1, -1,  1,  1,  1, -1,  1,  1,
     1, -1, -1,  1, -1,  1,  1, -1, -1, -1,  1, -1,
     1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1,
     1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1
};
__constant__ int8_t mach1_d4_h20[400] = {
     1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
    -1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
     1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1,
     1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1,
     1,  1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1, -1, -1,  1,  1, -1, -1,
     1, -1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1, -1,  1,
     1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1,  1,  1,
     1, -1,  1, -1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,
     1,  1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1,
     1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1,  1, -1,  1,
     1,  1, -1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1,
     1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1,  1,
     1,  1, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,
     1, -1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1,
     1,  1,  1,  1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1,  1,  1,
     1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1,  1, -1,
     1,  1, -1, -1,  1,  1, -1, -1, -1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1,
     1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1,
     1,  1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1, -1,
     1, -1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1, -1
};

template <int R, int KMAX>
static __device__ __forceinline__ void mach1_d4_fwht_radix(float * sh, const int d, const int M, const int tid, const int wg) {
    const int8_t * H = R == 12 ? mach1_d4_h12 : mach1_d4_h20;
    const float rs = sqrtf((float) (d / M));
    if (d <= KMAX*wg) {
        const int lm = 31 - __clz(M);
        float o[KMAX];
#pragma unroll
        for (int k = 0; k < KMAX; ++k) {
            if (k*wg >= d) {
                break;
            }
            const int i = tid + k*wg;
            if (i < d) {
                const int a = i >> lm;
                const int b = i & (M - 1);
                float acc = 0.0f;
#pragma unroll
                for (int c = 0; c < R; ++c) {
                    acc += (float) H[a*R + c]*sh[c*M + b];
                }
                o[k] = acc/rs;
            }
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < KMAX; ++k) {
            if (k*wg >= d) {
                break;
            }
            const int i = tid + k*wg;
            if (i < d) {
                sh[i] = o[k];
            }
        }
    } else {
        float t[R];
        for (int b = tid; b < M; b += wg) {
            for (int a = 0; a < R; ++a) {
                float acc = 0.0f;
#pragma unroll
                for (int c = 0; c < R; ++c) {
                    acc += (float) H[a*R + c]*sh[c*M + b];
                }
                t[a] = acc/rs;
            }
            for (int a = 0; a < R; ++a) {
                sh[a*M + b] = t[a];
            }
        }
    }
}

template <int R, int KMAX, bool OUT>
static __device__ __forceinline__ void mach1_d4_fwht_radix_st(const float * sh, const int d, const int M, const int tid,
                                                              const int wg, float * __restrict__ y,
                                                              const half * __restrict__ se) {
    const int8_t * H = R == 12 ? mach1_d4_h12 : mach1_d4_h20;
    const float rs = sqrtf((float) (d / M));
    const int lm = 31 - __clz(M);
#pragma unroll
    for (int k = 0; k < KMAX; ++k) {
        if (k*wg >= d) {
            break;
        }
        const int i = tid + k*wg;
        if (i < d) {
            const int a = i >> lm;
            const int b = i & (M - 1);
            const float s = OUT ? __half2float(se[i]) : 0.0f;
            float acc = 0.0f;
#pragma unroll
            for (int c = 0; c < R; ++c) {
                acc += (float) H[a*R + c]*sh[c*M + b];
            }
            const float o = acc/rs;
            y[i] = OUT ? o*s : o;
        }
    }
}

template <int R, bool OUT>
static __device__ __forceinline__ void mach1_d4_radix_st_sl(const float * sh, const int d, const int M, const int P,
                                                            const int p0, float * __restrict__ y,
                                                            const half * __restrict__ se) {
    const int8_t * H = R == 12 ? mach1_d4_h12 : mach1_d4_h20;
    const float rs = sqrtf((float) (d / M));
    const int lp = 31 - __clz(P);
    for (int i = threadIdx.x; i < R*P; i += blockDim.x) {
        const int a = i >> lp;
        const int b = p0 + (i & (P - 1));
        const int j = a*M + b;
        const float s = OUT ? __half2float(se[j]) : 0.0f;
        float acc = 0.0f;
#pragma unroll
        for (int c = 0; c < R; ++c) {
            acc += (float) H[a*R + c]*sh[c*M + b];
        }
        const float o = acc/rs;
        y[j] = OUT ? o*s : o;
    }
}

template <int KMAX = 32>
static __device__ void mach1_d4_fwht(float * sh, const int d, const int tid, const int wg) {
    int radix = 1;
    if ((d & (d - 1)) != 0) {
        const int m12 = d/12;
        radix = (d % 12 == 0 && (m12 & (m12 - 1)) == 0) ? 12 : 20;
    }
    const int M = d/radix;
    for (int span = 1, ls = 0; span < M; span <<= 1, ++ls) {
        for (int b = tid; b < d/2; b += wg) {
            const int base = ((b >> ls) << (ls + 1)) + (b & (span - 1));
            const float a0 = sh[base];
            const float a1 = sh[base + span];
            sh[base]        = a0 + a1;
            sh[base + span] = a0 - a1;
        }
        __syncthreads();
    }
    const float scm = sqrtf((float) M);
    for (int i = tid; i < d; i += wg) {
        sh[i] = sh[i]/scm;
    }
    __syncthreads();
    if (radix > 1) {
        if (radix == 12) {
            mach1_d4_fwht_radix<12, KMAX>(sh, d, M, tid, wg);
        } else {
            mach1_d4_fwht_radix<20, KMAX>(sh, d, M, tid, wg);
        }
        __syncthreads();
    }
}

template <int E, bool TL = false>
static __device__ __forceinline__ void mach1_d4_warp_fwht(float * v, float * sh, const int d, const int M,
                                                          const int c, const int lane, const bool own,
                                                          const bool radix = true) {
    if (own && TL) {
#pragma unroll
        for (int lm = 1; lm < 32; lm <<= 1) {
            const bool upper = (lane & lm) != 0;
#pragma unroll
            for (int e = 0; e < E; ++e) {
                const float o = __shfl_xor_sync(0xFFFFFFFFu, v[e], lm);
                v[e] = upper ? o - v[e] : v[e] + o;
            }
        }
#pragma unroll
        for (int span = 1; span < E; span <<= 1) {
#pragma unroll
            for (int e = 0; e < E; ++e) {
                if ((e & span) == 0) {
                    const float a0 = v[e];
                    const float a1 = v[e + span];
                    v[e]        = a0 + a1;
                    v[e + span] = a0 - a1;
                }
            }
        }
    } else if (own) {
#pragma unroll
        for (int span = 1; span < E; span <<= 1) {
#pragma unroll
            for (int e = 0; e < E; ++e) {
                if ((e & span) == 0) {
                    const float a0 = v[e];
                    const float a1 = v[e + span];
                    v[e]        = a0 + a1;
                    v[e + span] = a0 - a1;
                }
            }
        }
        for (int span = E; span < M; span <<= 1) {
            const int  lm    = span / E;
            const bool upper = (lane & lm) != 0;
#pragma unroll
            for (int e = 0; e < E; ++e) {
                const float o = __shfl_xor_sync(0xFFFFFFFFu, v[e], lm);
                v[e] = upper ? o - v[e] : v[e] + o;
            }
        }
    }
    const float scm = sqrtf((float) M);
    if (own) {
#pragma unroll
        for (int e = 0; e < E; ++e) {
            sh[TL ? c*M + e*32 + lane : c*M + lane*E + e] = v[e]/scm;
        }
    }
    __syncthreads();
    if (!radix) {
        return;
    }
    if (d / M == 12) {
        mach1_d4_fwht_radix<12, 8>(sh, d, M, threadIdx.x, 1024);
    } else {
        mach1_d4_fwht_radix<20, 8>(sh, d, M, threadIdx.x, 1024);
    }
    __syncthreads();
}

template <int E, bool OUT, bool TL = false>
static __global__ void __launch_bounds__(1024) mach1_d4_warp_stage_kernel(const float * src, const half * side,
        const int32_t * ids, float * dst, const int d, const int n_used, const int xne1, const int nb0, const int nb1,
        const float * src2 = nullptr, const half * side2 = nullptr, float * dst2 = nullptr) {
    MACH1_D4_DYN_SMEM(float, sh);
    if (blockIdx.y != 0) {
        src = src2; side = side2; dst = dst2;
    }
    mach1_pdl_trigger();
    mach1_pdl_wait();
    const int p    = blockIdx.x;
    const int e    = ids[(p % n_used)*nb0 + (p / n_used)*nb1];
    const int t    = p / n_used;
    const int lane = threadIdx.x & 31;
    const int c    = threadIdx.x >> 5;
    const int M    = d / (d % 12 == 0 && ((d/12) & (d/12 - 1)) == 0 ? 12 : 20);
    const bool own = c < d / M;
    const half  * se = side + (int64_t) e*d;
    const float * xc = src + (int64_t)(OUT ? p : (xne1 == 1 ? t : p))*d;
    float v[E];
#pragma unroll
    for (int k = 0; k < E; ++k) {
        const int i = TL ? c*M + k*32 + lane : c*M + lane*E + k;
        v[k] = own ? (OUT ? xc[i] : __half2float(se[i])*xc[i]) : 0.0f;
    }
    float * y = dst + (int64_t) p*d;
    if (TL && d <= 8*1024) {
        mach1_d4_warp_fwht<E, TL>(v, sh, d, M, c, lane, own, false);
        if (gridDim.z > 1) {
            const int P = M / (int) gridDim.z;
            if (d / M == 12) {
                mach1_d4_radix_st_sl<12, OUT>(sh, d, M, P, blockIdx.z*P, y, se);
            } else {
                mach1_d4_radix_st_sl<20, OUT>(sh, d, M, P, blockIdx.z*P, y, se);
            }
            return;
        }
        if (d / M == 12) {
            mach1_d4_fwht_radix_st<12, 8, OUT>(sh, d, M, threadIdx.x, 1024, y, se);
        } else {
            mach1_d4_fwht_radix_st<20, 8, OUT>(sh, d, M, threadIdx.x, 1024, y, se);
        }
        return;
    }
    mach1_d4_warp_fwht<E, TL>(v, sh, d, M, c, lane, own);
    for (int i = threadIdx.x; i < d; i += 1024) {
        y[i] = OUT ? sh[i]*__half2float(se[i]) : sh[i];
    }
}

template <int R>
static __device__ __forceinline__ float mach1_d4_radix_one(const float * sh, const int d, const int M, const int i) {
    const int8_t * H = R == 12 ? mach1_d4_h12 : mach1_d4_h20;
    const float rs = sqrtf((float) (d / M));
    const int lm = 31 - __clz(M);
    const int a = i >> lm;
    const int b = i & (M - 1);
    float acc = 0.0f;
#pragma unroll
    for (int c = 0; c < R; ++c) {
        acc += (float) H[a*R + c]*sh[c*M + b];
    }
    return acc/rs;
}

template <int R>
static __global__ void __launch_bounds__(1024) mach1_d4_glu_stage_kernel(const float * pg, const float * pu,
        const half * svg, const half * svu, const half * sud, const int32_t * ids, float * u, const int d, const int M,
        const int n_used, const int nb0, const int nb1) {
    MACH1_D4_DYN_SMEM(float, sh);
    mach1_pdl_trigger();
    mach1_pdl_wait();
    const int p    = blockIdx.x;
    const int e    = ids[(p % n_used)*nb0 + (p / n_used)*nb1];
    const int lane = threadIdx.x & 31;
    const int c    = threadIdx.x >> 5;
    const int i    = threadIdx.x;
    const bool own = c < d / M;
    const int64_t off = (int64_t) e*d;
    const float xg = own ? pg[(int64_t) p*d + i] : 0.0f;
    const float xu = own ? pu[(int64_t) p*d + i] : 0.0f;
    const float sg = i < d ? __half2float(svg[off + i]) : 0.0f;
    const float sq = i < d ? __half2float(svu[off + i]) : 0.0f;
    const float sd = own ? __half2float(sud[off + i]) : 0.0f;
    float v[1];

    v[0] = xg;
    mach1_d4_warp_fwht<1, true>(v, sh, d, M, c, lane, own, false);
    float yg = 0.0f;
    if (i < d) {
        const float o = mach1_d4_radix_one<R>(sh, d, M, i);
        yg = o*sg;
    }
    __syncthreads();

    v[0] = xu;
    mach1_d4_warp_fwht<1, true>(v, sh, d, M, c, lane, own, false);
    float yu = 0.0f;
    if (i < d) {
        const float o = mach1_d4_radix_one<R>(sh, d, M, i);
        yu = o*sq;
    }
    __syncthreads();

    const float h = (yg / (1.0f + expf(-yg))) * yu;
    v[0] = own ? sd*h : 0.0f;
    mach1_d4_warp_fwht<1, true>(v, sh, d, M, c, lane, own, false);
    if (i < d) {
        u[(int64_t) p*d + i] = mach1_d4_radix_one<R>(sh, d, M, i);
    }
}

template <int E>
static __global__ void __launch_bounds__(1024) mach1_d4_out_wsum_kernel(const float * src, const half * side,
        const int32_t * ids, const float * w, float * wbuf, float * out, int * counters, const int d,
        const int n_used, const int nb0, const int nb1, const int64_t out_stride) {
    MACH1_D4_DYN_SMEM(float, sh);
    __shared__ int is_last;
    mach1_pdl_trigger();
    mach1_pdl_wait();
    const int p    = blockIdx.x;
    const int t    = p / n_used;
    const int e    = ids[(p % n_used)*nb0 + (p / n_used)*nb1];
    const int lane = threadIdx.x & 31;
    const int c    = threadIdx.x >> 5;
    const int M    = d / (d % 12 == 0 && ((d/12) & (d/12 - 1)) == 0 ? 12 : 20);
    const bool own = c < d / M;
    const int P    = M / (int) gridDim.y;
    const int p0   = blockIdx.y*P;
    const int lp   = 31 - __clz(P);
    const int no   = (d / M)*P;
    const half  * se = side + (int64_t) e*d;
    const float * xc = src + (int64_t) p*d;
    float v[E];
#pragma unroll
    for (int k = 0; k < E; ++k) {
        v[k] = own ? xc[c*M + k*32 + lane] : 0.0f;
    }
    const float wp = w[p];
    const int   j0 = (threadIdx.x >> lp)*M + p0 + (threadIdx.x & (P - 1));
    const float s0 = threadIdx.x < no ? __half2float(se[j0]) : 0.0f;
    mach1_d4_warp_fwht<E, true>(v, sh, d, M, c, lane, own, false);
    float * y = wbuf + (int64_t) p*d;
    if (d / M == 12) {
        for (int i = threadIdx.x; i < no; i += 1024) {
            const int j = (i >> lp)*M + p0 + (i & (P - 1));
            const float ys = mach1_d4_radix_one<12>(sh, d, M, j)*(i == threadIdx.x ? s0 : __half2float(se[j]));
            y[j] = ys*wp;
        }
    } else {
        for (int i = threadIdx.x; i < no; i += 1024) {
            const int j = (i >> lp)*M + p0 + (i & (P - 1));
            const float ys = mach1_d4_radix_one<20>(sh, d, M, j)*(i == threadIdx.x ? s0 : __half2float(se[j]));
            y[j] = ys*wp;
        }
    }
    __threadfence();
    __syncthreads();
    int * cnt = counters + t*(int) gridDim.y + (int) blockIdx.y;
    if (threadIdx.x == 0) {
        is_last = atomicAdd(cnt, 1) == n_used - 1;
    }
    __syncthreads();
    if (!is_last) {
        return;
    }
    if (threadIdx.x == 0) {
        *cnt = 0;
    }
    __threadfence();
    const float * wt = wbuf + (int64_t) t*n_used*d;
    for (int i = threadIdx.x; i < no; i += 1024) {
        const int j = (i >> lp)*M + p0 + (i & (P - 1));
        float ws[13];
#pragma unroll
        for (int s = 0; s < 13; ++s) {
            ws[s] = s < n_used ? __ldcg(&wt[(int64_t) s*d + j]) : 0.0f;
        }
        float acc = ws[0];
#pragma unroll
        for (int s = 1; s < 13; ++s) {
            if (s < n_used) {
                acc += ws[s];
            }
        }
        out[t*out_stride + j] = acc;
    }
}

#ifndef MACH1_D4_EMU
static bool mach1_d4_warp_ok(int d) {
    static const bool on = getenv("GGML_MACH1_D4_WARP") == nullptr || atoi(getenv("GGML_MACH1_D4_WARP")) != 0;
    if (!on || (d & (d - 1)) == 0) {
        return false;
    }
    const int m12 = d/12;
    const int r   = (d % 12 == 0 && (m12 & (m12 - 1)) == 0) ? 12 : 20;
    const int M   = d/r;
    return M >= 32 && M <= 1024 && M*r == d;
}

static int mach1_d4_split(int M, int n_tok) {
    static const int sp = getenv("GGML_MACH1_D4_SPLIT") == nullptr ? 4 : atoi(getenv("GGML_MACH1_D4_SPLIT"));
    static const bool tl = getenv("GGML_MACH1_D4_TL") == nullptr || atoi(getenv("GGML_MACH1_D4_TL")) != 0;
    int S = M >= 128 && tl && n_tok <= 64 ? std::min(sp, 8) : 1;
    while (S > 1 && (M % S != 0 || (S & (S - 1)) != 0)) {
        --S;
    }
    return std::max(S, 1);
}

static bool mach1_d4_warp_stage(bool out, const float * src, const half * side, const int32_t * ids, float * dst,
                                int d, int n_pairs, int n_used, int xne1, int nb0, int nb1, cudaStream_t stream,
                                const float * src2 = nullptr, const half * side2 = nullptr, float * dst2 = nullptr) {
    if (!mach1_d4_warp_ok(d)) {
        return false;
    }
    if (mach1_skip(out ? 16 : 4)) {
        return true;
    }
    const int r = (d % 12 == 0 && ((d/12) & (d/12 - 1)) == 0) ? 12 : 20;
    const int M = d/r;
    const dim3 grid((unsigned) n_pairs, side2 != nullptr ? 2 : 1, (unsigned) mach1_d4_split(M, n_pairs/n_used));
    static const bool tl = getenv("GGML_MACH1_D4_TL") == nullptr || atoi(getenv("GGML_MACH1_D4_TL")) != 0;
#define M1_D4W(E) \
    if (tl) { \
        if (out) { mach1_launch_pdl_raw(mach1_d4_warp_stage_kernel<E, true,  true>, grid, dim3(1024), d*sizeof(float), stream, src, side, ids, dst, d, n_used, xne1, nb0, nb1, src2, side2, dst2); } \
        else     { mach1_launch_pdl_raw(mach1_d4_warp_stage_kernel<E, false, true>, grid, dim3(1024), d*sizeof(float), stream, src, side, ids, dst, d, n_used, xne1, nb0, nb1, src2, side2, dst2); } \
    } else if (out) { mach1_launch_pdl_raw(mach1_d4_warp_stage_kernel<E, true >, grid, dim3(1024), d*sizeof(float), stream, src, side, ids, dst, d, n_used, xne1, nb0, nb1, src2, side2, dst2); } \
    else            { mach1_launch_pdl_raw(mach1_d4_warp_stage_kernel<E, false>, grid, dim3(1024), d*sizeof(float), stream, src, side, ids, dst, d, n_used, xne1, nb0, nb1, src2, side2, dst2); }
    switch (M/32) {
        case 1:  M1_D4W(1);  break;
        case 2:  M1_D4W(2);  break;
        case 4:  M1_D4W(4);  break;
        case 8:  M1_D4W(8);  break;
        case 16: M1_D4W(16); break;
        default: M1_D4W(32); break;
    }
#undef M1_D4W
    return true;
}
#endif

static __device__ __forceinline__ int mach1_d4_id(const int32_t * ids, int p, int n_used, int nb0, int nb1) {
    return ids[(p % n_used)*nb0 + (p / n_used)*nb1];
}

template <int WG = MACH1_D4_WG>
static __global__ void __launch_bounds__(WG) mach1_d4_ustage_kernel(const float * x, const half * su, const int32_t * ids, float * scr_u,
        const int d, const int n_used, const int xne1, const int nb0, const int nb1, float * uscale = nullptr) {
    MACH1_D4_DYN_SMEM(float, sh);
    const int p = blockIdx.x;
    const int e = mach1_d4_id(ids, p, n_used, nb0, nb1);
    const int t = p / n_used;
    const float * xc = x + (int64_t)(xne1 == 1 ? t : p)*d;
    const half  * se = su + (int64_t) e*d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        sh[i] = __half2float(se[i])*xc[i];
    }
    __syncthreads();
    mach1_d4_fwht<WG >= 1024 ? 8 : 32>(sh, d, threadIdx.x, blockDim.x);
    float * u = scr_u + (int64_t) p*d;
    if (uscale != nullptr) {
        __shared__ float wmax[32];
        float mx = 0.0f;
        for (int i = threadIdx.x; i < d; i += blockDim.x) {
            mx = fmaxf(mx, fabsf(sh[i]));
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, off));
        }
        if ((threadIdx.x & 31) == 0) {
            wmax[threadIdx.x >> 5] = mx;
        }
        __syncthreads();
        mx = 0.0f;
        for (int w = 0; w < (int) (blockDim.x >> 5); ++w) {
            mx = fmaxf(mx, wmax[w]);
        }
        int ex = 0;
        if (mx > 0.0f && mx <= 3.4e38f) {
            frexpf(mx, &ex);
        }
        const int   k  = ex - 14;
        const float is = ldexpf(1.0f, -k);
        for (int i = threadIdx.x; i < d; i += blockDim.x) {
            u[i] = sh[i]*is;
        }
        if (threadIdx.x == 0) {
            uscale[p] = ldexpf(1.0f, k);
        }
        return;
    }
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        u[i] = sh[i];
    }
}

static __global__ void mach1_d4_map_kernel(const int32_t * ids, int32_t * map,
        const int E, const int n_used, const int P, const int nb0, const int nb1) {
    MACH1_D4_DYN_SMEM(int, shm);
    const int tid = threadIdx.x;
    const int wg  = blockDim.x;
    int * cnt = shm;
    for (int e = tid; e < E; e += wg) {
        cnt[e] = 0;
    }
    __syncthreads();
    for (int p = tid; p < P; p += wg) {
        atomicAdd(&cnt[mach1_d4_id(ids, p, n_used, nb0, nb1)], 1);
    }
    __syncthreads();
    if (tid >= 32) {
        return;
    }
    int acc = 0;
    for (int e0 = 0; e0 < E; e0 += 32) {
        const int e = e0 + tid;
        const int c = e < E ? cnt[e] : 0;
        int x = c;
#pragma unroll
        for (int s = 1; s < 32; s <<= 1) {
            const int y = __shfl_up_sync(0xFFFFFFFFu, x, s);
            x += tid >= s ? y : 0;
        }
        if (e < E) {
            map[e] = acc + x - c;
            cnt[e] = acc + x - c;
        }
        acc += __shfl_sync(0xFFFFFFFFu, x, 31);
    }
    if (tid == 0) {
        map[E] = acc;
    }
    __syncwarp();
    for (int p0 = 0; p0 < P; p0 += 32) {
        const int      p    = p0 + tid;
        const int      e    = p < P ? mach1_d4_id(ids, p, n_used, nb0, nb1) : -1;
        const unsigned grp  = __match_any_sync(0xFFFFFFFFu, e);
        const int      rank = __popc(grp & ((1u << tid) - 1u));
        const int      base = p < P ? cnt[e] : 0;
        __syncwarp();
        if (p < P) {
            map[E + 1 + base + rank] = p;
            if (rank == 0) {
                cnt[e] = base + __popc(grp);
            }
        }
        __syncwarp();
    }
}

struct mach1_d4_hash {
    int32_t v[15];
};

static __device__ __forceinline__ void mach1_d4_unpack(uint32_t v, float * w) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        w[c] = (float)((int)((v >> (4*c)) & 15u) - 8);
    }
}

static __device__ __forceinline__ void mach1_d4_levels(uint32_t s, uint32_t mult, uint64_t lv, const uint16_t * zr, float * w) {
    if (mult != 0) {
        const uint32_t h = (s*mult) >> 16;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const uint32_t k = (h >> (4*c)) & 15u;
            w[c] = (float)(((int32_t)((uint32_t)(lv >> (4*k)) << 28)) >> 28);
        }
    } else {
        const uint32_t v = zr[s];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            w[c] = (float)((int)((v >> (4*c)) & 15u) - 8);
        }
    }
}

template <bool STAGE, int NWP = 8, bool TAB = false, bool CMP = false, int TTG = MACH1_D4_TT, bool TSTG = false, int RB = 1>
static __global__ void __launch_bounds__(NWP*32) mach1_d4_walk_kernel(
        const uint16_t * trellis, const int32_t * offs, const float * gw, const uint16_t * zt, const float * units,
        const int32_t * ids, const int32_t * map, const float * scr_u, float * scr_p,
        const int n, const int Mb, const int Nb, const int n_used, const int n_expert, const int nb0, const int nb1,
        const int grouped_arg, const int stage_arg, const mach1_d4_hash hash,
        const uint16_t * trellis2 = nullptr, const int32_t * offs2 = nullptr, const float * gw2 = nullptr,
        const float * scr_u2 = nullptr, float * scr_p2 = nullptr) {
    MACH1_D4_DYN_SMEM(float, tg_u);
    if (blockIdx.z != 0) {
        trellis = trellis2; offs = offs2; gw = gw2; scr_u = scr_u2; scr_p = scr_p2;
    }
    mach1_pdl_trigger();
    mach1_pdl_wait();
    static_assert(!CMP || (STAGE && !TAB), "CMP is the staged computed-code decode walk");
    static_assert(!TSTG || CMP, "TSTG stages the CMP walk's trellis");
    static_assert(RB == 1 || (CMP && !TSTG), "RB row tiles: CMP walk without TSTG");
    constexpr int TT = STAGE ? MACH1_D4_TT : TTG;
    const bool stage   = STAGE;
    const int  grouped = STAGE ? 0 : grouped_arg;
    (void) stage_arg;
    __shared__ float red[RB][NWP][16];
    __shared__ __align__(64) float slev[CMP ? 16 : 1];
    __shared__ float sgw[CMP ? 512 + RB - 1 : 1];

    const int rb   = blockIdx.x*RB;
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int m    = Mb*16;

    int e, np, pl0;
    if (grouped) {
        e   = blockIdx.y;
        pl0 = map[e];
        np  = map[e + 1] - pl0;
        if (np == 0) {
            return;
        }
    } else {
        pl0 = blockIdx.y;
        e   = mach1_d4_id(ids, pl0, n_used, nb0, nb1);
        np  = 1;
    }

    const int      K4    = offs[2*e + 1];
    const int      words = 4*K4;
    const int      r     = K4 - 4;
    const float    un    = units[r];
    const uint32_t mult  = (uint32_t) hash.v[3*r];
    const uint64_t lv    = (uint64_t)(uint32_t) hash.v[3*r + 1] | ((uint64_t)(uint32_t) hash.v[3*r + 2] << 32);
    const uint16_t * zr  = zt + (int64_t) r*65536;
    const uint16_t * tr  = trellis + (int64_t) offs[2*e] + (int64_t) rb*Nb*words;
    const float    * gwe = gw + (int64_t) e*(Mb + Nb);

    const int b0  = 2*lane*K4;
    const int wi  = b0 >> 4;
    const int o   = b0 & 15;
    const int wi1 = wi + 1 < words ? wi + 1 : wi + 1 - words;
    const int wi2 = wi + 2 < words ? wi + 2 : wi + 2 - words;
    const int ri  = lane >> 1;
    const int ci  = (8*lane) & 15;

    const int depth = (Nb + NWP - 1)/NWP;
    const int cb0   = wid*depth;
    const int cbe   = min(cb0 + depth, Nb);

    uint16_t * sw = (uint16_t *) (tg_u + n);
    (void) sw;
    if constexpr (TSTG) {
        const int nw = Nb*words;
#ifndef MACH1_D4_EMU
        if ((((uintptr_t) tr) & 15) == 0 && (nw & 7) == 0) {
            const uint4 * src = (const uint4 *) tr;
            for (int i = threadIdx.x; i < nw/8; i += NWP*32) {
                cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(sw + 8*i), src + i);
            }
        } else
#endif
        {
            for (int i = threadIdx.x; i < nw; i += NWP*32) {
                sw[i] = tr[i];
            }
        }
    }
    if constexpr (CMP) {
        if (threadIdx.x < 16) {
            slev[threadIdx.x] = (float)(((int32_t)((uint32_t)(lv >> (4*threadIdx.x)) << 28)) >> 28);
        }
        for (int i = threadIdx.x; i < Nb + RB - 1; i += NWP*32) {
            sgw[i] = un*gwe[rb + i <= Nb - 1 ? Mb + Nb - 1 - (rb + i) : Mb + Nb - 2 - (rb + i)];
        }
    }
    if (stage) {
        const float * u = scr_u + (int64_t) pl0*n;
        for (int i = threadIdx.x; i < n; i += NWP*32) {
            tg_u[i] = u[i];
        }
#ifndef MACH1_D4_EMU
        if constexpr (TSTG) {
            cp_async_wait_all();
        }
#endif
        __syncthreads();
    }

    for (int k0 = 0; k0 < np; k0 += TT) {
        const int ntt = min(TT, np - k0);
        int pk[TT];
#pragma unroll
        for (int k = 0; k < TT; ++k) {
            pk[k] = grouped ? (k < ntt ? map[n_expert + 1 + pl0 + k0 + k] : 0) : pl0;
        }
        float acc[TT*RB];
#pragma unroll
        for (int k = 0; k < TT*RB; ++k) {
            acc[k] = 0.0f;
        }

        const auto wave = [&](int cb) {
            return rb + cb <= Nb - 1 ? Mb + Nb - 1 - (rb + cb) : Mb + Nb - 2 - (rb + cb);
        };
        const auto words48 = [&](int cb) {
            const uint16_t * tw = tr + (int64_t) cb*words;
            return ((uint64_t) tw[wi] << 32) | ((uint64_t) tw[wi1] << 16) | (uint64_t) tw[wi2];
        };
        const auto st0 = [&](uint64_t x) { return (uint32_t)(x >> (32 - o)) & 0xFFFFu; };
        const auto st1 = [&](uint64_t x) { return (uint32_t)(x >> (32 - o - K4)) & 0xFFFFu; };
        if constexpr (CMP) {
            const bool wl = lane < words;
            int s0 = wi, s1 = wi1, s2 = wi2;
            asm volatile("" : "+r"(s0), "+r"(s1), "+r"(s2));
            const auto word1 = [&](int j, int cb) -> uint32_t {
                if constexpr (TSTG) {
                    return wl ? (uint32_t) sw[cb*words + lane] : 0u;
                }
                return wl ? (uint32_t) tr[(int64_t) (j*Nb + cb)*words + lane] : 0u;
            };
            const auto win = [&](uint32_t x) -> uint64_t {
                const uint32_t a = __shfl_sync(0xFFFFFFFFu, x, s0);
                const uint32_t b = __shfl_sync(0xFFFFFFFFu, x, s1);
                const uint32_t c = __shfl_sync(0xFFFFFFFFu, x, s2);
                return ((uint64_t) a << 32) | ((uint64_t) b << 16) | (uint64_t) c;
            };
            const uint32_t lb = (uint32_t) __cvta_generic_to_shared(slev);
            const auto lev = [&](uint32_t a) {
                float v;
                asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(a));
                return v;
            };
            uint64_t nx48[RB];
            uint32_t nnw[RB];
#pragma unroll
            for (int j = 0; j < RB; ++j) {
                nx48[j] = 0;
                nnw[j]  = 0;
                if (cb0 < cbe) {
                    nx48[j] = win(word1(j, cb0));
                }
                if (cb0 + 1 < cbe) {
                    nnw[j] = word1(j, cb0 + 1);
                }
            }
            for (int cb = cb0; cb < cbe; ++cb) {
                float4 u0, u1;
#pragma unroll
                for (int j = 0; j < RB; ++j) {
                    const uint64_t w48 = nx48[j];
                    nx48[j] = win(nnw[j]);
                    if (cb + 2 < cbe) {
                        nnw[j] = word1(j, cb + 2);
                    }
                    const uint32_t h0 = (st0(w48)*mult) >> 16;
                    const uint32_t h1 = (st1(w48)*mult) >> 16;
                    float ws[8];
                    ws[0] = lev(lb | ((h0 << 2) & 0x3Cu));
                    ws[4] = lev(lb | ((h1 << 2) & 0x3Cu));
#pragma unroll
                    for (int c = 1; c < 4; ++c) {
                        ws[c]     = lev(lb | ((h0 >> (4*c - 2)) & 0x3Cu));
                        ws[c + 4] = lev(lb | ((h1 >> (4*c - 2)) & 0x3Cu));
                    }
                    const float sc = sgw[cb + j];
                    if (j == 0) {
                        const int uo = cb*16 + ci;
                        u0 = *(const float4 *)(tg_u + uo);
                        u1 = *(const float4 *)(tg_u + uo + 4);
                    }
                    acc[j] += sc*(ws[0]*u0.x + ws[1]*u0.y + ws[2]*u0.z + ws[3]*u0.w
                                + ws[4]*u1.x + ws[5]*u1.y + ws[6]*u1.z + ws[7]*u1.w);
                }
            }
        } else {
        const bool tab = mult == 0;
        uint64_t nx48 = 0, nn48 = 0;
        float    nxg  = 0.0f, nng = 0.0f;
        uint32_t nz0  = 0, nz1 = 0;
        if (cb0 < cbe) {
            nx48 = words48(cb0);
            nxg  = gwe[wave(cb0)];
            if (tab) {
                nz0 = MACH1_D4_LDG(zr + st0(nx48));
                nz1 = MACH1_D4_LDG(zr + st1(nx48));
            }
        }
        if (cb0 + 1 < cbe) {
            nn48 = words48(cb0 + 1);
            nng  = gwe[wave(cb0 + 1)];
        }
        for (int cb = cb0; cb < cbe; ++cb) {
            const uint64_t w48 = nx48;
            const float    gv  = nxg;
            const uint32_t z0  = nz0;
            const uint32_t z1  = nz1;
            nx48 = nn48;
            nxg  = nng;
            if (cb + 2 < cbe) {
                nn48 = words48(cb + 2);
                nng  = gwe[wave(cb + 2)];
            }
            if (tab && cb + 1 < cbe) {
                nz0 = MACH1_D4_LDG(zr + st0(nx48));
                nz1 = MACH1_D4_LDG(zr + st1(nx48));
            }
            float ws[8];
            if (tab) {
                mach1_d4_unpack(z0, ws);
                mach1_d4_unpack(z1, ws + 4);
            } else {
                mach1_d4_levels(st0(w48), mult, lv, zr, ws);
                mach1_d4_levels(st1(w48), mult, lv, zr, ws + 4);
            }
            const float sc = un*gv;
            const int uo = cb*16 + ci;
            if (stage) {
                const float4 u0 = *(const float4 *)(tg_u + uo);
                const float4 u1 = *(const float4 *)(tg_u + uo + 4);
                acc[0] += sc*(ws[0]*u0.x + ws[1]*u0.y + ws[2]*u0.z + ws[3]*u0.w
                            + ws[4]*u1.x + ws[5]*u1.y + ws[6]*u1.z + ws[7]*u1.w);
            } else {
#pragma unroll
                for (int k = 0; k < TT; ++k) {
                    if (k < ntt) {
                        const float * up = scr_u + (int64_t) pk[k]*n + uo;
                        const float4 u0 = *(const float4 *)(up);
                        const float4 u1 = *(const float4 *)(up + 4);
                        acc[k] += sc*(ws[0]*u0.x + ws[1]*u0.y + ws[2]*u0.z + ws[3]*u0.w
                                    + ws[4]*u1.x + ws[5]*u1.y + ws[6]*u1.z + ws[7]*u1.w);
                    }
                }
            }
        }
        }

#pragma unroll
        for (int k = 0; k < TT; ++k) {
            if (k < ntt) {
#pragma unroll
                for (int j = 0; j < RB; ++j) {
                    const float sred = acc[k*RB + j] + __shfl_xor_sync(0xFFFFFFFFu, acc[k*RB + j], 1);
                    if ((lane & 1) == 0) {
                        red[j][wid][ri] = sred;
                    }
                }
                __syncthreads();
                if ((int) threadIdx.x < 16*RB) {
                    const int j = threadIdx.x >> 4;
                    float s2 = 0.0f;
                    for (int w = 0; w < NWP; ++w) {
                        s2 += red[j][w][threadIdx.x & 15];
                    }
                    scr_p[(int64_t) pk[k]*m + rb*16 + threadIdx.x] = s2;
                }
                __syncthreads();
            }
        }
    }
}

template <int WG = MACH1_D4_WG>
static __global__ void __launch_bounds__(WG) mach1_d4_redout_kernel(const float * scr_p, const half * sv, const int32_t * ids, float * dst,
        const int d, const int n_used, const int nb0, const int nb1) {
    MACH1_D4_DYN_SMEM(float, sh);
    const int p = blockIdx.x;
    const int e = mach1_d4_id(ids, p, n_used, nb0, nb1);
    const float * v = scr_p + (int64_t) p*d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        sh[i] = v[i];
    }
    __syncthreads();
    mach1_d4_fwht<WG >= 1024 ? 8 : 32>(sh, d, threadIdx.x, blockDim.x);
    const half * se = sv + (int64_t) e*d;
    float * y = dst + (int64_t) p*d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        y[i] = sh[i]*__half2float(se[i]);
    }
}


#define MACH1_D4_TC_LO 2048.0f
#if defined(MACH1_D4_EMU) || defined(MACH1_D4_HAVE_WMMA)
#define MACH1_D4_TC_BUILT true
#else
#define MACH1_D4_TC_BUILT false
#endif
#define MACH1_D4_TC_PC 64

template <int NW, int MINB = 0, int CB = 1>
static __global__ void __launch_bounds__(NW*32, MINB) mach1_d4_walk_tc_kernel(
        const uint16_t * trellis, const int32_t * offs, const float * gw, const uint16_t * zt, const float * units,
        const int32_t * map, const float * scr_u, const float * uscale, float * scr_p,
        const int n, const int Mb, const int Nb, const int n_expert, const mach1_d4_hash hash) {
#if defined(MACH1_D4_EMU) || (defined(MACH1_D4_HAVE_WMMA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700)
    using namespace nvcuda;
    __shared__ __align__(32) half  sa[NW][CB][256];
    __shared__ __align__(32) half  sbh[2*CB][16*MACH1_D4_TC_PC];
    __shared__ __align__(32) half  sbl[2*CB][16*MACH1_D4_TC_PC];
    __shared__ __align__(32) float so[NW][256];
    __shared__ int   pidx[MACH1_D4_TC_PC];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int w    = tid >> 5;
    const int e    = blockIdx.y;
    const int pl0  = map[e];
    const int np   = map[e + 1] - pl0;
    if (np == 0) {
        return;
    }
    const int m  = Mb*16;
    const int rt = blockIdx.x*NW + w;

    const int      K4    = offs[2*e + 1];
    const int      words = 4*K4;
    const int      r     = K4 - 4;
    const float    un    = units[r];
    const uint32_t mult  = (uint32_t) hash.v[3*r];
    const uint64_t lv    = (uint64_t)(uint32_t) hash.v[3*r + 1] | ((uint64_t)(uint32_t) hash.v[3*r + 2] << 32);
    const uint16_t * zr  = zt + (int64_t) r*65536;
    const uint16_t * tr  = trellis + (int64_t) offs[2*e] + (int64_t) rt*Nb*words;
    const float    * gwe = gw + (int64_t) e*(Mb + Nb);

    const int b0  = 2*lane*K4;
    const int wi  = b0 >> 4;
    const int o   = b0 & 15;
    const int wi1 = wi + 1 < words ? wi + 1 : wi + 1 - words;
    const int wi2 = wi + 2 < words ? wi + 2 : wi + 2 - words;
    const int ri  = lane >> 1;
    const int ci  = (8*lane) & 15;

    for (int k0 = 0; k0 < np; k0 += MACH1_D4_TC_PC) {
        const int npc = min(MACH1_D4_TC_PC, np - k0);
        const int nf  = (npc + 15)/16;
        for (int i = tid; i < MACH1_D4_TC_PC; i += NW*32) {
            pidx[i] = i < npc ? map[n_expert + 1 + pl0 + k0 + i] : -1;
        }
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MACH1_D4_TC_PC/16];
        for (int f = 0; f < MACH1_D4_TC_PC/16; ++f) {
            wmma::fill_fragment(acc[f], 0.0f);
        }
        __syncthreads();

        const auto words48 = [&](int cb) {
            const uint16_t * tw = tr + (int64_t) cb*words;
            return ((uint64_t) tw[wi] << 32) | ((uint64_t) tw[wi1] << 16) | (uint64_t) tw[wi2];
        };
        const auto st0 = [&](uint64_t x) { return (uint32_t)(x >> (32 - o)) & 0xFFFFu; };
        const auto st1 = [&](uint64_t x) { return (uint32_t)(x >> (32 - o - K4)) & 0xFFFFu; };
        const bool tab = mult == 0;
        constexpr int BI = (16*MACH1_D4_TC_PC + NW*32 - 1)/(NW*32);
        const int pw = nf*16;
        int   bdst[BI];
        int   brow[BI];
        int   bk[BI];
        float bv[CB][BI];
        uint64_t nw48[CB];
#pragma unroll
        for (int i = 0; i < BI; ++i) {
            const int idx = tid + i*NW*32;
            const int k   = idx / pw;
            const int j   = idx - k*pw;
            bdst[i] = idx < 16*pw ? k*MACH1_D4_TC_PC + j : -1;
            brow[i] = idx < 16*pw ? pidx[j] : -1;
            bk[i]   = k;
        }
#pragma unroll
        for (int c = 0; c < CB; ++c) {
#pragma unroll
            for (int i = 0; i < BI; ++i) {
                bv[c][i] = brow[i] >= 0 && c < Nb ? scr_u[(int64_t) brow[i]*n + c*16 + bk[i]] : 0.0f;
            }
            nw48[c] = c < Nb ? words48(c) : 0;
        }
        for (int cb0 = 0; cb0 < Nb; cb0 += CB) {
            uint64_t w48[CB];
#pragma unroll
            for (int c = 0; c < CB; ++c) {
                w48[c] = nw48[c];
                if (cb0 + CB + c < Nb) {
                    nw48[c] = words48(cb0 + CB + c);
                }
            }
#pragma unroll
            for (int c = 0; c < CB; ++c) {
                const int cb = cb0 + c;
                if (cb < Nb) {
                    half * bh = sbh[cb % (2*CB)];
                    half * bl = sbl[cb % (2*CB)];
#pragma unroll
                    for (int i = 0; i < BI; ++i) {
                        if (bdst[i] >= 0) {
                            const float v = bv[c][i];
                            const half  h = __float2half(v);
                            bh[bdst[i]] = h;
                            bl[bdst[i]] = __float2half((v - __half2float(h))*MACH1_D4_TC_LO);
                        }
                        if (cb + CB < Nb) {
                            bv[c][i] = brow[i] >= 0 ? scr_u[(int64_t) brow[i]*n + (cb + CB)*16 + bk[i]] : 0.0f;
                        }
                    }
                }
            }
            __syncwarp();
#pragma unroll
            for (int c = 0; c < CB; ++c) {
                if (cb0 + c < Nb) {
                    float ws[8];
                    if (tab) {
                        mach1_d4_unpack(MACH1_D4_LDG(zr + st0(w48[c])), ws);
                        mach1_d4_unpack(MACH1_D4_LDG(zr + st1(w48[c])), ws + 4);
                    } else {
                        mach1_d4_levels(st0(w48[c]), mult, lv, zr, ws);
                        mach1_d4_levels(st1(w48[c]), mult, lv, zr, ws + 4);
                    }
                    half2 hv[4];
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        hv[q] = __halves2half2(__float2half(ws[2*q]), __float2half(ws[2*q + 1]));
                    }
                    *(uint4 *) (sa[w][c] + ri*16 + ci) = *(const uint4 *) hv;
                }
            }
            __syncthreads();

#pragma unroll
            for (int c = 0; c < CB; ++c) {
                const int cb = cb0 + c;
                if (cb < Nb) {
                    const int   wv = rt + cb <= Nb - 1 ? Mb + Nb - 1 - (rt + cb) : Mb + Nb - 2 - (rt + cb);
                    const float sc = un*gwe[wv];
                    const half * bh = sbh[cb % (2*CB)];
                    const half * bl = sbl[cb % (2*CB)];
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
                    wmma::load_matrix_sync(fa, sa[w][c], 16);
                    for (int f = 0; f < MACH1_D4_TC_PC/16; ++f) {
                        if (f < nf) {
                            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fbh, fbl;
                            wmma::load_matrix_sync(fbh, bh + f*16, MACH1_D4_TC_PC);
                            wmma::load_matrix_sync(fbl, bl + f*16, MACH1_D4_TC_PC);
                            wmma::fragment<wmma::accumulator, 16, 16, 16, float> t, tl;
                            wmma::fill_fragment(t, 0.0f);
                            wmma::fill_fragment(tl, 0.0f);
                            wmma::mma_sync(t, fa, fbh, t);
                            wmma::mma_sync(tl, fa, fbl, tl);
                            for (int i = 0; i < t.num_elements; ++i) {
                                acc[f].x[i] += sc*(t.x[i] + tl.x[i]*(1.0f/MACH1_D4_TC_LO));
                            }
                        }
                    }
                }
            }
        }

        for (int f = 0; f < MACH1_D4_TC_PC/16; ++f) {
            if (f < nf) {
                wmma::store_matrix_sync(so[w], acc[f], 16, wmma::mem_row_major);
                __syncwarp();
                for (int idx = lane; idx < 256; idx += 32) {
                    const int rr = idx / 16;
                    const int j  = f*16 + idx % 16;
                    if (j < npc) {
                        scr_p[(int64_t) pidx[j]*m + rt*16 + rr] = so[w][idx]*uscale[pidx[j]];
                    }
                }
                __syncwarp();
            }
        }
        __syncthreads();
    }
#else
    GGML_UNUSED_VARS(trellis, offs, gw, zt, units, map, scr_u, uscale, scr_p, n, Mb, Nb, n_expert, hash);
    NO_DEVICE_CODE;
#endif
}

static __global__ void mach1_d4_ustage2_kernel(const float * x, const half * su_g, const half * su_u,
        const int32_t * ids, float * u_g, float * u_u,
        const int d, const int n_used, const int xne1, const int nb0, const int nb1) {
    MACH1_D4_DYN_SMEM(float, sh);
    const int p = blockIdx.x;
    const int e = mach1_d4_id(ids, p, n_used, nb0, nb1);
    const int t = p / n_used;
    const float * xc = x + (int64_t)(xne1 == 1 ? t : p)*d;
    const half  * se = (blockIdx.y == 0 ? su_g : su_u) + (int64_t) e*d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        sh[i] = __half2float(se[i])*xc[i];
    }
    __syncthreads();
    mach1_d4_fwht(sh, d, threadIdx.x, blockDim.x);
    float * u = (blockIdx.y == 0 ? u_g : u_u) + (int64_t) p*d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        u[i] = sh[i];
    }
}

static __global__ void mach1_d4_glu_kernel(const float * p_g, const float * p_u, const half * sv_g, const half * sv_u,
        const half * su_d, const int32_t * ids, float * u_d, const int d, const int n_used, const int nb0, const int nb1) {
    MACH1_D4_DYN_SMEM(float, sh);
    float * sg = sh;
    float * su = sh + d;
    const int p = blockIdx.x;
    const int e = mach1_d4_id(ids, p, n_used, nb0, nb1);
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        sg[i] = p_g[(int64_t) p*d + i];
        su[i] = p_u[(int64_t) p*d + i];
    }
    __syncthreads();
    mach1_d4_fwht(sg, d, threadIdx.x, blockDim.x);
    mach1_d4_fwht(su, d, threadIdx.x, blockDim.x);
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float g = sg[i]*__half2float(sv_g[(int64_t) e*d + i]);
        const float u = su[i]*__half2float(sv_u[(int64_t) e*d + i]);
        const float h = g/(1.0f + expf(-g))*u;
        sg[i] = __half2float(su_d[(int64_t) e*d + i])*h;
    }
    __syncthreads();
    mach1_d4_fwht(sg, d, threadIdx.x, blockDim.x);
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        u_d[(int64_t) p*d + i] = sg[i];
    }
}

#define MACH1_D4_SUM_MAXD 4096

static __global__ void mach1_d4_redsum_kernel(const float * p_d, const half * sv_d, const int32_t * ids,
        const float * w, float * out, const int d, const int n_used, const int nb0, const int nb1,
        const int64_t out_stride) {
    MACH1_D4_DYN_SMEM(float, sh);
    const int t = blockIdx.x;
    float acc[MACH1_D4_SUM_MAXD/MACH1_D4_WG];
    for (int s = 0; s < n_used; ++s) {
        const int p = t*n_used + s;
        const int e = mach1_d4_id(ids, p, n_used, nb0, nb1);
        for (int i = threadIdx.x; i < d; i += blockDim.x) {
            sh[i] = p_d[(int64_t) p*d + i];
        }
        __syncthreads();
        mach1_d4_fwht(sh, d, threadIdx.x, blockDim.x);
        const float wp = w[p];
        for (int k = 0, i = threadIdx.x; i < d; ++k, i += blockDim.x) {
            const float v = sh[i]*__half2float(sv_d[(int64_t) e*d + i])*wp;
            acc[k] = s == 0 ? v : acc[k] + v;
        }
        __syncthreads();
    }
    for (int k = 0, i = threadIdx.x; i < d; ++k, i += blockDim.x) {
        out[t*out_stride + i] = acc[k];
    }
}

static __global__ void mach1_d4_redwsum_kernel(const float * p_d, const half * sv_d, const int32_t * ids,
        const float * w, float * wbuf, float * out, int * counters, const int d, const int n_used,
        const int nb0, const int nb1, const int64_t out_stride) {
    MACH1_D4_DYN_SMEM(float, sh);
    __shared__ int is_last;
    const int p = blockIdx.x;
    const int t = p / n_used;
    const int e = mach1_d4_id(ids, p, n_used, nb0, nb1);
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        sh[i] = p_d[(int64_t) p*d + i];
    }
    __syncthreads();
    mach1_d4_fwht(sh, d, threadIdx.x, blockDim.x);
    const float wp = w[p];
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        wbuf[(int64_t) p*d + i] = sh[i]*__half2float(sv_d[(int64_t) e*d + i])*wp;
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        is_last = atomicAdd(&counters[t], 1) == n_used - 1;
    }
    __syncthreads();
    if (!is_last) {
        return;
    }
    if (threadIdx.x == 0) {
        counters[t] = 0;
    }
    __threadfence();
    const float * wt = wbuf + (int64_t) t*n_used*d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float acc = __ldcg(&wt[i]);
        for (int s = 1; s < n_used; ++s) {
            acc += __ldcg(&wt[(int64_t) s*d + i]);
        }
        out[t*out_stride + i] = acc;
    }
}

#ifndef MACH1_D4_EMU

#include <mutex>

#define MACH1_D4_GROUP_MIN 4

static int * mach1_d4_counters(int device) {
    static int * cnt[GGML_CUDA_MAX_DEVICES] = { nullptr };
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    if (cnt[device] != nullptr) {
        return cnt[device];
    }
    int * c = nullptr;
    if (cudaMalloc(&c, 512*sizeof(int)) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    cudaStream_t up = nullptr;
    if (cudaStreamCreateWithFlags(&up, cudaStreamNonBlocking) != cudaSuccess) {
        cudaGetLastError();
        cudaFree(c);
        return nullptr;
    }
    CUDA_CHECK(cudaMemsetAsync(c, 0, 512*sizeof(int), up));
    CUDA_CHECK(cudaStreamSynchronize(up));
    CUDA_CHECK(cudaStreamDestroy(up));
    cnt[device] = c;
    return c;
}

bool ggml_cuda_mach1_d4_supported(const ggml_tensor * op) {
    const int64_t n = op->src[2]->ne[0];
    const int64_t m = op->src[3]->ne[0];
    const int64_t E = op->src[1]->ne[1];
    auto dim_ok = [](int64_t d) {
        if (d <= 0 || d % 16 != 0) {
            return false;
        }
        while (d % 2 == 0) {
            d /= 2;
        }
        return d == 1 || d == 3 || d == 5;
    };
    return dim_ok(n) && dim_ok(m) && n <= 8192 && m <= 8192 && E + 1024 <= 12288 &&
           ggml_is_contiguous(op->src[8]) && ggml_is_contiguous(op);
}

static bool mach1_d4_cmp_ok(const mach1_d4_hash & hash) {
    static const bool on = getenv("GGML_MACH1_D4_CMP") == nullptr || atoi(getenv("GGML_MACH1_D4_CMP")) != 0;
    if (!on) {
        return false;
    }
    for (int r = 0; r < 5; ++r) {
        if (hash.v[3*r] == 0) {
            return false;
        }
    }
    return true;
}

static bool mach1_d4_tstg_on() {
    static const bool on = getenv("GGML_MACH1_D4_TSTG") != nullptr && atoi(getenv("GGML_MACH1_D4_TSTG")) != 0;
    return on;
}

static int mach1_d4_rb(bool pair, int Mb) {
    static const int rs = getenv("GGML_MACH1_D4_RB") == nullptr ? 1 : atoi(getenv("GGML_MACH1_D4_RB"));
    static const int rp = getenv("GGML_MACH1_D4_RB_PAIR") == nullptr ? 1 : atoi(getenv("GGML_MACH1_D4_RB_PAIR"));
    const int r = pair ? rp : rs;
    return (r == 2 || r == 4) && Mb % r == 0 ? r : 1;
}

static void mach1_d4_l1_init() {
    static bool l1_done[GGML_CUDA_MAX_DEVICES] = {false};
    const int l1_dev = ggml_cuda_get_device();
    if (l1_done[l1_dev]) {
        return;
    }
    l1_done[l1_dev] = true;
    const char * e = getenv("GGML_MACH1_D4_L1");
    if (e != nullptr && atoi(e) == 0) {
        return;
    }
    CUDA_CHECK(cudaFuncSetAttribute(mach1_d4_walk_kernel<true, 16, true>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1));
    CUDA_CHECK(cudaFuncSetAttribute(mach1_d4_walk_kernel<true, 8, true>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1));
    CUDA_CHECK(cudaFuncSetAttribute(mach1_d4_walk_kernel<false, 8, true>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1));
}

struct mach1_d4_wsum {
    const float * w;
    float       * out;
    int64_t       out_stride;
    int         * counters;
};

static void mach1_d4_mm_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const float * u_pre,
                             const mach1_d4_wsum * ws = nullptr) {
    const ggml_tensor * trellis = dst->src[0];
    const ggml_tensor * offs    = dst->src[1];
    const ggml_tensor * su      = dst->src[2];
    const ggml_tensor * sv      = dst->src[3];
    const ggml_tensor * gw      = dst->src[4];
    const ggml_tensor * zt      = dst->src[5];
    const ggml_tensor * units   = dst->src[6];
    const ggml_tensor * ids     = dst->src[7];
    const ggml_tensor * x       = dst->src[8];

    const int n        = (int) su->ne[0];
    const int m        = (int) sv->ne[0];
    const int n_expert = (int) offs->ne[1];
    const int n_used   = (int) ids->ne[0];
    const int n_tok    = (int) ids->ne[1];
    const int n_pairs  = n_used*n_tok;
    const int xne1     = (int) x->ne[1];
    const int nb0      = (int)(ids->nb[0]/sizeof(int32_t));
    const int nb1      = (int)(ids->nb[1]/sizeof(int32_t));

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<float>   u_buf  (ctx.pool(), (size_t) n_pairs*n);
    ggml_cuda_pool_alloc<float>   p_buf  (ctx.pool(), (size_t) n_pairs*m);
    ggml_cuda_pool_alloc<int32_t> map_buf(ctx.pool(), (size_t) n_expert + 1 + n_pairs);

    mach1_d4_hash hash;
    memset(&hash, 0, sizeof(hash));
    if (ggml_get_op_params_i32(dst, 15) == 1) {
        for (int i = 0; i < 15; ++i) {
            hash.v[i] = ggml_get_op_params_i32(dst, i);
        }
    }

    const bool grouped = n_tok >= MACH1_D4_GROUP_MIN;
    const bool stage   = !grouped && (size_t) n*sizeof(float) <= 32768;

    const int32_t * idsd = (const int32_t *) ids->data;

    static const bool tc_env = getenv("GGML_MACH1_D4_TC") == nullptr || atoi(getenv("GGML_MACH1_D4_TC")) != 0;
    const bool tc = MACH1_D4_TC_BUILT && grouped && tc_env && m % 64 == 0 &&
        ggml_cuda_info().devices[ctx.device].cc >= GGML_CUDA_CC_VOLTA && !GGML_CUDA_CC_IS_AMD(ggml_cuda_info().devices[ctx.device].cc);
    const bool tab = hash.v[0] == 0;
    mach1_d4_l1_init();
    ggml_cuda_pool_alloc<float> us_buf(ctx.pool());
    if (tc) {
        us_buf.alloc((size_t) n_pairs);
    }

    static const int uwg_env = getenv("GGML_MACH1_D4_UWG") == nullptr ? 1024 : atoi(getenv("GGML_MACH1_D4_UWG"));
    const bool uwide = !grouped && uwg_env >= 1024;
    GGML_ASSERT(u_pre == nullptr || (uwide && !tc));
    if (u_pre != nullptr) {
    } else if (uwide && !tc && mach1_d4_warp_stage(false, (const float *) x->data, (const half *) su->data, idsd, u_buf.get(),
                                                   n, n_pairs, n_used, xne1, nb0, nb1, stream)) {
    } else
    (uwide ? mach1_d4_ustage_kernel<1024> : mach1_d4_ustage_kernel<MACH1_D4_WG>)<<<n_pairs, uwide ? 1024 : MACH1_D4_WG, n*sizeof(float), stream>>>(
        (const float *) x->data, (const half *) su->data, idsd, u_buf.get(), n, n_used, xne1, nb0, nb1,
        tc ? us_buf.get() : nullptr);
    if (grouped) {
        const int wg = 1024;
        mach1_d4_map_kernel<<<1, wg, (n_expert + wg)*sizeof(int32_t), stream>>>(
            idsd, map_buf.get(), n_expert, n_used, n_pairs, nb0, nb1);
    }
    if (tc) {
        static const bool w8_env = getenv("GGML_MACH1_D4_TC_W8") == nullptr || atoi(getenv("GGML_MACH1_D4_TC_W8")) != 0;
        static const int minb = getenv("GGML_MACH1_D4_TC_MINB") == nullptr ? 0 : atoi(getenv("GGML_MACH1_D4_TC_MINB"));
        const bool w8 = w8_env && m % 128 == 0;
        static const int cbn = getenv("GGML_MACH1_D4_TC_CB") == nullptr ? 2 : atoi(getenv("GGML_MACH1_D4_TC_CB"));
        (w8 ? (minb == 3 ? (cbn == 2 ? mach1_d4_walk_tc_kernel<8, 3, 2> : mach1_d4_walk_tc_kernel<8, 3>)
                         : (cbn == 2 ? mach1_d4_walk_tc_kernel<8, 0, 2> : mach1_d4_walk_tc_kernel<8>))
            : (minb == 3 ? (cbn == 2 ? mach1_d4_walk_tc_kernel<4, 6, 2> : mach1_d4_walk_tc_kernel<4, 6>)
                         : (cbn == 2 ? mach1_d4_walk_tc_kernel<4, 0, 2> : mach1_d4_walk_tc_kernel<4>)))
            <<<dim3((unsigned)(m/(w8 ? 128 : 64)), (unsigned) n_expert, 1), w8 ? 256 : 128, 0, stream>>>(
            (const uint16_t *) trellis->data, (const int32_t *) offs->data, (const float *) gw->data,
            (const uint16_t *) zt->data, (const float *) units->data, map_buf.get(), u_buf.get(), us_buf.get(), p_buf.get(),
            n, m/16, n/16, n_expert, hash);
        mach1_d4_redout_kernel<MACH1_D4_WG><<<n_pairs, MACH1_D4_WG, m*sizeof(float), stream>>>(
            p_buf.get(), (const half *) sv->data, idsd, (float *) dst->data, m, n_used, nb0, nb1);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    dim3 grid((unsigned)(m/16), (unsigned)(grouped ? n_expert : n_pairs), 1);
    static const int w16_env = getenv("GGML_MACH1_D4_W16") == nullptr ? -1 : atoi(getenv("GGML_MACH1_D4_W16"));
    static const int w16_minnb = getenv("GGML_MACH1_D4_W16_MINNB") == nullptr ? 32 : atoi(getenv("GGML_MACH1_D4_W16_MINNB"));
    const bool w16 = stage && n/16 >= std::max(32, w16_minnb) && (w16_env > 0 || (w16_env < 0 && tab));
    const bool cmp = !tab && mach1_d4_cmp_ok(hash);
    const bool tstg = cmp && stage && !w16 && mach1_d4_tstg_on();
    const int  rbn  = cmp && stage && !w16 && !tstg ? mach1_d4_rb(false, m/16) : 1;
    grid.x /= rbn;
    static const int ttg = getenv("GGML_MACH1_D4_TTG") == nullptr ? 8 : atoi(getenv("GGML_MACH1_D4_TTG"));
    mach1_launch_pdl_raw(
     (tab ? (w16 ? mach1_d4_walk_kernel<true, 16, true> : stage ? mach1_d4_walk_kernel<true, 8, true>
                : ttg == 16 ? mach1_d4_walk_kernel<false, 8, true, false, 16> : mach1_d4_walk_kernel<false, 8, true>)
          : (w16 ? mach1_d4_walk_kernel<true, 16>       : stage ? (cmp ? (tstg ? mach1_d4_walk_kernel<true, 8, false, true, MACH1_D4_TT, true>
                                                                        : rbn == 4 ? mach1_d4_walk_kernel<true, 8, false, true, MACH1_D4_TT, false, 4>
                                                                        : rbn == 2 ? mach1_d4_walk_kernel<true, 8, false, true, MACH1_D4_TT, false, 2>
                                                                                : mach1_d4_walk_kernel<true, 8, false, true>) : mach1_d4_walk_kernel<true>)
                : ttg == 16 ? mach1_d4_walk_kernel<false, 8, false, false, 16> : mach1_d4_walk_kernel<false>)),
        grid, dim3(w16 ? 512 : MACH1_D4_WG), stage ? n*sizeof(float) + (tstg ? (size_t) (n/16)*64 : 0) : 0, stream,
        (const uint16_t *) trellis->data, (const int32_t *) offs->data, (const float *) gw->data,
        (const uint16_t *) zt->data, (const float *) units->data, idsd, map_buf.get(), u_pre != nullptr ? u_pre : u_buf.get(),
        p_buf.get(), n, m/16, n/16, n_used, n_expert, nb0, nb1, grouped ? 1 : 0, stage ? 1 : 0, hash,
        nullptr, nullptr, nullptr, nullptr, nullptr);
    if (ws != nullptr) {
        GGML_ASSERT(uwide && mach1_d4_warp_ok(m) && !mach1_skip(16));
        const int M = m / ((m % 12 == 0 && ((m/12) & (m/12 - 1)) == 0) ? 12 : 20);
        ggml_cuda_pool_alloc<float> wb(ctx.pool(), (size_t) n_pairs*m);
        const dim3 wgrid((unsigned) n_pairs, (unsigned) mach1_d4_split(M, n_tok), 1);
#define M1_D4WS(E) mach1_launch_pdl_raw(mach1_d4_out_wsum_kernel<E>, wgrid, dim3(1024), m*sizeof(float), stream, \
            (const float *) p_buf.get(), (const half *) sv->data, idsd, ws->w, wb.get(), ws->out, ws->counters, \
            m, n_used, nb0, nb1, ws->out_stride)
        switch (M/32) {
            case 1:  M1_D4WS(1);  break;
            case 2:  M1_D4WS(2);  break;
            case 4:  M1_D4WS(4);  break;
            case 8:  M1_D4WS(8);  break;
            case 16: M1_D4WS(16); break;
            default: M1_D4WS(32); break;
        }
#undef M1_D4WS
    } else if (uwide && mach1_d4_warp_stage(true, p_buf.get(), (const half *) sv->data, idsd, (float *) dst->data,
                                     m, n_pairs, n_used, 1, nb0, nb1, stream)) {
    } else
    (uwide ? mach1_d4_redout_kernel<1024> : mach1_d4_redout_kernel<MACH1_D4_WG>)<<<n_pairs, uwide ? 1024 : MACH1_D4_WG, m*sizeof(float), stream>>>(
        p_buf.get(), (const half *) sv->data, idsd, (float *) dst->data, m, n_used, nb0, nb1);
    CUDA_CHECK(cudaGetLastError());
}


void ggml_cuda_op_mach1_d4_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    mach1_d4_mm_impl(ctx, dst, nullptr);
}

static int mach1_d4_wsum_match(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx,
                               const ggml_tensor * dn, int n_used, int n_tok, mach1_d4_wsum * ws) {
    static const bool on = getenv("GGML_MACH1_D4_WSUM") == nullptr || atoi(getenv("GGML_MACH1_D4_WSUM")) != 0;
    const int md  = (int) dn->src[3]->ne[0];
    const int len = 4 + 2*n_used;
    if (!on || n_used < 2 || n_used > 13 || n_tok > 64 || node_idx + len > cgraph->n_nodes ||
        !mach1_d4_warp_ok(md) || mach1_skip(16)) {
        return 0;
    }
    const ggml_tensor * mul = cgraph->nodes[node_idx + 4];
    if (mul->op != GGML_OP_MUL || mul->src[0] != dn || mul->type != GGML_TYPE_F32) {
        return 0;
    }
    const ggml_tensor * weights = mul->src[1];
    if (weights->type != GGML_TYPE_F32 || !ggml_is_contiguous(weights) ||
        ggml_nelements(weights) != (int64_t) n_used*n_tok || weights->ne[0] != 1 || weights->ne[1] != n_used ||
        mul->ne[0] != md || mul->ne[1] != n_used || mul->ne[2] != n_tok || !ggml_is_contiguous(mul)) {
        return 0;
    }
    const ggml_tensor * acc = nullptr;
    const ggml_tensor * views[13] = { nullptr };
    int nv = 0, na = 0;
    ggml_op ops[4 + 2*13];
    ops[0] = GGML_OP_MACH1_D4_MM; ops[1] = GGML_OP_MACH1_D4_MM; ops[2] = GGML_OP_GLU;
    ops[3] = GGML_OP_MACH1_D4_MM; ops[4] = GGML_OP_MUL;
    for (int k = node_idx + 5; k < node_idx + len; ++k) {
        const ggml_tensor * t = cgraph->nodes[k];
        ops[k - node_idx] = t->op;
        if (t->op == GGML_OP_VIEW) {
            if (nv >= n_used || t->view_src != mul || t->ne[0] != md || t->ne[1] != n_tok || t->nb[1] != mul->nb[2] ||
                (const char *) t->data != (const char *) mul->data + nv*mul->nb[1]) {
                return 0;
            }
            views[nv++] = t;
        } else if (t->op == GGML_OP_ADD) {
            const ggml_tensor * lhs = na == 0 ? views[0] : acc;
            if (lhs == nullptr || na + 1 >= n_used || t->src[0] != lhs || views[na + 1] == nullptr || t->src[1] != views[na + 1] ||
                t->type != GGML_TYPE_F32 || (t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                return 0;
            }
            acc = t;
            na++;
        } else {
            return 0;
        }
    }
    if (nv != n_used || na != n_used - 1) {
        return 0;
    }
    ggml_tensor * out = cgraph->nodes[node_idx + len - 1];
    if (out->ne[0] != md || out->ne[1] != n_tok || out->nb[0] != sizeof(float) || out->type != GGML_TYPE_F32) {
        return 0;
    }
    const int out_idx = node_idx + len - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, len, ops, &out_idx, 1)) {
        return 0;
    }
    int * cnt = mach1_d4_counters(ctx.device);
    if (cnt == nullptr) {
        return 0;
    }
    ws->w          = (const float *) weights->data;
    ws->out        = (float *) out->data;
    ws->out_stride = (int64_t)(out->nb[1]/sizeof(float));
    ws->counters   = cnt;
    return len - 4;
}

int ggml_cuda_mach1_d4_pair_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    static const bool on = getenv("GGML_MACH1_D4_PAIR") == nullptr || atoi(getenv("GGML_MACH1_D4_PAIR")) != 0;
    static const int uwg_env = getenv("GGML_MACH1_D4_UWG") == nullptr ? 1024 : atoi(getenv("GGML_MACH1_D4_UWG"));
    if (!on || uwg_env < 1024 || node_idx + 1 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * a = cgraph->nodes[node_idx];
    ggml_tensor * b = cgraph->nodes[node_idx + 1];
    if (b->op != GGML_OP_MACH1_D4_MM || (b->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return 0;
    }
    const ggml_tensor * ids = a->src[7];
    const ggml_tensor * x   = a->src[8];
    if (b->src[7] != ids || b->src[8] != x || b->src[5] != a->src[5] || b->src[6] != a->src[6] ||
        memcmp(a->op_params, b->op_params, 16*sizeof(int32_t)) != 0) {
        return 0;
    }
    for (int k = 0; k < 7; ++k) {
        if (a->src[k] == b || b->src[k] == a) {
            return 0;
        }
    }
    const int n = (int) a->src[2]->ne[0];
    const int m = (int) a->src[3]->ne[0];
    const int n_used = (int) ids->ne[0];
    const int n_tok  = (int) ids->ne[1];
    if ((int) b->src[2]->ne[0] != n || (int) b->src[3]->ne[0] != m || (int) b->src[1]->ne[1] != (int) a->src[1]->ne[1] ||
        n_tok >= MACH1_D4_GROUP_MIN || (size_t) n*sizeof(float) > 32768 ||
        !mach1_d4_warp_ok(n) || !mach1_d4_warp_ok(m) ||
        !ggml_cuda_mach1_d4_supported(a) || !ggml_cuda_mach1_d4_supported(b) ||
        x->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
        return 0;
    }

    const int n_expert = (int) a->src[1]->ne[1];
    const int n_pairs  = n_used*n_tok;
    const int xne1     = (int) x->ne[1];
    const int nb0      = (int)(ids->nb[0]/sizeof(int32_t));
    const int nb1      = (int)(ids->nb[1]/sizeof(int32_t));
    const int32_t * idsd = (const int32_t *) ids->data;
    cudaStream_t stream = ctx.stream();

    static const bool glu_on = getenv("GGML_MACH1_D4_GLU") == nullptr || atoi(getenv("GGML_MACH1_D4_GLU")) != 0;
    const int m_r = (m % 12 == 0 && ((m/12) & (m/12 - 1)) == 0) ? 12 : 20;
    ggml_tensor * g  = node_idx + 3 < cgraph->n_nodes ? cgraph->nodes[node_idx + 2] : nullptr;
    ggml_tensor * dn = node_idx + 3 < cgraph->n_nodes ? cgraph->nodes[node_idx + 3] : nullptr;
    const bool glu = glu_on && g != nullptr && m == m_r*32 &&
        g->op == GGML_OP_GLU && ggml_get_glu_op(g) == GGML_GLU_OP_SWIGLU && ggml_get_op_params_i32(g, 1) == 0 &&
        g->src[0] == a && g->src[1] == b && g->type == GGML_TYPE_F32 && ggml_is_contiguous(g) &&
        (g->flags & GGML_TENSOR_FLAG_COMPUTE) != 0 &&
        dn->op == GGML_OP_MACH1_D4_MM && (dn->flags & GGML_TENSOR_FLAG_COMPUTE) != 0 && dn->src[8] == g &&
        dn->src[7] == ids && (int) dn->src[2]->ne[0] == m && dn->type == GGML_TYPE_F32 &&
        ggml_cuda_mach1_d4_supported(dn) &&
        ggml_node_has_n_uses(cgraph, node_idx, 1) && ggml_node_has_n_uses(cgraph, node_idx + 1, 1) &&
        ggml_node_has_n_uses(cgraph, node_idx + 2, 1);

    mach1_d4_hash hash;
    memset(&hash, 0, sizeof(hash));
    if (ggml_get_op_params_i32(a, 15) == 1) {
        for (int i = 0; i < 15; ++i) {
            hash.v[i] = ggml_get_op_params_i32(a, i);
        }
    }
    const bool tab = hash.v[0] == 0;
    mach1_d4_l1_init();

    ggml_cuda_pool_alloc<float> ua(ctx.pool(), (size_t) n_pairs*n);
    ggml_cuda_pool_alloc<float> ub(ctx.pool(), (size_t) n_pairs*n);
    ggml_cuda_pool_alloc<float> pa(ctx.pool(), (size_t) n_pairs*m);
    ggml_cuda_pool_alloc<float> pb(ctx.pool(), (size_t) n_pairs*m);

    mach1_d4_warp_stage(false, (const float *) x->data, (const half *) a->src[2]->data, idsd, ua.get(),
                        n, n_pairs, n_used, xne1, nb0, nb1, stream,
                        (const float *) x->data, (const half *) b->src[2]->data, ub.get());
    static const int w16_env   = getenv("GGML_MACH1_D4_W16") == nullptr ? -1 : atoi(getenv("GGML_MACH1_D4_W16"));
    static const int w16_minnb = getenv("GGML_MACH1_D4_W16_MINNB") == nullptr ? 32 : atoi(getenv("GGML_MACH1_D4_W16_MINNB"));
    const bool w16 = n/16 >= std::max(32, w16_minnb) && (w16_env > 0 || (w16_env < 0 && tab));
    const bool cmp = !tab && mach1_d4_cmp_ok(hash);
    const bool tstg = cmp && !w16 && mach1_d4_tstg_on();
    const int  rbn  = cmp && !w16 && !tstg ? mach1_d4_rb(true, m/16) : 1;
    mach1_launch_pdl_raw(
     (tab ? (w16 ? mach1_d4_walk_kernel<true, 16, true> : mach1_d4_walk_kernel<true, 8, true>)
          : (w16 ? mach1_d4_walk_kernel<true, 16>       : cmp ? (tstg ? mach1_d4_walk_kernel<true, 8, false, true, MACH1_D4_TT, true>
                                                              : rbn == 4 ? mach1_d4_walk_kernel<true, 8, false, true, MACH1_D4_TT, false, 4>
                                                              : rbn == 2 ? mach1_d4_walk_kernel<true, 8, false, true, MACH1_D4_TT, false, 2>
                                                                      : mach1_d4_walk_kernel<true, 8, false, true>) : mach1_d4_walk_kernel<true>)),
        dim3((unsigned)(m/16/rbn), (unsigned) n_pairs, 2), dim3(w16 ? 512 : MACH1_D4_WG), n*sizeof(float) + (tstg ? (size_t) (n/16)*64 : 0), stream,
        (const uint16_t *) a->src[0]->data, (const int32_t *) a->src[1]->data, (const float *) a->src[4]->data,
        (const uint16_t *) a->src[5]->data, (const float *) a->src[6]->data, idsd, nullptr, ua.get(), pa.get(),
        n, m/16, n/16, n_used, n_expert, nb0, nb1, 0, 1, hash,
        (const uint16_t *) b->src[0]->data, (const int32_t *) b->src[1]->data, (const float *) b->src[4]->data,
        ub.get(), pb.get());
    if (glu) {
        ggml_cuda_pool_alloc<float> ud(ctx.pool(), (size_t) n_pairs*m);
        if (!mach1_skip(8))
        mach1_launch_pdl_raw(m_r == 12 ? mach1_d4_glu_stage_kernel<12> : mach1_d4_glu_stage_kernel<20>,
            dim3(n_pairs), dim3(1024), m*sizeof(float), stream, pa.get(), pb.get(), (const half *) a->src[3]->data,
            (const half *) b->src[3]->data, (const half *) dn->src[2]->data, idsd, ud.get(), m, 32, n_used, nb0, nb1);
        CUDA_CHECK(cudaGetLastError());
        mach1_d4_wsum ws;
        const int tail = mach1_d4_wsum_match(ctx, cgraph, node_idx, dn, n_used, n_tok, &ws);
        mach1_d4_mm_impl(ctx, dn, ud.get(), tail > 0 ? &ws : nullptr);
        return 3 + tail;
    }
    mach1_d4_warp_stage(true, pa.get(), (const half *) a->src[3]->data, idsd, (float *) a->data,
                        m, n_pairs, n_used, 1, nb0, nb1, stream,
                        pb.get(), (const half *) b->src[3]->data, (float *) b->data);
    CUDA_CHECK(cudaGetLastError());
    return 1;
}

int ggml_cuda_mach1_d4_ffn_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    static const bool on = getenv("GGML_MACH1_D4_FFN") != nullptr && atoi(getenv("GGML_MACH1_D4_FFN")) != 0;
    if (!on) {
        return 0;
    }
    const ggml_tensor * gate = cgraph->nodes[node_idx];
    const ggml_tensor * ids  = gate->src[7];
    const ggml_tensor * x    = gate->src[8];
    const int n_used = (int) ids->ne[0];
    const int n_tok  = (int) ids->ne[1];
    if (n_tok < 1 || n_tok >= MACH1_D4_GROUP_MIN || n_used < 2 || n_used > 16) {
        return 0;
    }
    const int len = 4 + 2*n_used;
    if (node_idx + len > cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * up   = cgraph->nodes[node_idx + 1];
    const ggml_tensor * glu  = cgraph->nodes[node_idx + 2];
    const ggml_tensor * down = cgraph->nodes[node_idx + 3];
    const ggml_tensor * mul  = cgraph->nodes[node_idx + 4];
    if (up->op != GGML_OP_MACH1_D4_MM || glu->op != GGML_OP_GLU || down->op != GGML_OP_MACH1_D4_MM ||
        mul->op != GGML_OP_MUL) {
        return 0;
    }
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || glu->src[0] != gate || glu->src[1] != up ||
        ggml_get_op_params_i32(glu, 1) != 0 || glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu)) {
        return 0;
    }
    if (up->src[8] != x || down->src[8] != glu || up->src[7] != ids || down->src[7] != ids) {
        return 0;
    }
    for (const ggml_tensor * t : { gate, up, down }) {
        if (t->src[5] != gate->src[5] || t->src[6] != gate->src[6] ||
            memcmp(t->op_params, gate->op_params, 16*sizeof(int32_t)) != 0 || !ggml_cuda_mach1_d4_supported(t)) {
            return 0;
        }
    }
    const int n   = (int) gate->src[2]->ne[0];
    const int mff = (int) gate->src[3]->ne[0];
    const int md  = (int) down->src[3]->ne[0];
    if ((int) up->src[2]->ne[0] != n || (int) up->src[3]->ne[0] != mff || (int) down->src[2]->ne[0] != mff ||
        md > MACH1_D4_SUM_MAXD || (size_t) n*sizeof(float) > 32768 || (size_t) mff*sizeof(float) > 32768) {
        return 0;
    }
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || ids->type != GGML_TYPE_I32) {
        return 0;
    }
    const ggml_tensor * weights = mul->src[1];
    if (mul->src[0] != down || weights->type != GGML_TYPE_F32 || !ggml_is_contiguous(weights) ||
        ggml_nelements(weights) != (int64_t) n_used*n_tok || weights->ne[1] != n_used ||
        mul->ne[0] != md || mul->ne[1] != n_used || mul->ne[2] != n_tok) {
        return 0;
    }
    const ggml_tensor * acc = nullptr;
    int nv = 0, na = 0;
    ggml_op ops[4 + 2*16];
    ops[0] = GGML_OP_MACH1_D4_MM; ops[1] = GGML_OP_MACH1_D4_MM; ops[2] = GGML_OP_GLU;
    ops[3] = GGML_OP_MACH1_D4_MM; ops[4] = GGML_OP_MUL;
    const ggml_tensor * views[16] = { nullptr };
    for (int k = node_idx + 5; k < node_idx + len; ++k) {
        const ggml_tensor * t = cgraph->nodes[k];
        ops[k - node_idx] = t->op;
        if (t->op == GGML_OP_VIEW) {
            if (nv >= n_used || t->view_src != mul || t->ne[0] != md || t->ne[1] != n_tok || t->nb[1] != mul->nb[2] ||
                (const char *) t->data != (const char *) mul->data + nv*mul->nb[1]) {
                return 0;
            }
            views[nv++] = t;
        } else if (t->op == GGML_OP_ADD) {
            const ggml_tensor * lhs = na == 0 ? views[0] : acc;
            if (lhs == nullptr || t->src[0] != lhs || t->src[1] != views[na + 1] || views[na + 1] == nullptr ||
                t->type != GGML_TYPE_F32) {
                return 0;
            }
            acc = t;
            na++;
        } else {
            return 0;
        }
    }
    if (nv != n_used || na != n_used - 1) {
        return 0;
    }
    ggml_tensor * out = cgraph->nodes[node_idx + len - 1];
    if (out->ne[0] != md || out->ne[1] != n_tok || out->nb[0] != sizeof(float) || out->type != GGML_TYPE_F32) {
        return 0;
    }
    const int out_idx = node_idx + len - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, len, ops, &out_idx, 1)) {
        return 0;
    }

    const int n_pairs = n_used*n_tok;
    const int xne1    = (int) x->ne[1];
    const int nb0     = (int)(ids->nb[0]/sizeof(int32_t));
    const int nb1     = (int)(ids->nb[1]/sizeof(int32_t));
    const int32_t * idsd = (const int32_t *) ids->data;
    cudaStream_t stream = ctx.stream();

    mach1_d4_hash hash;
    memset(&hash, 0, sizeof(hash));
    if (ggml_get_op_params_i32(gate, 15) == 1) {
        for (int i = 0; i < 15; ++i) {
            hash.v[i] = ggml_get_op_params_i32(gate, i);
        }
    }

    ggml_cuda_pool_alloc<float> ug(ctx.pool(), (size_t) n_pairs*n);
    ggml_cuda_pool_alloc<float> uu(ctx.pool(), (size_t) n_pairs*n);
    ggml_cuda_pool_alloc<float> pg(ctx.pool(), (size_t) n_pairs*mff);
    ggml_cuda_pool_alloc<float> pu(ctx.pool(), (size_t) n_pairs*mff);
    ggml_cuda_pool_alloc<float> ud(ctx.pool(), (size_t) n_pairs*mff);
    ggml_cuda_pool_alloc<float> pd(ctx.pool(), (size_t) n_pairs*md);

    const bool tab = hash.v[0] == 0;
    const bool cmp = !tab && mach1_d4_cmp_ok(hash);
    static const int w16_env = getenv("GGML_MACH1_D4_W16") == nullptr ? -1 : atoi(getenv("GGML_MACH1_D4_W16"));
    const auto walk = [&](const ggml_tensor * t, const float * u, float * pout, int nn, int mm) {
        const dim3 grid((unsigned)(mm/16), (unsigned) n_pairs, 1);
        const bool w16 = nn/16 >= 32 && (w16_env > 0 || (w16_env < 0 && tab));
        (tab ? (w16 ? mach1_d4_walk_kernel<true, 16, true> : mach1_d4_walk_kernel<true, 8, true>)
             : (w16 ? mach1_d4_walk_kernel<true, 16>       : cmp ? mach1_d4_walk_kernel<true, 8, false, true> : mach1_d4_walk_kernel<true>))
            <<<grid, w16 ? 512 : MACH1_D4_WG, nn*sizeof(float), stream>>>(
            (const uint16_t *) t->src[0]->data, (const int32_t *) t->src[1]->data, (const float *) t->src[4]->data,
            (const uint16_t *) t->src[5]->data, (const float *) t->src[6]->data, idsd, nullptr, u, pout,
            nn, mm/16, nn/16, n_used, (int) t->src[1]->ne[1], nb0, nb1, 0, 1, hash, nullptr, nullptr, nullptr, nullptr, nullptr);
    };

    mach1_d4_ustage2_kernel<<<dim3((unsigned) n_pairs, 2, 1), MACH1_D4_WG, n*sizeof(float), stream>>>(
        (const float *) x->data, (const half *) gate->src[2]->data, (const half *) up->src[2]->data, idsd,
        ug.get(), uu.get(), n, n_used, xne1, nb0, nb1);
    walk(gate, ug.get(), pg.get(), n, mff);
    walk(up,   uu.get(), pu.get(), n, mff);
    mach1_d4_glu_kernel<<<n_pairs, MACH1_D4_WG, 2*mff*sizeof(float), stream>>>(
        pg.get(), pu.get(), (const half *) gate->src[3]->data, (const half *) up->src[3]->data,
        (const half *) down->src[2]->data, idsd, ud.get(), mff, n_used, nb0, nb1);
    walk(down, ud.get(), pd.get(), mff, md);
    int * cnt = n_tok <= 64 ? mach1_d4_counters(ctx.device) : nullptr;
    if (cnt != nullptr) {
        ggml_cuda_pool_alloc<float> wb(ctx.pool(), (size_t) n_pairs*md);
        mach1_d4_redwsum_kernel<<<n_pairs, MACH1_D4_WG, md*sizeof(float), stream>>>(
            pd.get(), (const half *) down->src[3]->data, idsd, (const float *) weights->data, wb.get(),
            (float *) out->data, cnt, md, n_used, nb0, nb1, (int64_t)(out->nb[1]/sizeof(float)));
    } else {
        mach1_d4_redsum_kernel<<<n_tok, MACH1_D4_WG, md*sizeof(float), stream>>>(
            pd.get(), (const half *) down->src[3]->data, idsd, (const float *) weights->data, (float *) out->data,
            md, n_used, nb0, nb1, (int64_t)(out->nb[1]/sizeof(float)));
    }
    CUDA_CHECK(cudaGetLastError());
    return len - 1;
}

#endif

#else

bool ggml_cuda_mach1_d4_supported(const ggml_tensor *) {
    return false;
}

void ggml_cuda_op_mach1_d4_mm(ggml_backend_cuda_context &, ggml_tensor * dst) {
    GGML_LOG_ERROR("%s: mach1 ops need CUDA\n", ggml_op_desc(dst));
}

int ggml_cuda_mach1_d4_ffn_fuse(ggml_backend_cuda_context &, const ggml_cgraph *, int) {
    return 0;
}

int ggml_cuda_mach1_d4_pair_fuse(ggml_backend_cuda_context &, const ggml_cgraph *, int) {
    return 0;
}

#endif
