
#pragma once

#include "mach1-pdl.cuh"

#define MACH1_RT_HADR_MAX 12288

#ifndef MACH1_RT_DYN_SMEM
#define MACH1_RT_DYN_SMEM(T, name) extern __shared__ T name[]
#endif

__constant__ uint32_t mach1_hadr_m12[12] = {
    4094u, 1140u, 2280u, 466u, 932u, 1864u, 3728u, 3362u, 2630u, 1166u, 2332u, 570u,
};
__constant__ uint32_t mach1_hadr_m20[20] = {
    1048574u, 398820u, 797640u, 546706u, 44838u, 89676u, 179352u, 358704u, 717408u, 386242u,
    772484u, 496394u, 992788u, 937002u, 825430u, 602286u, 155998u, 311996u, 623992u, 199410u,
};

template <int R, bool TR>
static __device__ __forceinline__ void mach1_hadr_radix_col(float * sh, const int M, const int b,
                                                            const float sc) {
    const uint32_t * H = R == 12 ? mach1_hadr_m12 : mach1_hadr_m20;
    float col[R];
#pragma unroll
    for (int c = 0; c < R; ++c) {
        col[c] = sh[c*M + b];
    }
    float out[R];
#pragma unroll
    for (int a = 0; a < R; ++a) {
        float acc = 0.0f;
#pragma unroll
        for (int c = 0; c < R; ++c) {
            const bool neg = TR ? ((H[c] >> a) & 1u) != 0u : ((H[a] >> c) & 1u) != 0u;
            acc = neg ? acc - col[c] : acc + col[c];
        }
        out[a] = __fdiv_rn(acc, sc);
    }
#pragma unroll
    for (int a = 0; a < R; ++a) {
        sh[a*M + b] = out[a];
    }
}

template <int R, bool TR, int WG>
static __device__ __forceinline__ void mach1_hadr_radix_par(float * sh, const int d, const int M,
                                                            const int tid, const float sc) {
    constexpr int KMAX = (MACH1_RT_HADR_MAX + WG - 1)/WG;
    const uint32_t * H = R == 12 ? mach1_hadr_m12 : mach1_hadr_m20;
    const int lm = 31 - __clz(M);
    float o[KMAX];
#pragma unroll
    for (int k = 0; k < KMAX; ++k) {
        if (k*WG >= d) {
            break;
        }
        const int i = tid + k*WG;
        if (i < d) {
            const int a = i >> lm;
            const int b = i & (M - 1);
            const uint32_t row = TR ? 0u : H[a];
            float acc = 0.0f;
#pragma unroll
            for (int c = 0; c < R; ++c) {
                const float v   = sh[c*M + b];
                const bool  neg = TR ? ((H[c] >> a) & 1u) != 0u : ((row >> c) & 1u) != 0u;
                acc = neg ? acc - v : acc + v;
            }
            o[k] = __fdiv_rn(acc, sc);
        }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < KMAX; ++k) {
        if (k*WG >= d) {
            break;
        }
        const int i = tid + k*WG;
        if (i < d) {
            sh[i] = o[k];
        }
    }
}

template <int WG, bool TR, bool PAR = false>
static __device__ void mach1_hadr_block(float * sh, const int d, const int r, const int tid) {
    const int M = d / r;
    int ls = 0;
    for (int span = 1; span < M; span <<= 1, ++ls) {
        for (int b = tid; b < d/2; b += WG) {
            const int   base = ((b >> ls) << (ls + 1)) + (b & (span - 1));
            const float a0   = sh[base];
            const float a1   = sh[base + span];
            sh[base]        = a0 + a1;
            sh[base + span] = a0 - a1;
        }
        __syncthreads();
    }
    const float sc = __fsqrt_rn((float) d);
    if (PAR) {
        if (r == 12) {
            mach1_hadr_radix_par<12, TR, WG>(sh, d, M, tid, sc);
        } else {
            mach1_hadr_radix_par<20, TR, WG>(sh, d, M, tid, sc);
        }
        __syncthreads();
        return;
    }
    for (int b = tid; b < M; b += WG) {
        if (r == 12) {
            mach1_hadr_radix_col<12, TR>(sh, M, b, sc);
        } else {
            mach1_hadr_radix_col<20, TR>(sh, M, b, sc);
        }
    }
    __syncthreads();
}

template <int WG, bool PAR = false>
__global__ void __launch_bounds__(WG) mach1_rt_u_hadr_kernel(
        const float * __restrict__ su,
        const float * __restrict__ x,
        float       * __restrict__ scr_u,
        const int n, const int r) {
    MACH1_RT_DYN_SMEM(float, sh);

    const int t   = blockIdx.x;
    const int tid = threadIdx.x;

    for (int i = tid; i < n; i += WG) {
        sh[i] = su[i] * x[(int64_t) t*n + i];
    }
    __syncthreads();
    mach1_hadr_block<WG, true, PAR>(sh, n, r, tid);
    for (int i = tid; i < n; i += WG) {
        scr_u[(int64_t) t*n + i] = sh[i];
    }
}

template <int WG, bool PAR = false>
__global__ void __launch_bounds__(WG) mach1_rt_out_hadr_kernel(
        const float * __restrict__ sv,
        const float * __restrict__ scr_v,
        float       * __restrict__ dst,
        const int m, const int r,
        const int nsplit = 1, const int64_t sstride = 0) {
    MACH1_RT_DYN_SMEM(float, sh);

    const int t   = blockIdx.x;
    const int tid = threadIdx.x;

    for (int i = tid; i < m; i += WG) {
        float v = scr_v[(int64_t) t*m + i];
        for (int k = 1; k < nsplit; ++k) {
            v += scr_v[k*sstride + (int64_t) t*m + i];
        }
        sh[i] = v;
    }
    __syncthreads();
    mach1_hadr_block<WG, false, PAR>(sh, m, r, tid);
    for (int i = tid; i < m; i += WG) {
        dst[(int64_t) t*m + i] = sh[i] * sv[i];
    }
}

template <int E>
static __device__ __forceinline__ void mach1_hadr_warp_fwht(float * v, const int M, const int lane) {
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

template <int E>
static __device__ __forceinline__ void mach1_hadr_warp_fwht_t(float * v, const int lane) {
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
}

template <int E>
static __device__ __forceinline__ void mach1_hadr_warp_fwht_v4_reg(float * v, const int span) {
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

template <int E>
static __device__ __forceinline__ void mach1_hadr_warp_fwht_v4(float * v, const int lane) {
    static_assert(E % 4 == 0, "V4 holds whole float4 rows");
    mach1_hadr_warp_fwht_v4_reg<E>(v, 1);
    mach1_hadr_warp_fwht_v4_reg<E>(v, 2);
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
    for (int span = 4; span < E; span <<= 1) {
        mach1_hadr_warp_fwht_v4_reg<E>(v, span);
    }
}

static __device__ __forceinline__ void mach1_hadr_ld4(float * v, const float * p) {
    const float4 w = *(const float4 *) p;
    v[0] = w.x; v[1] = w.y; v[2] = w.z; v[3] = w.w;
}

template <int R, bool TR, int WG, bool OUT>
static __device__ __forceinline__ void mach1_hadr_radix_par_st(const float * sh, const int ds, const int P,
                                                               const int tid, const float sc, float * __restrict__ y,
                                                               const float * __restrict__ sv, const int M, const int p0,
                                                               const float * sp, const int * perm, const int * pp,
                                                               const uint32_t * hr = nullptr) {
    constexpr int KMAX = (MACH1_RT_HADR_MAX + WG - 1)/WG;
    const uint32_t * H = R == 12 ? mach1_hadr_m12 : mach1_hadr_m20;
    const int lp = 31 - __clz(P);
#pragma unroll 1
    for (int k = 0; k < KMAX; ++k) {
        if (k*WG >= ds) {
            break;
        }
        const int i = tid + k*WG;
        if (i < ds) {
            const int a = i >> lp;
            const int b = i & (P - 1);
            const int j = a*M + p0 + b;
            const float s = OUT ? (k == 0 ? sp[0] : k == 1 ? sp[1] : sv[j]) : 0.0f;
            const bool  pre = hr != nullptr && k < 2;
            const uint32_t row = pre ? (k == 0 ? hr[0] : hr[1]) : TR ? 0u : H[a];
            float acc = 0.0f;
#pragma unroll
            for (int c = 0; c < R; ++c) {
                const float v   = sh[c*P + b];
                const bool  neg = TR && !pre ? ((H[c] >> a) & 1u) != 0u : ((row >> c) & 1u) != 0u;
                acc = neg ? acc - v : acc + v;
            }
            const float o = __fdiv_rn(acc, sc);
            if (OUT && perm != nullptr) {
                y[k == 0 ? pp[0] : k == 1 ? pp[1] : perm[j]] = o * s;
            } else {
                y[j] = OUT ? o * s : o;
            }
        }
    }
}

template <bool TR>
static __device__ __forceinline__ uint32_t mach1_hadr_sign_row(const int r, const int a) {
    const uint32_t * H = r == 12 ? mach1_hadr_m12 : mach1_hadr_m20;
    if (!TR) {
        return H[a];
    }
    uint32_t row = 0u;
    for (int c = 0; c < r; ++c) {
        row |= ((H[c] >> a) & 1u) << c;
    }
    return row;
}

template <int WG, int E, bool OUT, bool GATE = false, bool V4 = false>
static __device__ __forceinline__ void mach1_hadr_warp_tl(const float * a, const float * b, float * dst, const int d,
                                                          const int r, const int nsplit, const int64_t sstride,
                                                          float * sh, const int * perm,
                                                          const float * gs = nullptr, const float * go = nullptr) {
    const int   t    = blockIdx.x;
    const int   tid  = threadIdx.x;
    const int   lane = tid & 31;
    const int   c    = tid >> 5;
    const int   M    = r == 12 ? d / 12 : d / 20;
    const int   P    = M >> (31 - __clz((int) gridDim.y));
    const int   p0   = blockIdx.y * P;
    const int   lp   = 31 - __clz(P);
    const int   ds   = r*P;
    const bool  own  = c < r;
    const int   lo   = V4 ? 4*lane : lane;
    const float sc   = __fsqrt_rn((float) d);
    float v[E];
    float sp[2] = {0.0f, 0.0f};
    int   pp[2] = {0, 0};
    if (OUT) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int i = tid + k*WG;
            if (i < ds) {
                const int j = (i >> lp)*M + p0 + (i & (P - 1));
                sp[k] = a[j];
                if (perm != nullptr) {
                    pp[k] = perm[j];
                }
            }
        }
    } else if constexpr (V4) {
#pragma unroll
        for (int e = 0; e < E; ++e) {
            v[e] = 0.0f;
        }
        if (own) {
#pragma unroll
            for (int k = 0; k < E/4; ++k) {
                mach1_hadr_ld4(v + 4*k, a + c*M + lo + 128*k);
            }
        }
    } else {
#pragma unroll
        for (int e = 0; e < E; ++e) {
            v[e] = own ? a[c*M + e*32 + lane] : 0.0f;
        }
    }
    uint32_t hr[2] = {0u, 0u};
    if (OUT) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int i = tid + k*WG;
            if (i < ds) {
                hr[k] = mach1_hadr_sign_row<false>(r, i >> lp);
            }
        }
    }
    mach1_pdl_wait();
    b   = mach1_pdl_dep(b);
    dst = mach1_pdl_dep(dst);
    if constexpr (GATE) {
        gs = mach1_pdl_dep(gs) + c*M + lo;
        go = mach1_pdl_dep(go) + c*M + lo;
        b  = gs;
    }
    const float * bt = b + (int64_t) t*d + c*M + lo;
    if constexpr (GATE && V4) {
        if (own) {
            float g[E], o[E];
#pragma unroll
            for (int k = 0; k < E/4; ++k) {
                mach1_hadr_ld4(g + 4*k, gs + 128*k);
                mach1_hadr_ld4(o + 4*k, go + 128*k);
            }
#pragma unroll
            for (int e = 0; e < E; ++e) {
                v[e] = __fmul_rn(v[e], (1.0f / (1.0f + expf(-g[e]))) * o[e]);
            }
        }
    } else if constexpr (GATE) {
        if (own) {
#pragma unroll
            for (int e = 0; e < E; ++e) {
                v[e] *= (1.0f / (1.0f + expf(-gs[e*32]))) * go[e*32];
            }
        }
    } else if constexpr (V4) {
        if (own) {
            if (OUT) {
#pragma unroll
                for (int q = 0; q < E/4; ++q) {
                    mach1_hadr_ld4(v + 4*q, bt + 128*q);
                }
            }
#pragma unroll 1
            for (int k = OUT ? 1 : 0; k < (OUT ? nsplit : 1); ++k) {
#pragma unroll
                for (int q = 0; q < E/4; ++q) {
                    float w[4];
                    mach1_hadr_ld4(w, bt + k*sstride + 128*q);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        v[4*q + j] = OUT ? v[4*q + j] + w[j] : __fmul_rn(v[4*q + j], w[j]);
                    }
                }
            }
        } else if (OUT) {
#pragma unroll
            for (int e = 0; e < E; ++e) {
                v[e] = 0.0f;
            }
        }
    } else if (OUT) {
#pragma unroll
        for (int e = 0; e < E; ++e) {
            v[e] = own ? bt[e*32] : 0.0f;
        }
        for (int k = 1; k < nsplit; ++k) {
            if (own) {
#pragma unroll
                for (int e = 0; e < E; ++e) {
                    v[e] += bt[k*sstride + e*32];
                }
            }
        }
    } else if (own) {
#pragma unroll
        for (int e = 0; e < E; ++e) {
            v[e] *= bt[e*32];
        }
    }
    if (own) {
        if constexpr (V4) {
            mach1_hadr_warp_fwht_v4<E>(v, lane);
        } else {
            mach1_hadr_warp_fwht_t<E>(v, lane);
        }
#pragma unroll
        for (int e = 0; e < E; ++e) {
            const int q = (V4 ? lo + 128*(e/4) + e%4 : e*32 + lane) - p0;
            if (q >= 0 && q < P) {
                sh[c*P + q] = v[e];
            }
        }
    }
    __syncthreads();
    float * y = dst + (int64_t) t*d;
    if (r == 12) {
        mach1_hadr_radix_par_st<12, !OUT, WG, OUT>(sh, ds, P, tid, sc, y, a, M, p0, sp, perm, pp, OUT ? hr : nullptr);
    } else {
        mach1_hadr_radix_par_st<20, !OUT, WG, OUT>(sh, ds, P, tid, sc, y, a, M, p0, sp, perm, pp, OUT ? hr : nullptr);
    }
}

template <int R>
__global__ void __launch_bounds__(1024) mach1_hadr_glu_in_kernel(
        const float * __restrict__ svg, const float * __restrict__ svu,
        const float * vg, const float * vu, const float * __restrict__ su, float * uo) {
    constexpr int M = 32;
    constexpr int D = R*M;
    __shared__ float shg[D], shu[D], og[D], ou[D];
    const int   tid  = threadIdx.x;
    const int   lane = tid & 31;
    const int   c    = tid >> 5;
    const bool  own  = c < R;
    const float sc   = __fsqrt_rn((float) D);
    mach1_pdl_trigger();
    float spg[2] = {0.0f, 0.0f}, spu[2] = {0.0f, 0.0f}, sin_ = 0.0f;
    if (tid < D) {
        spg[0] = svg[tid];
        spu[0] = svu[tid];
    }
    if (own) {
        sin_ = su[c*M + lane];
    }
    mach1_pdl_wait();
    vg = mach1_pdl_dep(vg);
    vu = mach1_pdl_dep(vu);
    uo = mach1_pdl_dep(uo);
    if (own) {
        float v = vg[c*M + lane];
        mach1_hadr_warp_fwht_t<1>(&v, lane);
        shg[c*M + lane] = v;
        float w = vu[c*M + lane];
        mach1_hadr_warp_fwht_t<1>(&w, lane);
        shu[c*M + lane] = w;
    }
    __syncthreads();
    mach1_hadr_radix_par_st<R, false, 1024, true>(shg, D, M, tid, sc, og, svg, M, 0, spg, nullptr, nullptr);
    mach1_hadr_radix_par_st<R, false, 1024, true>(shu, D, M, tid, sc, ou, svu, M, 0, spu, nullptr, nullptr);
    if (tid < D) {
        const float x = og[tid];
        og[tid] = (x / (1.0f + expf(-x))) * ou[tid];
    }
    __syncthreads();
    if (own) {
        float v = sin_;
        v *= og[c*M + lane];
        mach1_hadr_warp_fwht_t<1>(&v, lane);
        shg[c*M + lane] = v;
    }
    __syncthreads();
    mach1_hadr_radix_par_st<R, true, 1024, false>(shg, D, M, tid, sc, uo, su, M, 0, nullptr, nullptr, nullptr);
}

template <int E, bool V4 = false>
__global__ void __launch_bounds__(1024) mach1_hadr_gated_in_kernel(
        const float * __restrict__ su, const float * gs, const float * go, float * dst, const int d, const int r) {
    MACH1_RT_DYN_SMEM(float, sh);
    mach1_pdl_trigger();
    mach1_hadr_warp_tl<1024, E, false, true, V4>(su, nullptr, dst, d, r, 1, 0, sh, nullptr, gs, go);
}

struct mach1_hadr_xops {
    const float * a[2];
    const float * b[2];
    float       * dst[2];
    int           d[2];
    int           r[2];
};

template <int WG, int E, bool OUT, bool TL = false, bool V4 = false>
__global__ void __launch_bounds__(WG) mach1_rt_hadr_warp_kernel(
        const float * __restrict__ a,
        const float * __restrict__ b,
        float       * __restrict__ dst,
        int d, int r,
        const int nsplit, const int64_t sstride, const mach1_hadr_xops xo, const int * perm) {
    MACH1_RT_DYN_SMEM(float, sh);
    if (blockIdx.z != 0) {
        const int k = blockIdx.z - 1;
        a = k == 0 ? xo.a[0] : xo.a[1]; b = k == 0 ? xo.b[0] : xo.b[1]; dst = k == 0 ? xo.dst[0] : xo.dst[1];
        d = k == 0 ? xo.d[0] : xo.d[1]; r = k == 0 ? xo.r[0] : xo.r[1];
    }
    mach1_pdl_trigger();
    if constexpr (TL) {
        mach1_hadr_warp_tl<WG, E, OUT, false, V4>(a, b, dst, d, r, nsplit, sstride, sh, blockIdx.z == 0 ? perm : nullptr);
        return;
    }
    mach1_pdl_wait();
    b   = mach1_pdl_dep(b);
    dst = mach1_pdl_dep(dst);

    const int t    = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int c    = tid >> 5;
    const int M    = d / r;
    const int S    = gridDim.y;
    const int P    = M / S;
    const int p0   = blockIdx.y * P;
    const bool own = c < r;

    float v[E];
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const int i = c*M + lane*E + e;
        float x = 0.0f;
        if (own) {
            if (OUT) {
                x = b[(int64_t) t*d + i];
                for (int k = 1; k < nsplit; ++k) {
                    x += b[k*sstride + (int64_t) t*d + i];
                }
            } else {
                x = a[i] * b[(int64_t) t*d + i];
            }
        }
        v[e] = x;
    }
    if (own) {
        mach1_hadr_warp_fwht<E>(v, M, lane);
    }
    const int q0 = lane*E - p0;
    if (own && q0 >= 0 && q0 < P) {
#pragma unroll
        for (int e = 0; e < E; ++e) {
            sh[c*P + q0 + e] = v[e];
        }
    }
    __syncthreads();
    const float sc = __fsqrt_rn((float) d);
    const int   ds = r*P;
    if (r == 12) {
        mach1_hadr_radix_par<12, !OUT, WG>(sh, ds, P, tid, sc);
    } else {
        mach1_hadr_radix_par<20, !OUT, WG>(sh, ds, P, tid, sc);
    }
    __syncthreads();
    const int lp = 31 - __clz(P);
    for (int i = tid; i < ds; i += WG) {
        const int ca = i >> lp;
        const int j  = ca*M + p0 + (i & (P - 1));
        dst[(int64_t) t*d + j] = OUT ? sh[i] * a[j] : sh[i];
    }
}

static __device__ __forceinline__ bool mach1_hadr_u_warp_ok(const int n, const int r, const int wg) {
    const int M = n / r;
    return r <= wg/32 && (M == 32 || M == 64 || M == 128);
}

template <int WG, int E>
static __device__ __forceinline__ void mach1_hadr_u_warp_shared_e(const float * __restrict__ su,
                                                                 const float * __restrict__ xr,
                                                                 float * sh, const int n, const int r,
                                                                 const int tid) {
    const int lane = tid & 31;
    const int c    = tid >> 5;
    const int M    = n / r;
    const bool own = c < r;
    float v[E];
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const int i = c*M + lane*E + e;
        v[e] = own ? su[i] * xr[i] : 0.0f;
    }
    mach1_hadr_warp_fwht<E>(v, M, lane);
    if (own) {
#pragma unroll
        for (int e = 0; e < E; ++e) {
            sh[c*M + lane*E + e] = v[e];
        }
    }
    __syncthreads();
    const float sc = __fsqrt_rn((float) n);
    if (r == 12) {
        mach1_hadr_radix_par<12, true, WG>(sh, n, M, tid, sc);
    } else {
        mach1_hadr_radix_par<20, true, WG>(sh, n, M, tid, sc);
    }
}

template <int WG>
static __device__ __forceinline__ void mach1_hadr_u_warp_shared(const float * __restrict__ su,
                                                               const float * __restrict__ xr,
                                                               float * sh, const int n, const int r,
                                                               const int tid) {
    switch (n / r) {
        case 32:  mach1_hadr_u_warp_shared_e<WG, 1>(su, xr, sh, n, r, tid); break;
        case 64:  mach1_hadr_u_warp_shared_e<WG, 2>(su, xr, sh, n, r, tid); break;
        default:  mach1_hadr_u_warp_shared_e<WG, 4>(su, xr, sh, n, r, tid); break;
    }
}

template <int WG>
__global__ void mach1_rt_walk_kernel(
        const uint16_t * __restrict__ trellis,
        const half     * __restrict__ tlut,
        const float    * __restrict__ scr_u,
        float          * __restrict__ scr_v,
        const int m, const int n) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = WG / warp_size;
    __shared__ float red[16][8];
    __shared__ float slut[1024];

    const int t   = blockIdx.z;
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;

    constexpr int words = 64;
    const int tiles_y = n / 16;

    for (int i = tid; i < 1024; i += WG) {
        slut[i] = __half2float(tlut[i]);
    }
    __syncthreads();

    float partial[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        partial[i] = 0.0f;
    }

    for (int ct = tid; ct < tiles_y; ct += WG) {
        const int64_t tw = ((int64_t) tr*tiles_y + ct)*words;
        const float * ub = scr_u + (int64_t) t*n + ct*16;

        float uu[16];
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            uu[c] = ub[c];
        }

        for (int ri = 0; ri < 16; ++ri) {
            uint32_t w[5];
#pragma unroll
            for (int q = 0; q < 5; ++q) {
                w[q] = trellis[tw + ((4*ri + q) & 63)];
            }
            uint32_t ph[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t hi = w[j >> 1];
                const uint32_t st = (j & 1) == 0 ? hi
                    : (((hi << 8) | (w[(j >> 1) + 1] >> 8)) & 0xFFFFu);
                ph[j] = st*(st + 1u);
            }
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t row = (ph[j] >> 6) & 511u;
                float a0 = slut[2*row + 0];
                const float a1 = slut[2*row + 1];
                if (ph[j] & 0x8000u) {
                    a0 = -a0;
                }
                acc += a0*uu[2*j] + a1*uu[2*j + 1];
            }
            partial[ri] = ct == tid ? acc : partial[ri] + acc;
        }
    }

    const int lane = tid % warp_size;
    const int wid  = tid / warp_size;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float s = warp_reduce_sum<warp_size>(partial[i]);
        if (lane == 0) {
            red[i][wid] = s;
        }
    }
    __syncthreads();
    if (tid < 16) {
        float sum = 0.0f;
#pragma unroll
        for (int wj = 0; wj < n_warps; ++wj) {
            sum += red[tid][wj];
        }
        scr_v[(int64_t) t*m + tr*16 + tid] = sum;
    }
}
