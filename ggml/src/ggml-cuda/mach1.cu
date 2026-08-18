#include "common.cuh"
#include "cp-async.cuh"
#include "mach1.cuh"
#include "getrows.cuh"
#include "mmvf.cuh"
#include "mmvq.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"
#include "ggml-backend-impl.h"
#include <cooperative_groups.h>

// CUDA port of the 7 Mach-1 codec ops. Numeric source of truth is the CPU
// implementation (ggml-cpu/ops.cpp); the GPU decomposition mirrors the Vulkan
// shaders (ggml-vulkan/vulkan-shaders/mach1_*.comp) 1:1. The build compiles
// with -use_fast_math, so the documented two-rounding sites are pinned with
// _rn intrinsics (embed_rows q*step, exp_basis B-loop, exp_walk V2 fp16 round,
// every FWHT final scale); dot accumulations stay contractible on purpose.
//
// Prefill additionally has dense-materialize paths (mirroring the mach
// engine's expert_forward_prefill_dense / NeCodesLinear.dense split): instead
// of re-decoding the trellis per (pair, token), the walk stage is replaced by
// one decode of hatWr into a transient fp16 bank plus a plain fp32 apply.
// Because the Hadamards here act on ACTIVATIONS (exp_u/exp_out/rt_u/rt_out),
// the bank holds ONLY the fp16-rounded LUT values — no FWHT on weights. The
// bank values are the exact halves the fused walk multiplies (V2 pins the
// fp16 round; V8/rt tluts are pre-rounded F16), so the dense paths differ
// from the fused ones only in contractible fp32 summation order (and, for
// V8, in the association of the wave-gamma product — also contractible).
//   GGML_MACH1_DENSE_MIN     (default 1024): EXP_MM pairs (n_used*n_tok)
//                            at/above which the dense path runs.
//   GGML_MACH1_DENSE_MIN_TOK (default 4): RT_MM token count at/above which
//                            the dense path runs.
// Both are read once and cached; set =1 to force the dense paths in tests.

#define GGML_CUDA_MACH1_DEM_FLAG 0x40000000u

// dense-path thresholds, read once (host)
static int mach1_env_int(const char * name, const int def) {
    const char * s = getenv(name);
    return s != nullptr ? atoi(s) : def;
}

// Kernel pointer parameters use plain __restrict__, NOT GGML_CUDA_RESTRICT.
// That macro is __CUDA_ARCH__-dependent (empty under PDL on Hopper, see
// llama.cpp#24030), and __CUDA_ARCH__ is defined only in the device pass — so in
// a kernel SIGNATURE it expands differently on the host and device passes and
// nvcc emits a device stub that cannot match its own declaration. For templated
// kernels that is a hard sm_90 build error. Upstream only ever applies the macro
// to local variables. mach1 opts out of PDL below, so the conflict the macro
// guards against cannot arise here and the raw qualifier is both safe and
// arch-consistent. Do not reintroduce the macro into these signatures.
// mach1 kernels are launched with plain stream launches, NOT via
// ggml_cuda_kernel_launch: the multi-stage mach1 ops have true RAW
// dependencies through global pool scratch between consecutive kernels, and
// under programmatic dependent launch (PDL) on Hopper (H100) large prefills
// (e.g. llama-bench -p 1024) crash within seconds, while GGML_CUDA_PDL=0 is
// stable. Plain stream launches are always correctly ordered, so mach1 opts
// out of PDL entirely, and the kernels carry no ggml_cuda_pdl_sync() calls:
// they were no-ops here (grid dependencies are already satisfied at launch
// without the PDL attribute), and on sm_90 they expand to
// cudaGridDependencySynchronize(), which makes nvcc emit a device stub that
// does not match the declaration for every TEMPLATED __global__ in this file
// ("__wrapper__device_stub_... does not match any template declaration").
// That broke the Hopper-native build outright, so they are gone.
template <typename Kernel, typename... Args>
static void mach1_launch(Kernel kernel, const ggml_cuda_kernel_launch_params & p, Args &&... args) {
    kernel<<<p.block_nums, p.block_dims, p.shmem, p.stream>>>(std::forward<Args>(args)...);
    CUDA_CHECK(cudaGetLastError());
}

// --- measurement-only instrumentation (default off, no numeric effect) ---
// GGML_MACH1_TIME=1: wall-clock every mach1 kernel launch (stream sync before
// and after) keyed by stage + shape; totals dump to stderr at process exit.
// Do NOT combine with CUDA graph capture (set GGML_CUDA_DISABLE_GRAPHS=1).
// GGML_MACH1_DEBUG=1: one-time diagnostics (p4 repack status).
#include <chrono>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

static bool mach1_time_on() {
    // exactly 1: per-kernel mach1 timing. >= 2 is the whole-graph per-OP timer
    // in ggml-cuda.cu; the inner per-kernel syncs would double-instrument
    // mach1 ops there and make them incomparable to other op classes.
    static const bool on = mach1_env_int("GGML_MACH1_TIME", 0) == 1;
    return on;
}

static bool mach1_debug_on() {
    static const bool on = mach1_env_int("GGML_MACH1_DEBUG", 0) != 0;
    return on;
}

// GGML_MACH1_NT32=1: admit decode batches in (16, 32] into the fused nt
// stacks (expert mega QONCE with the widened counter layout, sibling qkvz/
// qkv/shexp batching through the TT/TC batch exec, head 16-token halves,
// GDN prep). Default off; nt == 16 keeps the certified exact-nt16 paths
// bit-for-bit (every extension below is an additional admission, never a
// change to the 16 branch).
static bool mach1_nt32_on() {
    static const bool on = mach1_env_int("GGML_MACH1_NT32", 0) != 0;
    return on;
}

// GGML_MACH1_NTLO=1: admit decode batches in [2, 16) into the fused nt
// stacks that today require exactly 16. QONCE mega and GDN_SGF take any
// nt >= 2 (their pair/seq structure is runtime); the 16-token MMA spines,
// sibling batching and head serve nt > 4 through zero-padded scratch rows
// (nt <= 4 stays on the tuned TT-batch/fork regions). Default off; nt == 16
// keeps the certified exact-nt16 paths - every extension is an additional
// admission, never a change to the 16 branch.
static bool mach1_ntlo_on() {
    static const bool on = [] {
        const bool v = mach1_env_int("GGML_MACH1_NTLO", 0) != 0;
        if (v && mach1_debug_on()) {
            fprintf(stderr, "mach1: NTLO master switch ON\n");
        }
        return v;
    }();
    return on;
}

// GGML_MACH1_NTLO_MIN (default 5): smallest nt admitted into the padded
// 16-token MMA forms (per-op spine, qkvz pair, qkv trio). 5 keeps nt <= 4 on
// the tuned TT-batch regions; 2 admits everything ntlo covers. The head and
// the shexp gate-up batch keep their own > 4 bounds - one warp-head unit
// already matches the padded launch price at nt <= 4, and the fused shexp
// region at nt <= 4 rides the mega fork, which the sibling batch would lose.
static int mach1_ntlo_min() {
    static const int v = std::max(2, std::min(16, mach1_env_int("GGML_MACH1_NTLO_MIN", 5)));
    return v;
}

// GGML_MACH1_NTLOW=1: admit decode batches BELOW 16 into the same fused nt
// stacks the exact-16 gates own today (expert mega QONCE, rt/head/sibling MMA
// spines, GDN state fusion). The audited 16-token kernels take the token-tile
// height as a template parameter - 16 is the certified instantiation, 8/4/2
// the new ones - and a batch is served as a sequence of {16,8,4,2}-token
// chunks at plain pointer offsets. Default off; nt == 16 remains ONE 16-chunk
// at zero offset, so its instantiation and launch bytes are unchanged.
static bool mach1_ntlow_on() {
    // GGML_MACH1_NTLO (the campaign line's env for the same downward widening)
    // is honored as an alias so the merged tree engages the chunked forms
    // under either name - the goal stacks set NTLO.
    static const bool on = mach1_env_int("GGML_MACH1_NTLOW", 0) != 0 ||
                           mach1_env_int("GGML_MACH1_NTLO", 0) != 0;
    return on;
}

// GGML_MACH1_NTLOW_MIN: narrowest batch NTLOW admits (2 = every even width).
// The fused stacks carry a per-step cost that is FIXED in the batch size (the
// spine decodes the dense weights once per step however many tokens ride it),
// so the rung pays off from some width upward; this knob races that boundary
// without touching the kernels. Chunk sizes are unaffected - it filters batch
// widths only.
static int mach1_ntlow_min() {
    // default 8: the chunkab-wall-s4 A/B measured the chunk-2/4 MMA forms
    // losing to the walk paths at nt 2-4 (B2 236 -> 273 t/s, B4 378 -> 397
    // same-draw; rung 2a's padded-MMA kill at chunk granularity). Chunks pay
    // from nt=8 up; the walk owns below. Env overrides for re-racing.
    static const int v = std::max(2, mach1_env_int("GGML_MACH1_NTLOW_MIN", 8));
    return v;
}

// widths NTLOW opens below the certified 16
static bool mach1_ntlow_ok(const int nt) {
    return mach1_ntlow_on() && nt >= mach1_ntlow_min() && nt < 16;
}

// tile height of the next chunk of a token batch; every even nt is tiled
// exactly by this greedy {16,8,4,2} decomposition
static __forceinline__ int mach1_nt_chunk(const int rem) {
    return rem >= 16 ? 16 : (rem >= 8 ? 8 : (rem >= 4 ? 4 : 2));
}

// token widths the chunked 16-token stacks serve. nt == 16 (and nt == 32 under
// NT32) are the pre-existing admissions; NTLOW adds the even widths below 16,
// and the two composed admit every even width up to 32 (24 = 16 + 8).
static bool mach1_nt_ok(const int nt) {
    if (nt == 16) {
        return true;
    }
    if (nt == 32 && mach1_nt32_on()) {
        return true;
    }
    if (!mach1_ntlow_on() || nt < 2 || (nt % 2) != 0) {
        return false;
    }
    return nt < 16 ? nt >= mach1_ntlow_min() : (nt <= 32 && mach1_nt32_on());
}

// GGML_MACH1_PPLOW=1: admit the PREFILL lanes below their certified nt >= 256
// window (rt bank apply as a cuBLAS GEMM, the expert s8/fp16 MMA applies, the
// grouped-GEMM expert apply, the skinny stock-op GEMM). The kernels are token
// count generic - the 256 floor is an amortization threshold, not a shape
// constraint - so widening it is an admission change only.
static bool mach1_pplow_on() {
    static const bool on = mach1_env_int("GGML_MACH1_PPLOW", 0) != 0;
    return on;
}

// GGML_MACH1_PPLOW_MIN: narrowest ubatch the PP lanes are admitted at. The
// floor is 64: every fused DECODE width tops out at 32 (MEGA_NT and the nt32
// rung both clamp there), so no decode shape can ever reach a PP lane through
// this knob, on or off.
static int mach1_pp_min() {
    static const int v = mach1_pplow_on()
        ? std::max(64, mach1_env_int("GGML_MACH1_PPLOW_MIN", 128)) : 256;
    return v;
}

// GGML_MACH1_PPLOW_GG_MIN: the grouped-GEMM lane's own floor. GG fetches the
// per-group pair counts D2H and blocks the host on them once per call, and
// that sync is a fixed cost the ubatch has to amortize - it is the one PP lane
// whose floor is structural rather than an amortization guess, so it keeps a
// separate knob (never below PPLOW_MIN).
static int mach1_pp_gg_min() {
    static const int v = mach1_pplow_on()
        ? std::max(mach1_pp_min(), mach1_env_int("GGML_MACH1_PPLOW_GG_MIN", 256)) : 256;
    return v;
}

// one line per lane the first time it takes a ubatch below the certified floor
static void mach1_pplow_mark(const char * lane, const int nt) {
    if (nt >= 256 || !mach1_debug_on()) {
        return;
    }
    static std::mutex mu;
    static std::set<std::string> seen;
    std::lock_guard<std::mutex> lk(mu);
    if (seen.insert(lane).second) {
        fprintf(stderr, "mach1: PPLOW %s ENGAGED at nt=%d\n", lane, nt);
    }
}

// GGML_MACH1_RT_STDOUT: standard-output-basis epilogues - the decoder's rows
// are treated as already standard-basis, so the output Hadamard and 1/sqrt(m)
// disappear and sv acts as a direct row scale. Current payloads are in the
// rotated basis, so OUTPUT IS GARBAGE until a re-encoded artifact passes the
// selector quality gate; the switch therefore refuses to engage without
// GGML_MACH1_TIME=1 and stays a timing instrument, never a model route.
static bool mach1_rt_stdout_on() {
    static const bool on = [] {
        const bool want = mach1_env_int("GGML_MACH1_RT_STDOUT", 0) != 0;
        if (want && !mach1_time_on()) {
            fprintf(stderr, "mach1: RT_STDOUT REFUSED (wrong-basis payloads; set GGML_MACH1_TIME=1 for the timing probe)\n");
            return false;
        }
        if (want) {
            fprintf(stderr, "mach1: standard-output row-scale epilogues ENGAGED (TIMING_ONLY WRONG_BASIS NOT_QUALITY)\n");
        }
        return want;
    }();
    return on;
}

// GGML_MACH1_ABLATE: MEASUREMENT ONLY — skip individual stages to attribute
// wall-clock per stage from end-to-end tok/s. OUTPUT IS GARBAGE when non-zero;
// this exists because ncu (GPU perf counters) and nsys are both unavailable in
// the test container, and per-launch stream syncs inflate small kernels beyond
// use. Bits: 1 rt_u, 2 rt_walk/apply, 4 rt_out, 8 exp_group, 16 exp_u,
// 32 exp_walk/apply, 64 exp_out, 128 head_mm.
enum mach1_ablate_bit {
    MACH1_ABLATE_RT_U     = 1,
    MACH1_ABLATE_RT_WALK  = 2,
    MACH1_ABLATE_RT_OUT   = 4,
    MACH1_ABLATE_EXP_GRP  = 8,
    MACH1_ABLATE_EXP_U    = 16,
    MACH1_ABLATE_EXP_WALK = 32,
    MACH1_ABLATE_EXP_OUT  = 64,
    MACH1_ABLATE_HEAD     = 128,
};

static bool mach1_ablate(int bit) {
    static const int mask = mach1_env_int("GGML_MACH1_ABLATE", 0);
    return (mask & bit) != 0;
}

struct mach1_time_table {
    std::mutex mtx;
    std::map<std::string, std::pair<long long, double>> acc;   // key -> (calls, total us)
    ~mach1_time_table() {
        double tot = 0.0;
        for (const auto & kv : acc) {
            tot += kv.second.second;
        }
        fprintf(stderr, "mach1-time: TOTAL %.1f ms across timed mach1 kernels\n", tot/1000.0);
        for (const auto & kv : acc) {
            fprintf(stderr, "mach1-time: %-64s calls=%7lld total=%10.1f us avg=%9.2f us\n",
                    kv.first.c_str(), kv.second.first, kv.second.second,
                    kv.second.second/(double) kv.second.first);
        }
    }
};

static mach1_time_table & mach1_times() {
    static mach1_time_table t;
    return t;
}

template <typename F>
static void mach1_timed(cudaStream_t stream, const std::string & key, F && launch) {
    if (!mach1_time_on()) {
        launch();
        return;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto t0 = std::chrono::steady_clock::now();
    launch();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - t0).count();
    mach1_time_table & tt = mach1_times();
    std::lock_guard<std::mutex> lock(tt.mtx);
    auto & e = tt.acc[key];
    e.first  += 1;
    e.second += us;
}

// --- persistent decoded-weight banks ---
//
// The packed trellis is the most bandwidth-efficient form of the weights, but
// decoding it is ALU-bound, and at batch 1 every token re-decodes the same
// values. Measured: an H200 has ~16x an L4's memory bandwidth and delivers only
// ~1.6x the decode rate (60.5 vs 38.6 tok/s), i.e. the walk kernels sit far
// below the bandwidth roof while the ALUs saturate. Where there is spare
// bandwidth and spare VRAM, decoding a tensor ONCE into the fp16 bank the
// prefill path already builds — and keeping it — converts that per-token ALU
// cost into a plain read.
//
// This is bit-exact: tluts are pre-rounded fp16 and the decode spec rounds every
// gathered value to fp16, so a bank holds exactly what the walk would produce.
// The apply path differs from the walk only in contractible fp32 summation
// order, which is the same difference the prefill dense path already carries.
//
// A bank is STRICTLY LARGER than the packed payload it replaces (fp16 vs ~4
// bits/weight for K4), so on a bandwidth-poor part this is a net loss.
//
// MEASURED (H200, Mach-1-Additive-35B): banking the whole rt spine — 206
// tensors, 2.16 GiB, so every per-token trellis walk on the spine is gone —
// moves decode 61.6 -> 62.5 tok/s and prefill 1355 -> 1379. That is noise on
// decode. The batch-1 bottleneck is NOT the cost of decoding weights: a token
// issues ~1231 mach1 kernel launches, so at ~16 ms/token each averages ~13 us
// for work that is often sub-microsecond. Latency per kernel dominates, and
// removing ALU from kernels that were never ALU-bound buys nothing. Fusing the
// per-op kernel chains is the lever; this is kept for the small prefill gain
// and because a fused kernel can read a bank instead of walking.
//
// DEFAULT OFF for that reason (and because of the lifetime caveat below).
//
//   GGML_MACH1_BANK=1            opt in (default: off)
//   GGML_MACH1_BANK_GB=N         cap total bank VRAM (default: 4 GiB)
//   GGML_MACH1_BANK_RESERVE_GB=N device memory a bank must leave free (8)
//
// The cap is a FIXED number, not a free-VRAM reading. Sizing it from
// cudaMemGetInfo at first use measures the device before the KV cache, the
// compute buffers and the pool have finished growing, so it hands out a
// budget that is already stale when it is spent. The reserve is the live
// half: it is re-read per bank, so a context that grows later still wins.
// Every miss returns nullptr and the caller keeps its decode path.
//
// LIFETIME: banks are keyed by the weight tensor's device pointer and live for
// the process. Freeing a model and loading another inside one process can reuse
// the same address, which would hand back a stale bank — set GGML_MACH1_BANK=0
// for a model-swapping server until banks are tied to buffer teardown.
struct mach1_bank_cache {
    std::mutex mtx;
    std::unordered_map<const void *, half *> banks;
    size_t used    = 0;
    size_t budget  = 0;
    size_t reserve = 0;
    bool   ready   = false;
    // Arena sub-allocation (GGML_MACH1_ARENA=1, default OFF): one cudaMalloc
    // per multi-GiB chunk instead of one per tensor. 370 scattered
    // allocations across 63 GiB is a very different page/TLB footprint from
    // the bf16 baseline's single contiguous weight buffer, and the baseline
    // reaches ~1.17 TB/s on shapes where our banks reach ~570 GB/s with the
    // SAME llama.cpp kernel (round 6 measured mmvf on our banks at the same
    // ~81 tok/s as our own GEMV, so the kernel is not the variable).
    char * chunk     = nullptr;
    size_t chunk_cap = 0;
    size_t chunk_off = 0;
};

static mach1_bank_cache & mach1_banks() {
    static mach1_bank_cache c;
    return c;
}

static bool mach1_fold_enabled();
static __global__ void mach1_rt_u16_kernel(const float * src, half * dst, const int64_t total);
static void mach1_fold_mmvf(ggml_backend_cuda_context & ctx, const half * W,
                            const float * y, float * out,
                            const int64_t cols, const int64_t rows);

static bool mach1_bank_enabled(int device) {
    GGML_UNUSED(device);
    // opt-in only: measured ~neutral for decode (see above), so it does not earn
    // 2+ GiB of VRAM and a process-lifetime cache by default. Fold mode lives in
    // the same cache, so it implies banking.
    static const bool on = mach1_env_int("GGML_MACH1_BANK", 0) != 0 || mach1_fold_enabled();
    return on;
}

// Returns a filled bank for `key`, or nullptr when banking is off, out of
// budget, or unsafe right now. `decode(dst)` is called at most once per key.
template <typename F>
static half * mach1_bank_get(const void * key, size_t elems, int device,
                             cudaStream_t stream, F && decode, const bool force = false) {
    // force: a scoped tier (HEAD_BANK) wants this one bank without flipping
    // the global bank enable, which region gates key on
    if (!mach1_bank_enabled(device) && !force) {
        return nullptr;
    }

    mach1_bank_cache & c = mach1_banks();
    std::lock_guard<std::mutex> lock(c.mtx);

    // A HIT is served even mid-capture. Hiding it sends the op down a branch
    // the uncaptured warmup never took, and that branch's first-use lazy init
    // - the z-table's D2H sync, a cold pool block - is illegal inside a
    // capture. That is what aborted every banked prefill graph.
    const auto it = c.banks.find(key);
    if (it != c.banks.end()) {
        return it->second;
    }

    // A MISS must not allocate mid-capture: cudaMalloc is illegal there and a
    // failed alloc would poison the capture. Banks fill on uncaptured calls.
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) {
        cudaGetLastError();
        return nullptr;
    }

    if (!c.ready) {
        c.budget  = (size_t) std::max(0, mach1_env_int("GGML_MACH1_BANK_GB", 4))*1024ull*1024ull*1024ull;
        c.reserve = (size_t) std::max(0, mach1_env_int("GGML_MACH1_BANK_RESERVE_GB", 8))*1024ull*1024ull*1024ull;
        c.ready   = true;
        if (mach1_debug_on()) {
            fprintf(stderr, "mach1: weight banks enabled, budget %.1f GiB, reserve %.1f GiB\n",
                    c.budget/1073741824.0, c.reserve/1073741824.0);
        }
    }

    const size_t bytes = elems*sizeof(half);
    if (bytes > c.budget - std::min(c.used, c.budget)) {
        return nullptr;   // out of budget: this tensor keeps using the walk path
    }

    // live half of the budget: never take the device below the reserve. Read
    // per bank, so a KV cache or pool that grows after the first bank counts.
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || free_b < bytes + c.reserve) {
        cudaGetLastError();
        if (mach1_debug_on()) {
            static std::once_flag once;
            std::call_once(once, [&]() {
                fprintf(stderr, "mach1: bank reserve reached at %.2f GiB banked - "
                        "remaining tensors keep the decode path\n", c.used/1073741824.0);
            });
        }
        return nullptr;
    }

    half * bank = nullptr;
    // MEASURED: arena 91.1 vs per-tensor cudaMalloc 100.2 tok/s (paired) -
    // page/TLB locality is NOT the 570 GB/s vs 1.17 TB/s gap. Default OFF.
    static const bool arena_on = mach1_env_int("GGML_MACH1_ARENA", 0) != 0;
    if (arena_on) {
        const size_t align = 256;
        const size_t need  = (bytes + align - 1) & ~(align - 1);
        if (c.chunk == nullptr || c.chunk_off + need > c.chunk_cap) {
            size_t cap = (size_t) 4*1024*1024*1024ull;   // 4 GiB chunks
            if (cap < need) {
                cap = need;
            }
            const size_t left = c.budget - std::min(c.used, c.budget);
            if (cap > left) {
                cap = left;
            }
            char * nc = nullptr;
            if (cap < need || cudaMalloc(&nc, cap) != cudaSuccess) {
                cudaGetLastError();
                c.budget = c.used;
                return nullptr;
            }
            c.chunk     = nc;
            c.chunk_cap = cap;
            c.chunk_off = 0;
            if (mach1_debug_on()) {
                fprintf(stderr, "mach1: bank arena chunk %.1f GiB\n", cap/1073741824.0);
            }
        }
        bank = (half *)(c.chunk + c.chunk_off);
        c.chunk_off += need;
    } else if (cudaMalloc(&bank, bytes) != cudaSuccess) {
        cudaGetLastError();
        c.budget = c.used;   // stop trying once the device says no
        return nullptr;
    }

    decode(bank);

    c.used += bytes;
    c.banks[key] = bank;
    if (mach1_debug_on()) {
        fprintf(stderr, "mach1: banked %.1f MiB (total %.2f GiB, %zu tensors)\n",
                bytes/1048576.0, c.used/1073741824.0, c.banks.size());
    }
    return bank;
}

// GGML_MACH1_DEBUG=2: report a tlut's lattice structure once per table.
// The z-bank tiers below need the decoded values to be exact integer
// multiples of one scale (that is what lets a bank hold small integers and a
// standard block-quant type reproduce the weights BITWISE); this measures it
// instead of assuming it. Prints the distinct level count, the inferred unit,
// the integer indices, and the worst |v/unit - round(v/unit)| residual.
static void mach1_lattice_report(const char * name, const void * dev_data,
                                 size_t n_elem, bool f16, cudaStream_t stream) {
    static std::mutex mtx;
    static std::set<const void *> seen;
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!seen.insert(dev_data).second) {
            return;
        }
    }

    std::vector<uint16_t> h16;
    std::vector<float>    h32;
    const size_t bytes = n_elem*(f16 ? 2 : 4);
    if (f16) {
        h16.resize(n_elem);
        CUDA_CHECK(cudaMemcpyAsync(h16.data(), dev_data, bytes, cudaMemcpyDeviceToHost, stream));
    } else {
        h32.resize(n_elem);
        CUDA_CHECK(cudaMemcpyAsync(h32.data(), dev_data, bytes, cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::set<float> levels;
    for (size_t i = 0; i < n_elem && levels.size() <= 4096; ++i) {
        levels.insert(f16 ? GGML_FP16_TO_FP32((ggml_fp16_t) h16[i]) : h32[i]);
    }

    float amin = 0.0f;
    for (const float v : levels) {
        const float a = fabsf(v);
        if (a > 0.0f && (amin == 0.0f || a < amin)) {
            amin = a;
        }
    }
    float resid = 0.0f;
    for (const float v : levels) {
        const float q = v/amin;
        resid = std::max(resid, fabsf(q - roundf(q)));
    }
    fprintf(stderr, "mach1-lattice: %-10s n=%zu distinct=%zu unit=%.9g resid=%.3g\n",
            name, n_elem, levels.size(), amin, resid);
    if (levels.size() <= 64) {
        std::string s;
        char b[32];
        for (const float v : levels) {
            snprintf(b, sizeof(b), "%.0f ", roundf(v/amin));
            s += b;
        }
        fprintf(stderr, "mach1-lattice: %-10s z = %s\n", name, s.c_str());
    }
}

// block size for the four FWHT-stage kernels (bit-exact for any value; the
// A/B knob GGML_MACH1_FWHT_WG selects 256, 512, or 1024)
static bool mach1_fwht_wg512() {
    static const bool wg512 = mach1_env_int("GGML_MACH1_FWHT_WG", 512) == 512;
    return wg512;
}

// The FWHT stages are 42% of the lossless decode step (per-stage ablation,
// commit 11aed4c: rt_u 1.72 + rt_out 2.62 + exp_u 1.25 + exp_out 1.25 ms of
// 16.3) and rt_u/rt_out launch ONE block on a 132-SM part. Before spending a
// cooperative-fusion rewrite on that, this knob answers whether a single
// block is compute-bound at all: every loop in mach1_fwht_block strides by
// the group size, so 1024 threads is bit-exact for any d. If 1024 does not
// move it, the stage is bound by its 12-13 __syncthreads and its lone SM's
// share of memory bandwidth, and only multi-block/fused forms can help.
static int mach1_fwht_wg() {
    static const int wg = mach1_env_int("GGML_MACH1_FWHT_WG", 512);
    return wg == 1024 ? 1024 : (wg == 256 ? 256 : 512);
}

// orthonormal Walsh-Hadamard butterflies over a shared-memory vector of
// power-of-two length d (Sylvester order). The caller loads sh, issues one
// __syncthreads(), calls this, then applies the single final
// __fdiv_rn(sh[i], __fsqrt_rn((float) d)) — the reference op order.
// SCALE folds a caller __fdiv_rn(., scn) normalization into the last shared
// pass's butterfly store: the same operation on the same value, so the result
// is bit-exact against scaling afterwards. Requires the shared loop to run
// (d != 64; at d = 64 the warp phase is the last pass).
template <bool SCALE = false>
static __device__ __forceinline__ void mach1_fwht_block(float * sh, const int d, const int tid, const int wg,
                                                        const float scn = 0.0f) {
    // span is a power of two, so (b / span, b % span) is a shift and a mask.
    // Written as div/mod the compiler cannot strength-reduce it — span is
    // loop-variant — and emits a real integer division per butterfly per pass
    // (~13 passes x 8 butterflies per thread at d = 8192). The index arithmetic
    // is identical, so this is bit-exact.
    // WARP PHASE. Pass ls pairs the indices that differ in bit ls, so passes
    // ls = 0..5 (spans 1..32) never leave an aligned 64-element chunk. Give a
    // warp one chunk, two elements per lane, and those six passes run in
    // registers with shuffles and NO __syncthreads - six block barriers and
    // twelve shared round trips per FWHT removed. Measured: the rt_u/rt_out
    // pair is ~3 ms/token across 500 single-block launches and is barrier
    // bound, which is what makes this worth doing.
    //
    // Lane L holds local indices 2L and 2L+1. For ls = 0 the partner is the
    // other element of the same lane; for ls = k+1 the partner index is
    // e ^ 2^(k+1) = 2*(L ^ 2^k) + j, i.e. lane L ^ 2^k with the same j. The
    // lane holding the LOWER index (the butterfly's base) is the one with
    // bit k of L clear, so it takes mine + theirs and the other takes
    // theirs - mine. Same two fp32 adds per butterfly, same order: bit-exact.
    constexpr int WS = ggml_cuda_get_physical_warp_size();
    int ls   = 0;
    int span = 1;
    if (WS == 32 && d >= 64) {
        const int lane = tid & 31;
        const int nw   = wg >> 5;
        for (int c = tid >> 5; c < d/64; c += nw) {
            const int   b  = c*64 + 2*lane;
            float       a0 = sh[b];
            float       a1 = sh[b + 1];
            {   // ls = 0: partner is this lane's other element
                const float t = a0;
                a0 = t + a1;
                a1 = t - a1;
            }
#pragma unroll
            for (int k = 0; k < 5; ++k) {   // ls = 1..5
                const int   mask = 1 << k;
                const float p0   = __shfl_xor_sync(0xffffffff, a0, mask);
                const float p1   = __shfl_xor_sync(0xffffffff, a1, mask);
                if ((lane & mask) == 0) {
                    a0 = a0 + p0;
                    a1 = a1 + p1;
                } else {
                    a0 = p0 - a0;
                    a1 = p1 - a1;
                }
            }
            sh[b]     = a0;
            sh[b + 1] = a1;
        }
        __syncthreads();
        ls   = 6;
        span = 64;
    }
    for (; span < d; span <<= 1, ++ls) {
        for (int b = tid; b < d/2; b += wg) {
            const int base = ((b >> ls) << (ls + 1)) + (b & (span - 1));
            const float a0 = sh[base];
            const float a1 = sh[base + span];
            if (SCALE && (span << 1) >= d) {
                sh[base]        = __fdiv_rn(a0 + a1, scn);
                sh[base + span] = __fdiv_rn(a0 - a1, scn);
            } else {
                sh[base]        = a0 + a1;
                sh[base + span] = a0 - a1;
            }
        }
        __syncthreads();
    }
}

// named-barrier sync for an aligned 128-thread tile: barrier ids 1..8 (0 is
// __syncthreads), so up to 8 tiles of one block sync independently
static __device__ __forceinline__ void mach1_tile_sync(const int bar_id) {
    asm volatile("bar.sync %0, 128;" :: "r"(bar_id));
}

// mach1_fwht_block for one 128-thread tile of a larger block: the butterflies,
// their pairing and their order are identical (values do not depend on wg or
// the sync primitive), only __syncthreads becomes a named barrier
static __device__ __forceinline__ void mach1_fwht_tile(float * sh, const int d,
                                                       const int tid, const int bar_id) {
    constexpr int WS = ggml_cuda_get_physical_warp_size();
    constexpr int wg = 128;
    int ls   = 0;
    int span = 1;
    if (WS == 32 && d >= 64) {
        const int lane = tid & 31;
        const int nw   = wg >> 5;
        for (int c = tid >> 5; c < d/64; c += nw) {
            const int   b  = c*64 + 2*lane;
            float       a0 = sh[b];
            float       a1 = sh[b + 1];
            {
                const float t = a0;
                a0 = t + a1;
                a1 = t - a1;
            }
#pragma unroll
            for (int k = 0; k < 5; ++k) {
                const int   mask = 1 << k;
                const float p0   = __shfl_xor_sync(0xffffffff, a0, mask);
                const float p1   = __shfl_xor_sync(0xffffffff, a1, mask);
                if ((lane & mask) == 0) {
                    a0 = a0 + p0;
                    a1 = a1 + p1;
                } else {
                    a0 = p0 - a0;
                    a1 = p1 - a1;
                }
            }
            sh[b]     = a0;
            sh[b + 1] = a1;
        }
        mach1_tile_sync(bar_id);
        ls   = 6;
        span = 64;
    }
    for (; span < d; span <<= 1, ++ls) {
        for (int b = tid; b < d/2; b += wg) {
            const int base = ((b >> ls) << (ls + 1)) + (b & (span - 1));
            const float a0 = sh[base];
            const float a1 = sh[base + span];
            sh[base]        = a0 + a1;
            sh[base + span] = a0 - a1;
        }
        mach1_tile_sync(bar_id);
    }
}

//
// Tensor-core Kronecker FWHT (GGML_MACH1_TC_FWHT, default 1; NVIDIA sm_80+).
//
// Sylvester order factorizes H_n = H_A (x) H_B with A the HIGH-bit factor, so
// y = H_n x becomes Y = Ha @ X @ Hb with X = reshape(x, [A, B]) row-major -
// two small mma.m16n8k16 GEMM passes instead of log2(n) butterfly passes.
// Probe: pocs/mach1-chainbench --tcfwht (the reg-resident two-dot runs 2-3x
// under the butterfly at every census dim; the orientation is verified
// 0.00e+00 against a float64 butterfly reference). NOT bit-exact: the
// operands round through fp16 (~2.8e-4 rel-RMS on uniform inputs), so paths
// served by this are certified by the KL gate, never the bitwise one; the
// flag defaults ON. Decisive probe facts kept verbatim:
//   - the 8-half row pad on every shared tile (without it every ldmatrix row
//     lands in one bank group and serializes 8-way - measured 3.5x SLOWER);
//   - the +-1 H tiles resident in registers as mma fragments;
//   - the per-transform alpha guard (heavy-tail activations overflow fp16
//     without it):
//     alpha = max(max|v|/30000, 1) divided out of every fp16 cast, pass 1
//     stores T/A (|T| <= A*30000 would overflow the cast), and the output
//     scale A*alpha/sqrt(n) undoes both and applies the orthonormal
//     1/sqrt(n) in one multiply.
//

static bool mach1_tc_fwht_ok(int device) {
    static const bool on = mach1_env_int("GGML_MACH1_TC_FWHT", 1) != 0;
    if (!on) {
        return false;
    }
    const auto & dev = ggml_cuda_info().devices[device];
    return ampere_mma_available(dev.cc) && dev.warp_size == 32;
}

// census factorizations; A carries the high bits of the index
static constexpr int mach1_tc_factor_a(const int d) {
    return d == 512 ? 16 : d == 2048 ? 32 : 64;
}
static constexpr int mach1_tc_factor_b(const int d) {
    return d == 512 ? 32 : d == 2048 ? 64 : d == 4096 ? 64 : 128;
}
// In-walk TC (the u prologue / out epilogue computed INSIDE a fused walk)
// measured a LOSS on the real graph while the standalone stages measured a
// win: rt_out 12.35 -> 11.01 us, but qkvzb_walk 45.42 -> 51.62 and
// qkvb_walk 40.44 -> 41.67 (TIME=1, decode, same container). The in-block
// form pays register/shared pressure the standalone kernels do not, so it is
// gated separately (GGML_MACH1_TC_WALK); it now defaults ON.
static bool mach1_tc_in_walk() {
    static const bool on = mach1_env_int("GGML_MACH1_TC_WALK", 1) != 0;
    return on;
}

static bool mach1_tc_fwht_dim(const int d) {
    return d == 512 || d == 2048 || d == 4096 || d == 8192;
}

// shared workspace: row-padded Ha, Hb, X and T tiles
static constexpr size_t mach1_tc_ws_bytes(const int d) {
    return sizeof(half)*(size_t)(mach1_tc_factor_a(d)*(mach1_tc_factor_a(d) + 8) +
                                 mach1_tc_factor_b(d)*(mach1_tc_factor_b(d) + 8) +
                                 2*mach1_tc_factor_a(d)*(mach1_tc_factor_b(d) + 8));
}

// dynamic shared beyond the default budget needs a one-time per-kernel
// opt-in; static shared can reach ~22 KB in the wg 1024 walk, so anything
// over 24 KB dynamic opts in to stay clear of the 48 KB combined default
static void mach1_smem_opt_in(const void * fn, const size_t smem) {
    if (smem <= 24*1024) {
        return;
    }
    static std::map<const void *, size_t> done;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    size_t & cur = done[fn];
    if (smem > cur) {
        CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        cur = smem;
    }
}

// +-1 Sylvester H tiles (16/32/64/128) as fp16 on device, one buffer per GPU
// at fixed offsets. Same capture guard as mach1_stream_counter: nullptr on
// first sight of a capturing stream - the caller keeps the butterfly then.
static __host__ __device__ constexpr int mach1_tc_hoff(const int s) {
    return s == 16 ? 0 : s == 32 ? 256 : s == 64 ? 1280 : 5376;
}

static const half * mach1_tc_htab(int device, cudaStream_t stream) {
    static half * tabs[GGML_CUDA_MAX_DEVICES] = {nullptr};
    static std::mutex mtx;

    std::lock_guard<std::mutex> lock(mtx);
    if (tabs[device] != nullptr) {
        return tabs[device];
    }
    // One-time upload must not ride the caller's stream: llama.cpp captures
    // CUDA graphs for decode, and the FIRST nt == 1 request lands inside
    // that capture. Bailing there left the table null forever, which
    // silently disabled TC-FWHT in every graph-enabled run (round 39).
    // cudaMalloc and a private stream are legal during another stream's
    // capture, so do the upload there and synchronize before returning.
    std::vector<half> h(mach1_tc_hoff(128) + 128*128);
    for (const int s : { 16, 32, 64, 128 }) {
        half * t = h.data() + mach1_tc_hoff(s);
        for (int r = 0; r < s; ++r) {
            for (int c = 0; c < s; ++c) {
                int par = r & c; // parity via xor-fold: MSVC has no __builtin_popcount
                par ^= par >> 4; par ^= par >> 2; par ^= par >> 1;
                t[r*s + c] = __float2half((par & 1) ? -1.0f : 1.0f);
            }
        }
    }
    half * d = nullptr;
    if (cudaMalloc(&d, h.size()*sizeof(half)) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    cudaStream_t up = nullptr;
    if (cudaStreamCreateWithFlags(&up, cudaStreamNonBlocking) != cudaSuccess) {
        cudaGetLastError();
        cudaFree(d);
        return nullptr;
    }
    CUDA_CHECK(cudaMemcpyAsync(d, h.data(), h.size()*sizeof(half), cudaMemcpyHostToDevice, up));
    CUDA_CHECK(cudaStreamSynchronize(up));
    CUDA_CHECK(cudaStreamDestroy(up));
    tabs[device] = d;
    GGML_UNUSED(stream);
    return d;
}

static const half * mach1_tc_fwht_tab(int device, cudaStream_t stream) {
    return mach1_tc_fwht_ok(device) ? mach1_tc_htab(device, stream) : nullptr;
}

// mma.m16n8k16 fragment idioms from pocs/mach1-chainbench.cu: fp16 in,
// fp32 acc; A row-major via ldmatrix.x4, B col-major via ldmatrix.x2.trans
// of k-major shared rows.
struct mach1_tc_frag_a { unsigned r[4]; };   // 16x16 A: 8 halves/lane
struct mach1_tc_frag_b { unsigned r[2]; };   // 16x8  B: 4 halves/lane
struct mach1_tc_frag_c { float f[4]; };      // 16x8  C: 4 floats/lane

// Signed-int8 m16n8k16 has the same 16x8 accumulator ownership as the fp16
// form, but A is eight bytes/lane and B four bytes/lane. Keeping these tiny
// fragment types local lets the compressed spine feed the MMA directly from
// its q8 activations and integer-lattice trellis decode.
struct mach1_tc_frag_ai8 { unsigned r[2]; };
struct mach1_tc_frag_bi8 { unsigned r; };
struct mach1_tc_frag_ci8 { int r[4]; };

static __device__ __forceinline__ void mach1_tc_mma(
        mach1_tc_frag_c & c, const mach1_tc_frag_a & a, const mach1_tc_frag_b & b) {
#ifdef AMPERE_MMA_AVAILABLE
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c.f[0]), "+f"(c.f[1]), "+f"(c.f[2]), "+f"(c.f[3])
        : "r"(a.r[0]), "r"(a.r[1]), "r"(a.r[2]), "r"(a.r[3]),
          "r"(b.r[0]), "r"(b.r[1]));
#else
    GGML_UNUSED(c); GGML_UNUSED(a); GGML_UNUSED(b);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void mach1_tc_mma_i8(
        mach1_tc_frag_ci8 & c, const mach1_tc_frag_ai8 & a, const mach1_tc_frag_bi8 & b) {
#ifdef AMPERE_MMA_AVAILABLE
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+r"(c.r[0]), "+r"(c.r[1]), "+r"(c.r[2]), "+r"(c.r[3])
        : "r"(a.r[0]), "r"(a.r[1]), "r"(b.r));
#else
    GGML_UNUSED(c); GGML_UNUSED(a); GGML_UNUSED(b);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void mach1_tc_ldmatrix_x4(const half * p, mach1_tc_frag_a & a) {
#ifdef AMPERE_MMA_AVAILABLE
    const unsigned addr = (unsigned) __cvta_generic_to_shared(p);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a.r[0]), "=r"(a.r[1]), "=r"(a.r[2]), "=r"(a.r[3])
        : "r"(addr));
#else
    GGML_UNUSED(p); GGML_UNUSED(a);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void mach1_tc_ldmatrix_x2t(const half * p, mach1_tc_frag_b & b) {
#ifdef AMPERE_MMA_AVAILABLE
    const unsigned addr = (unsigned) __cvta_generic_to_shared(p);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(b.r[0]), "=r"(b.r[1])
        : "r"(addr));
#else
    GGML_UNUSED(p); GGML_UNUSED(b);
    NO_DEVICE_CODE;
#endif
}

// Non-transposed B-fragment load for the compressed-spine MMA path. Its
// shared tile is stored [n][k] with mach1_tc_wt_off(), so m16n8k16 consumes
// it directly rather than through the k-major/transposed FWHT layout above.
static __device__ __forceinline__ void mach1_tc_ldmatrix_x2(const half * p, mach1_tc_frag_b & b) {
#ifdef AMPERE_MMA_AVAILABLE
    const unsigned addr = (unsigned) __cvta_generic_to_shared(p);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(b.r[0]), "=r"(b.r[1])
        : "r"(addr));
#else
    GGML_UNUSED(p); GGML_UNUSED(b);
    NO_DEVICE_CODE;
#endif
}

// Conflict-free [n][k] half-tile offset. Four output rows share a 128-byte
// super-row; the 16-byte chunk index is XOR-swizzled by the super-row index.
static __device__ __forceinline__ int mach1_tc_wt_off(const int n, const int kg) {
    const int sup = n >> 2;
    const int c   = ((n & 3) << 1) | kg;
    return (sup << 6) + ((c ^ (sup & 7)) << 3);
}

// One transform y = H_d(v)/sqrt(d) as the register-resident two-dot, the
// probe's tcfwht_tc_reg_kernel with its guard merged in. Every thread of the
// block must call (block barriers inside). v_i = mul[i]*src[i] (mul nullptr:
// src[i]), optionally read through __ldcg for cross-block visibility; the
// store folds the output scale and writes out[perm[i]] = y_i * sv[i]
// (sv/perm nullptr: identity). PACKED_F16 writes the same final fp32 value
// directly as half into the storage addressed by out; it is used only by the
// compressed-spine u stage. PACKED_Q8 instead quantizes each contiguous K16
// group and writes [N int8 values, N/16 float combined scales] into the
// existing N-float output slot. The final fp32 values temporarily occupy the
// now-dead Ha/Hb/X portion of ws, so that format conversion adds no global
// allocation or launch. out may be shared or global; ws is
// mach1_tc_ws_bytes(d) of 16-byte-aligned shared.
// TAIL folds the consumer nodes an rt out stage feeds into the same store:
// out[i] = ((y_i*sv[i]) * gate) + add[i] + add2[i], each optional. __fmul_rn
// and __fadd_rn pin the separate roundings of the MUL/ADD kernels the fold
// replaces (a contracted fma would change bits); y_i*sv[i] is untouched.
template <int A, int B, int WG, bool LDCG,
          bool PACKED_F16 = false, bool PACKED_Q8 = false, int SUMS = 1,
          bool TAIL = false>
static __device__ __forceinline__ void mach1_tc_fwht_apply(
        const float * __restrict__ src,
        const float *              mul,
        float       *              out,
        const float *              sv,
        const int   *              perm,
        const half  * __restrict__ gHa,
        const half  * __restrict__ gHb,
        half        *              ws,
        const int tid,
        const float q8_zstep = 0.0f,
        const float * __restrict__ sum_extra = nullptr,
        const float *              tail_add  = nullptr,
        const float *              tail_add2 = nullptr,
        const float *              tail_gate = nullptr) {
#ifdef AMPERE_MMA_AVAILABLE
    constexpr int  N     = A*B;
    constexpr int  NW    = WG/32;
    constexpr int  NJ    = B/8;                  // output column blocks
    constexpr int  NG    = NW/NJ;                // warp groups over mi rows
    constexpr int  MIW   = (A/16 + NG - 1)/NG;   // mi rows per warp
    constexpr bool HAREG = MIW*(A/16) <= 12;     // Ha frags in regs when small
    constexpr int  AP    = A + 8;                // the mandatory 8-half row pad
    constexpr int  BP    = B + 8;
    constexpr int  PER   = (N + WG - 1)/WG;
    static_assert(NW % NJ == 0, "nj-per-warp ownership needs NW divisible by B/8");
    static_assert(!(PACKED_F16 && PACKED_Q8), "packed FWHT output formats are exclusive");
    static_assert(SUMS == 1 || (SUMS == 4 && !PACKED_F16 && !PACKED_Q8),
                  "FWHT input summation is plain-fp32 SUMS1/SUMS4 only");
    static_assert(!PACKED_Q8 || A*AP + B*BP + A*BP >= 2*N,
                  "dead FWHT workspace must hold N fp32 q8 inputs");
    half * sHa = ws;                  // [A][AP]
    half * sHb = sHa + A*AP;          // [B][BP]
    half * sX  = sHb + B*BP;          // [A][BP]
    half * sT  = sX + A*BP;           // [A][BP]
    __shared__ float s_gred[32];      // NW <= 32
    __shared__ float s_alpha;

    const int lane  = tid & 31;
    const int warp  = tid >> 5;
    const int nj    = warp % NJ;
    const int migrp = warp/NJ;
    const int r     = lane % 16;
    const int c8    = (lane/16)*8;
    const int cr    = lane >> 2;      // FragC row
    const int cc    = (lane & 3)*2;   // FragC col pair

    for (int i = tid; i < A*A/8; i += WG) {   // 8-half chunks, row-aware pad
        *(int4 *) (sHa + (i/(A/8))*AP + (i % (A/8))*8) = ((const int4 *) gHa)[i];
    }
    for (int i = tid; i < B*B/8; i += WG) {
        *(int4 *) (sHb + (i/(B/8))*BP + (i % (B/8))*8) = ((const int4 *) gHb)[i];
    }

    float xr[PER];
    float mx = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int i = tid + j*WG;
        xr[j] = 0.0f;
        if (N % WG == 0 || i < N) {
            float v = LDCG ? __ldcg(&src[i]) : src[i];
            if constexpr (SUMS == 4) {
                // Fixed split order. The three extra [N] rows are token-local;
                // split 0 remains in the ordinary scr_v allocation.
                v += sum_extra[i];
                v += sum_extra[N + i];
                v += sum_extra[2*N + i];
            }
            xr[j] = mul != nullptr ? mul[i]*v : v;
            mx = fmaxf(mx, fabsf(xr[j]));
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off, 32));
    }
    if (lane == 0) {
        s_gred[warp] = mx;
    }
    __syncthreads();
    if (tid == 0) {
        float bm = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; ++w) {
            bm = fmaxf(bm, s_gred[w]);
        }
        s_alpha = fmaxf(bm/30000.0f, 1.0f);
    }
    __syncthreads();
    const float ainv = 1.0f/s_alpha;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int i = tid + j*WG;
        if (N % WG == 0 || i < N) {
            sX[(i/B)*BP + (i % B)] = __float2half(xr[j]*ainv);
        }
    }
    __syncthreads();

    mach1_tc_frag_b hbf[B/16];
#pragma unroll
    for (int kk = 0; kk < B/16; ++kk) {
        mach1_tc_ldmatrix_x2t(sHb + (kk*16 + r)*BP + nj*8, hbf[kk]);
    }
    mach1_tc_frag_a haf[HAREG ? MIW*(A/16) : 1];
    if (HAREG) {
#pragma unroll
        for (int g = 0; g < MIW; ++g) {
            const int mi = migrp + g*NG;
            if (mi < A/16) {
#pragma unroll
                for (int kk = 0; kk < A/16; ++kk) {
                    mach1_tc_ldmatrix_x4(sHa + (mi*16 + r)*AP + kk*16 + c8, haf[g*(A/16) + kk]);
                }
            }
        }
    }
    mach1_tc_frag_b xf[A/16];
#pragma unroll
    for (int kk = 0; kk < A/16; ++kk) {
        mach1_tc_ldmatrix_x2t(sX + (kk*16 + r)*BP + nj*8, xf[kk]);
    }

    // pass 1: T = Ha @ X, fp16 store with the exact 1/A renorm
    constexpr float invA = 1.0f/A;
#pragma unroll
    for (int g = 0; g < MIW; ++g) {
        const int mi = migrp + g*NG;
        if (mi < A/16) {
            mach1_tc_frag_c c;
            c.f[0] = c.f[1] = c.f[2] = c.f[3] = 0.0f;
#pragma unroll
            for (int kk = 0; kk < A/16; ++kk) {
                if (HAREG) {
                    mach1_tc_mma(c, haf[g*(A/16) + kk], xf[kk]);
                } else {
                    mach1_tc_frag_a a;
                    mach1_tc_ldmatrix_x4(sHa + (mi*16 + r)*AP + kk*16 + c8, a);
                    mach1_tc_mma(c, a, xf[kk]);
                }
            }
            sT[(mi*16 + cr + 0)*BP + nj*8 + cc + 0] = __float2half(c.f[0]*invA);
            sT[(mi*16 + cr + 0)*BP + nj*8 + cc + 1] = __float2half(c.f[1]*invA);
            sT[(mi*16 + cr + 8)*BP + nj*8 + cc + 0] = __float2half(c.f[2]*invA);
            sT[(mi*16 + cr + 8)*BP + nj*8 + cc + 1] = __float2half(c.f[3]*invA);
        }
    }
    __syncthreads();

    // pass 2: Y = T @ Hb; A*alpha/sqrt(n) undoes the renorm and the guard
    const float osc = __fdiv_rn((float) A, __fsqrt_rn((float) N)) * s_alpha;
#pragma unroll
    for (int g = 0; g < MIW; ++g) {
        const int mi = migrp + g*NG;
        if (mi < A/16) {
            mach1_tc_frag_c c;
            c.f[0] = c.f[1] = c.f[2] = c.f[3] = 0.0f;
#pragma unroll
            for (int kk = 0; kk < B/16; ++kk) {
                mach1_tc_frag_a a;
                mach1_tc_ldmatrix_x4(sT + (mi*16 + r)*BP + kk*16 + c8, a);
                mach1_tc_mma(c, a, hbf[kk]);
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int idx = (mi*16 + cr + (q < 2 ? 0 : 8))*B + nj*8 + cc + (q & 1);
                const float y = c.f[q]*osc;
                const int oi = perm != nullptr ? perm[idx] : idx;
                float v = sv != nullptr ? y*sv[idx] : y;
                if constexpr (TAIL) {
                    if (tail_gate != nullptr) {
                        v = __fmul_rn(v, *tail_gate);
                    }
                    if (tail_add != nullptr) {
                        v = __fadd_rn(v, tail_add[oi]);
                    }
                    if (tail_add2 != nullptr) {
                        v = __fadd_rn(v, tail_add2[oi]);
                    }
                }
                if constexpr (PACKED_Q8) {
                    ((float *) ws)[idx] = v;
                } else if constexpr (PACKED_F16) {
                    ((half *) out)[oi] = __float2half(v);
                } else {
                    out[oi] = v;
                }
            }
        }
    }
    if constexpr (PACKED_Q8) {
        // One warp owns each K16 group. Only its first half participates;
        // width=16 keeps the max reduction and broadcast inside that half.
        // This is the same q8 rule and multiplication order as ZDP.
        __syncthreads();
        int8_t * qout = (int8_t *) out;
        float  * qsc  = (float *) (qout + N);
        for (int g = warp; g < N/16; g += NW) {
            const float v = lane < 16 ? ((float *) ws)[g*16 + lane] : 0.0f;
            float du = fmaxf(0.0f, fabsf(v));
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) {
                du = fmaxf(du, __shfl_down_sync(0xffffffffu, du, off, 16));
            }
            du = __shfl_sync(0xffffffffu, du, 0, 16);
            if (lane < 16) {
                const float qsf = du > 0.0f ? 127.0f/du : 0.0f;
                qout[g*16 + lane] = (int8_t) __float2int_rn(v*qsf);
            }
            if (lane == 0) {
                qsc[g] = q8_zstep*du*(1.0f/127.0f);
            }
        }
    }
#else
    GGML_UNUSED(src); GGML_UNUSED(mul); GGML_UNUSED(out); GGML_UNUSED(sv);
    GGML_UNUSED(perm); GGML_UNUSED(gHa); GGML_UNUSED(gHb); GGML_UNUSED(ws);
    GGML_UNUSED(tid); GGML_UNUSED(q8_zstep); GGML_UNUSED(sum_extra);
    GGML_UNUSED(tail_add); GGML_UNUSED(tail_add2); GGML_UNUSED(tail_gate);
    NO_DEVICE_CODE;
#endif
}

// runtime-d dispatch over the census factorizations (the hosts gate d to
// exactly these before selecting a TCF kernel)
template <int WG, bool LDCG>
static __device__ __forceinline__ void mach1_tc_fwht_dyn(
        const int d,
        const float * __restrict__ src, const float * mul, float * out,
        const float * sv, const int * perm,
        const half * __restrict__ htab, half * ws, const int tid) {
    switch (d) {
        case 512:
            mach1_tc_fwht_apply<16, 32, WG, LDCG>(src, mul, out, sv, perm,
                htab + mach1_tc_hoff(16), htab + mach1_tc_hoff(32), ws, tid);
            break;
        case 2048:
            mach1_tc_fwht_apply<32, 64, WG, LDCG>(src, mul, out, sv, perm,
                htab + mach1_tc_hoff(32), htab + mach1_tc_hoff(64), ws, tid);
            break;
        case 4096:
            mach1_tc_fwht_apply<64, 64, WG, LDCG>(src, mul, out, sv, perm,
                htab + mach1_tc_hoff(64), htab + mach1_tc_hoff(64), ws, tid);
            break;
        default:   // 8192
            mach1_tc_fwht_apply<64, 128, WG, LDCG>(src, mul, out, sv, perm,
                htab + mach1_tc_hoff(64), htab + mach1_tc_hoff(128), ws, tid);
            break;
    }
}

// GGML_MACH1_TC_WG: threads per transform block. The decode transforms are a
// grid of nt blocks on a 142-SM part, so their cost is the block's own
// latency, not occupancy - 1024 threads halve the per-warp MMA row count
// wherever A/16 >= (WG/32)/(B/8), i.e. A*B >= 4096. VALUES ARE UNCHANGED at
// any width: the block max is a max-reduction, and every output element is
// still one warp accumulating the same k range in the same order; only the
// (mi, nj) -> warp assignment moves.
static int mach1_tc_wg() {
    static const int wg = mach1_env_int("GGML_MACH1_TC_WG", 512) == 1024 ? 1024 : 512;
    return wg;
}

// standalone TC stages (decode, nt == 1): one 512-thread block per transform,
// replacing mach1_rt_u_kernel / mach1_rt_out_kernel launches
// grid.z = token: block t transforms row t (nt = 1 keeps the old shape)
template <int A, int B, bool PACKED_F16 = false, bool PACKED_Q8 = false, int WG = 512>
__global__ void __launch_bounds__(WG) mach1_rt_u_tc_kernel(
        const half  * __restrict__ htab,
        const float * __restrict__ su,
        const float * __restrict__ x,
        float       * __restrict__ scr_u,
        const float q8_zstep) {
    extern __shared__ float obuf[];
    const int64_t off = (int64_t) blockIdx.z*((int64_t) A*B);
    if constexpr (PACKED_Q8) {
        // Keep the ordinary N-float token stride. The q8 payload plus one
        // float scale per K16 consumes only 5/16 of that existing slot.
        mach1_tc_fwht_apply<A, B, WG, false, false, true>(x + off, su, scr_u + off,
            nullptr, nullptr, htab + mach1_tc_hoff(A), htab + mach1_tc_hoff(B),
            (half *) obuf, threadIdx.x, q8_zstep);
    } else if constexpr (PACKED_F16) {
        // x is a separate input, so all token CTAs may write their packed rows
        // directly without the cross-row overwrite hazard of in-place packing.
        float * packed = (float *) ((half *) scr_u + off);
        mach1_tc_fwht_apply<A, B, WG, false, true>(x + off, su, packed, nullptr, nullptr,
            htab + mach1_tc_hoff(A), htab + mach1_tc_hoff(B), (half *) obuf, threadIdx.x);
    } else {
        mach1_tc_fwht_apply<A, B, WG, false>(x + off, su, scr_u + off, nullptr, nullptr,
            htab + mach1_tc_hoff(A), htab + mach1_tc_hoff(B), (half *) obuf, threadIdx.x);
    }
}

template <int A, int B>
__global__ void mach1_rt_out_tc_kernel(
        const half  * __restrict__ htab,
        const float * __restrict__ sv,
        const float * __restrict__ scr_v,
        float       * __restrict__ dst) {
    extern __shared__ float obuf[];
    const int64_t off = (int64_t) blockIdx.z*((int64_t) A*B);
    mach1_tc_fwht_apply<A, B, 512, false>(scr_v + off, nullptr, dst + off, sv, nullptr,
        htab + mach1_tc_hoff(A), htab + mach1_tc_hoff(B), (half *) obuf, threadIdx.x);
}

// KS4 companion for the exact B16 out-projection shape. In the opt-in arm the
// ordinary scr_v allocation expands to token-major [nt, 4, m] partials, so the
// existing TC transform can fold their fixed-order FP32 sum into its input
// load without a reduction launch or a persistent buffer.
template <int A, int B>
__global__ void mach1_rt_out_tc_sum4_kernel(
        const half  * __restrict__ htab,
        const float * __restrict__ sv,
        const float * __restrict__ partials,
        float       * __restrict__ dst) {
    extern __shared__ float obuf[];
    constexpr int N = A*B;
    const int64_t off = (int64_t) blockIdx.z*N;
    const float * parts = partials + (int64_t) blockIdx.z*4*N;
    mach1_tc_fwht_apply<A, B, 512, false, false, false, 4>(
        parts, nullptr, dst + off, sv, nullptr,
        htab + mach1_tc_hoff(A), htab + mach1_tc_hoff(B),
        (half *) obuf, threadIdx.x, 0.0f, parts + N);
}

// GGML_MACH1_RT_TAIL companion: same transform, with the rt op's elementwise
// consumers folded into the store. dst may share a base with an add source
// (the graph allocator runs those adds in place), so neither is __restrict__.
template <int A, int B>
__global__ void mach1_rt_out_tc_sum4_tail_kernel(
        const half  * __restrict__ htab,
        const float * __restrict__ sv,
        const float * __restrict__ partials,
        float       *              dst,
        const float *              tail_add,
        const float *              tail_add2,
        const float *              tail_gate) {
    extern __shared__ float obuf[];
    constexpr int N = A*B;
    const int64_t off = (int64_t) blockIdx.z*N;
    const float * parts = partials + (int64_t) blockIdx.z*4*N;
    mach1_tc_fwht_apply<A, B, 512, false, false, false, 4, true>(
        parts, nullptr, dst + off, sv, nullptr,
        htab + mach1_tc_hoff(A), htab + mach1_tc_hoff(B),
        (half *) obuf, threadIdx.x, 0.0f, parts + N,
        tail_add  != nullptr ? tail_add  + off : nullptr,
        tail_add2 != nullptr ? tail_add2 + off : nullptr,
        tail_gate != nullptr ? tail_gate + blockIdx.z : nullptr);
}

static void mach1_rt_u_tc(const half * htab, const float * su, const float * x,
                          float * scr_u, const int n, cudaStream_t stream, const int nt = 1,
                          const bool packed_f16 = false, const bool packed_q8 = false,
                          const float q8_zstep = 0.0f) {
    GGML_ASSERT(!(packed_f16 && packed_q8));
    const ggml_cuda_kernel_launch_params p =
        ggml_cuda_kernel_launch_params(dim3(1, 1, nt), dim3(512, 1, 1), mach1_tc_ws_bytes(n), stream);
    // n = 4096 is A = B = 64, the one standalone u shape where 1024 threads
    // keep every warp group on an mi row (A/16 == NW/NJ) and halve MIW
    if (n == 4096 && mach1_tc_wg() == 1024) {
        const ggml_cuda_kernel_launch_params pw =
            ggml_cuda_kernel_launch_params(dim3(1, 1, nt), dim3(1024, 1, 1), mach1_tc_ws_bytes(n), stream);
        if (packed_q8) {
            mach1_smem_opt_in((const void *) mach1_rt_u_tc_kernel<64, 64, false, true, 1024>, mach1_tc_ws_bytes(n));
            mach1_launch(mach1_rt_u_tc_kernel<64, 64, false, true, 1024>, pw,
                htab, su, x, scr_u, q8_zstep);
        } else if (packed_f16) {
            mach1_smem_opt_in((const void *) mach1_rt_u_tc_kernel<64, 64, true, false, 1024>, mach1_tc_ws_bytes(n));
            mach1_launch(mach1_rt_u_tc_kernel<64, 64, true, false, 1024>, pw,
                htab, su, x, scr_u, 0.0f);
        } else {
            mach1_smem_opt_in((const void *) mach1_rt_u_tc_kernel<64, 64, false, false, 1024>, mach1_tc_ws_bytes(n));
            mach1_launch(mach1_rt_u_tc_kernel<64, 64, false, false, 1024>, pw,
                htab, su, x, scr_u, 0.0f);
        }
        return;
    }
    switch (n) {
        case 512:
            if (packed_q8) {
                mach1_launch(mach1_rt_u_tc_kernel<16, 32, false, true>, p,
                    htab, su, x, scr_u, q8_zstep);
            } else if (packed_f16) {
                mach1_launch(mach1_rt_u_tc_kernel<16, 32, true>, p,
                    htab, su, x, scr_u, 0.0f);
            } else {
                mach1_launch(mach1_rt_u_tc_kernel<16, 32>, p,
                    htab, su, x, scr_u, 0.0f);
            }
            break;
        case 2048:
            if (packed_q8) {
                mach1_launch(mach1_rt_u_tc_kernel<32, 64, false, true>, p,
                    htab, su, x, scr_u, q8_zstep);
            } else if (packed_f16) {
                mach1_launch(mach1_rt_u_tc_kernel<32, 64, true>, p,
                    htab, su, x, scr_u, 0.0f);
            } else {
                mach1_launch(mach1_rt_u_tc_kernel<32, 64>, p,
                    htab, su, x, scr_u, 0.0f);
            }
            break;
        default:   // 4096 (the rt_u shared bound)
            if (packed_q8) {
                mach1_smem_opt_in((const void *) mach1_rt_u_tc_kernel<64, 64, false, true>, mach1_tc_ws_bytes(n));
                mach1_launch(mach1_rt_u_tc_kernel<64, 64, false, true>, p,
                    htab, su, x, scr_u, q8_zstep);
            } else if (packed_f16) {
                mach1_smem_opt_in((const void *) mach1_rt_u_tc_kernel<64, 64, true>, mach1_tc_ws_bytes(n));
                mach1_launch(mach1_rt_u_tc_kernel<64, 64, true>, p,
                    htab, su, x, scr_u, 0.0f);
            } else {
                mach1_smem_opt_in((const void *) mach1_rt_u_tc_kernel<64, 64>, mach1_tc_ws_bytes(n));
                mach1_launch(mach1_rt_u_tc_kernel<64, 64>, p,
                    htab, su, x, scr_u, 0.0f);
            }
            break;
    }
}

static void mach1_rt_out_tc(const half * htab, const float * sv, const float * scr_v,
                            float * dst, const int m, cudaStream_t stream, const int nt = 1) {
    const size_t smem = mach1_tc_ws_bytes(m);
    const ggml_cuda_kernel_launch_params p =
        ggml_cuda_kernel_launch_params(dim3(1, 1, nt), dim3(512, 1, 1), smem, stream);
    switch (m) {
        case 512:
            mach1_launch(mach1_rt_out_tc_kernel<16, 32>, p, htab, sv, scr_v, dst);
            break;
        case 2048:
            mach1_launch(mach1_rt_out_tc_kernel<32, 64>, p, htab, sv, scr_v, dst);
            break;
        case 4096:
            mach1_smem_opt_in((const void *) mach1_rt_out_tc_kernel<64, 64>, smem);
            mach1_launch(mach1_rt_out_tc_kernel<64, 64>, p, htab, sv, scr_v, dst);
            break;
        default:   // 8192 (the rt_out shared bound)
            mach1_smem_opt_in((const void *) mach1_rt_out_tc_kernel<64, 128>, smem);
            mach1_launch(mach1_rt_out_tc_kernel<64, 128>, p, htab, sv, scr_v, dst);
            break;
    }
}

static void mach1_rt_out_tc_sum4_p16(
        const half * htab, const float * sv, const float * partials,
        float * dst, cudaStream_t stream, const int nt = 16,
        const mach1_rt_tail * tail = nullptr) {
    constexpr int M = 2048;
    const size_t smem = mach1_tc_ws_bytes(M);
    const ggml_cuda_kernel_launch_params p =
        ggml_cuda_kernel_launch_params(dim3(1, 1, nt), dim3(512, 1, 1), smem, stream);
    if (tail != nullptr) {
        mach1_launch(mach1_rt_out_tc_sum4_tail_kernel<32, 64>, p,
            htab, sv, partials, dst, tail->add, tail->add2, tail->gate);
        return;
    }
    mach1_launch(mach1_rt_out_tc_sum4_kernel<32, 64>, p, htab, sv, partials, dst);
}

// standard-output-basis epilogues (mach1_rt_stdout_on): y = sv .* v, no
// output Hadamard, no 1/sqrt(m). The sum4 form folds the token-major
// [nt, 4, m] split-K partials in the same fixed (((p0+p1)+p2)+p3) order as
// the TC transform's input load.
__global__ void mach1_rt_out_rs_kernel(
        const float * __restrict__ sv, const float * __restrict__ scr_v,
        float * __restrict__ dst, const int m) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= m) {
        return;
    }
    const int64_t off = (int64_t) blockIdx.z*m;
    dst[off + i] = scr_v[off + i]*sv[i];
}

__global__ void mach1_rt_out_rs_sum4_kernel(
        const float * __restrict__ sv, const float * __restrict__ partials,
        float * __restrict__ dst, const int m) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= m) {
        return;
    }
    const float * parts = partials + (int64_t) blockIdx.z*4*m;
    float s = parts[i];
    s += parts[m + i];
    s += parts[2*m + i];
    s += parts[3*m + i];
    dst[(int64_t) blockIdx.z*m + i] = s*sv[i];
}

static void mach1_rt_out_rs(const float * sv, const float * scr_v, float * dst,
                            const int m, cudaStream_t stream, const int nt = 1) {
    mach1_launch(mach1_rt_out_rs_kernel,
        ggml_cuda_kernel_launch_params(dim3((m + 255)/256, 1, nt), dim3(256, 1, 1), 0, stream),
        sv, scr_v, dst, m);
}

static void mach1_rt_out_rs_sum4_p16(const float * sv, const float * partials,
                                     float * dst, cudaStream_t stream,
                                     const int nt = 16) {
    constexpr int M = 2048;
    mach1_launch(mach1_rt_out_rs_sum4_kernel,
        ggml_cuda_kernel_launch_params(dim3((M + 255)/256, 1, nt), dim3(256, 1, 1), 0, stream),
        sv, partials, dst, M);
}

// store 16 contiguous halves as two 16-byte transactions (the value bits are
// exactly the input halves — packing only changes the store width). dst must
// be 16-byte aligned; every bank row chunk is (offsets are multiples of 16
// halves = 32 bytes).
static __device__ __forceinline__ void mach1_store_half16(half * dst, const half * v) {
    uint4 a, b;
    a.x = (uint32_t) __half_as_ushort(v[0])  | ((uint32_t) __half_as_ushort(v[1])  << 16);
    a.y = (uint32_t) __half_as_ushort(v[2])  | ((uint32_t) __half_as_ushort(v[3])  << 16);
    a.z = (uint32_t) __half_as_ushort(v[4])  | ((uint32_t) __half_as_ushort(v[5])  << 16);
    a.w = (uint32_t) __half_as_ushort(v[6])  | ((uint32_t) __half_as_ushort(v[7])  << 16);
    b.x = (uint32_t) __half_as_ushort(v[8])  | ((uint32_t) __half_as_ushort(v[9])  << 16);
    b.y = (uint32_t) __half_as_ushort(v[10]) | ((uint32_t) __half_as_ushort(v[11]) << 16);
    b.z = (uint32_t) __half_as_ushort(v[12]) | ((uint32_t) __half_as_ushort(v[13]) << 16);
    b.w = (uint32_t) __half_as_ushort(v[14]) | ((uint32_t) __half_as_ushort(v[15]) << 16);
    ((uint4 *) dst)[0] = a;
    ((uint4 *) dst)[1] = b;
}

// storage-expert group of flat pair rank p (see mach1_exp_group.comp)
static __device__ __forceinline__ uint32_t mach1_group_of(
        const int32_t * __restrict__ remap,
        const int32_t * __restrict__ ids,
        const int p, const int n_used, const int n_kept,
        const int ids_s0, const int ids_s1) {
    const int s = p % n_used;
    const int t = p / n_used;
    const uint32_t id = (uint32_t) ids[s*ids_s0 + t*ids_s1];
    const uint32_t rm = (uint32_t) remap[id];
    return (rm & GGML_CUDA_MACH1_DEM_FLAG) != 0u ? (uint32_t) n_kept + (rm & ~GGML_CUDA_MACH1_DEM_FLAG) : rm;
}

//
// EMBED_GATHER — one thread per output element (mach1_embed_gather.comp)
//
// The loader places token_embd.m1_codes/m1_lut in the input layer's host
// buffer, so the gather kernel zero-copies both tables over PCIe on every
// call (prefill2k census: 16 ms per 2048-token prefill). Mirror them to the
// device once per process; same bytes, same kernel, so the output is
// bit-identical. GGML_MACH1_EMBED_DEV=0 restores the zero-copy reads.

struct mach1_embed_dev {
    const void * src_codes = nullptr;
    const void * src_lut   = nullptr;
    uint8_t    * codes     = nullptr;
    uint16_t   * lut       = nullptr;
};

static bool mach1_embed_dev_on() {
    static const bool on = mach1_env_int("GGML_MACH1_EMBED_DEV", 1) != 0;
    return on;
}

static const mach1_embed_dev * mach1_embed_dev_cache(
        const int device, const ggml_tensor * codes, const ggml_tensor * lut) {
    static mach1_embed_dev caches[GGML_CUDA_MAX_DEVICES];
    static bool failed[GGML_CUDA_MAX_DEVICES] = {false};
    static std::mutex mtx;

    std::lock_guard<std::mutex> lock(mtx);
    mach1_embed_dev & c = caches[device];
    if (c.codes != nullptr && c.src_codes == codes->data && c.src_lut == lut->data) {
        return &c;
    }
    if (failed[device]) {
        return nullptr;
    }
    if (c.codes != nullptr) {   // model reload moved the tables: drop the stale mirror
        cudaFree(c.codes);
        cudaFree(c.lut);
        c = {};
    }
    const size_t cb = ggml_nbytes(codes);
    const size_t lb = ggml_nbytes(lut);
    uint8_t  * dc = nullptr;
    uint16_t * dl = nullptr;
    if (cudaMalloc(&dc, cb) != cudaSuccess || cudaMalloc(&dl, lb) != cudaSuccess) {
        cudaGetLastError();
        if (dc) { cudaFree(dc); }
        failed[device] = true;
        return nullptr;
    }
    // First call can land inside a decode graph capture; cudaMalloc and a
    // private stream are legal there (same idiom as mach1_tc_htab).
    cudaStream_t up = nullptr;
    if (cudaStreamCreateWithFlags(&up, cudaStreamNonBlocking) != cudaSuccess) {
        cudaGetLastError();
        cudaFree(dc);
        cudaFree(dl);
        failed[device] = true;
        return nullptr;
    }
    CUDA_CHECK(cudaMemcpyAsync(dc, codes->data, cb, cudaMemcpyHostToDevice, up));
    CUDA_CHECK(cudaMemcpyAsync(dl, lut->data,   lb, cudaMemcpyHostToDevice, up));
    CUDA_CHECK(cudaStreamSynchronize(up));
    CUDA_CHECK(cudaStreamDestroy(up));
    c.src_codes = codes->data;
    c.src_lut   = lut->data;
    c.codes     = dc;
    c.lut       = dl;
    fprintf(stderr, "mach1: embed tables mirrored to device (%zu MB)\n",
            (cb + lb)/(1024*1024));
    return &c;
}

static __global__ void mach1_embed_gather_kernel(
        const uint8_t  * __restrict__ codes,
        const uint16_t * __restrict__ lut,    // bf16 bit patterns
        const int32_t  * __restrict__ ids,
        float          * __restrict__ dst,
        const int n_embd, const int64_t total) {
    const int64_t gid = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (gid >= total) {
        return;
    }
    const int64_t tok = gid / n_embd;
    const int     j   = (int)(gid % n_embd);
    const int64_t r   = ids[tok];
    const int     ng  = n_embd / 64;

    const uint32_t by = codes[r*(n_embd/2) + (j >> 1)];
    const uint32_t q  = (j & 1) ? (by >> 4) : (by & 0x0Fu);
    const uint32_t bits = (uint32_t) lut[(r*ng + (j >> 6))*16 + q] << 16;
    dst[gid] = __uint_as_float(bits);
}

void ggml_cuda_op_mach1_embed_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * codes = dst->src[0];
    const ggml_tensor * lut   = dst->src[1];
    const ggml_tensor * ids   = dst->src[2];

    GGML_ASSERT(codes->type == GGML_TYPE_I8);
    GGML_ASSERT(lut->type   == GGML_TYPE_BF16);
    GGML_ASSERT(ids->type   == GGML_TYPE_I32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(codes));
    GGML_ASSERT(ggml_is_contiguous(lut));
    GGML_ASSERT(ggml_is_contiguous(ids));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int     n_embd = (int) dst->ne[0];
    const int64_t nt     = ids->ne[0];
    const int64_t total  = nt*n_embd;

    const uint8_t  * codes_p = (const uint8_t  *) codes->data;
    const uint16_t * lut_p   = (const uint16_t *) lut->data;
    if (mach1_embed_dev_on() && codes->buffer != nullptr &&
            ggml_backend_buffer_is_host(codes->buffer)) {
        const mach1_embed_dev * c = mach1_embed_dev_cache(ctx.device, codes, lut);
        if (c != nullptr) {
            codes_p = c->codes;
            lut_p   = c->lut;
        }
    }

    const dim3 grid((unsigned)((total + 255)/256), 1, 1);
    mach1_launch(mach1_embed_gather_kernel,
        ggml_cuda_kernel_launch_params(grid, dim3(256, 1, 1), 0, ctx.stream()),
        codes_p,
        lut_p,
        (const int32_t  *) ids->data,
        (float          *) dst->data,
        n_embd, total);
}

//
// EMBED_ROWS — one thread per (token, group), serial 3-bit unpack
// (mach1_embed_rows.comp)
//

static __global__ void mach1_embed_rows_kernel(
        const uint8_t * __restrict__ q,
        const half    * __restrict__ mn,
        const half    * __restrict__ mx,
        const int32_t * __restrict__ ids,
        float         * __restrict__ dst,
        const int n_embd, const int ng, const int grp, const int n_tokens, const int row_bytes) {
    const int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if (gid >= n_tokens*ng) {
        return;
    }
    const int     tok = gid / ng;
    const int     g   = gid % ng;
    const int64_t r   = ids[tok];

    const float mnf  = __half2float(mn[r*ng + g]);
    const float d    = __half2float(mx[r*ng + g]) - mnf;
    // reference: step = max(mx - mn, 1e-8) / 7, all fp32 (pinned rounding)
    const float step = __fdiv_rn(d > 1e-8f ? d : 1e-8f, 7.0f);

    int64_t  pbi   = r*row_bytes + (g*grp*3)/8;   // groups are byte-aligned (grp*3 % 8 == 0)
    uint32_t acc   = 0;
    int      nbits = 0;
    const int64_t obase = (int64_t) tok*n_embd + (int64_t) g*grp;
    for (int t = 0; t < grp; ++t) {
        while (nbits < 3) {
            acc = (acc << 8) | q[pbi++];
            nbits += 8;
        }
        nbits -= 3;
        const uint32_t qv = (acc >> nbits) & 0x7u;
        // reference rounds q*step to fp32 BEFORE the add — pinned, no fma
        const float prod = __fmul_rn((float) qv, step);
        dst[obase + t] = __fadd_rn(mnf, prod);
    }
}

void ggml_cuda_op_mach1_embed_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * mn  = dst->src[1];
    const ggml_tensor * mx  = dst->src[2];
    const ggml_tensor * ids = dst->src[3];

    GGML_ASSERT(q->type   == GGML_TYPE_I8);
    GGML_ASSERT(mn->type  == GGML_TYPE_F16);
    GGML_ASSERT(mx->type  == GGML_TYPE_F16);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(mn));
    GGML_ASSERT(ggml_is_contiguous(mx));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int n_embd    = (int) dst->ne[0];
    const int ng        = (int) mn->ne[0];
    const int grp       = n_embd / ng;
    const int n_tokens  = (int) ids->ne[0];
    const int row_bytes = (int) q->ne[0];

    const dim3 grid((unsigned)((n_tokens*ng + 255)/256), 1, 1);
    mach1_launch(mach1_embed_rows_kernel,
        ggml_cuda_kernel_launch_params(grid, dim3(256, 1, 1), 0, ctx.stream()),
        (const uint8_t *) q->data,
        (const half    *) mn->data,
        (const half    *) mx->data,
        (const int32_t *) ids->data,
        (float         *) dst->data,
        n_embd, ng, grp, n_tokens, row_bytes);
}

//
// HEAD_MM — one thread per vocab row x 8-token tile (mach1_head_mm.comp)
//

static __global__ void mach1_head_mm_kernel(
        const uint8_t * __restrict__ qp,
        const half    * __restrict__ gscale,
        const float   * __restrict__ x,
        float         * __restrict__ dst,
        const int n, const int vocab, const int n_tokens) {
    const int r = blockIdx.x*blockDim.x + threadIdx.x;
    if (r >= vocab) {
        return;
    }
    const int jt0 = blockIdx.y*8;
    const int njt = min(8, n_tokens - jt0);

    const int64_t row_bytes = (int64_t) n/8*5;
    const int     ng        = n/64;

    float accs[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        accs[j] = 0.0f;
    }

    // the row is a little-endian 5-bit code stream (code j at bits [5j, 5j+5));
    // process 4 blocks (32 codes, 20 bytes) per iteration with 5 aligned
    // 32-bit loads instead of byte loads. Rows are 4-byte aligned: row_bytes =
    // n/8*5 and n % 64 == 0 (gscale groups), so row_bytes % 4 == 0. The
    // extracted code bits — and therefore all arithmetic — are identical to
    // the byte-wise unpack.
    const uint32_t * pw = (const uint32_t *)(qp + (int64_t) r*row_bytes);
    for (int b4i = 0; b4i < n/32; ++b4i) {
        uint32_t w5[5];
#pragma unroll
        for (int q = 0; q < 5; ++q) {
            w5[q] = pw[b4i*5 + q];
        }
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int bit0 = 5*i;
            const int wi   = bit0 >> 5;
            const int o    = bit0 & 31;
            uint32_t qv = w5[wi] >> o;
            if (o > 27) {
                qv |= w5[wi + 1] << (32 - o);   // straddles a word boundary
            }
            qv &= 31u;
            const int j = b4i*32 + i;
            // reference decode: q_f32 * gscale_f32, one fp32 rounding
            const float w = (float)((int) qv - 16) * __half2float(gscale[(int64_t) r*ng + (j >> 6)]);
#pragma unroll
            for (int jt = 0; jt < 8; ++jt) {
                if (jt < njt) {
                    accs[jt] += w * x[(int64_t)(jt0 + jt)*n + j];
                }
            }
        }
    }
#pragma unroll
    for (int jt = 0; jt < 8; ++jt) {
        if (jt < njt) {
            dst[(int64_t)(jt0 + jt)*vocab + r] = accs[jt];
        }
    }
}

// warp-per-row head for small nt (2 <= nt <= 8): the row-per-thread kernel
// above streams each 1280 B row from ONE thread - a 32-way uncoalesced
// pattern that measured 1.34 ms against a 0.39 ms bandwidth floor at nt=2
// (250 GB/s effective). This form is the tg kernel's shape with NTT token
// accumulators per lane: lanes stride the row's 20-byte code blocks, x reads
// come from L2 (the whole x tile is a few KB and stays resident), and the
// extraction arithmetic is the tg kernel's bit for bit.
template <int NTT>
static __global__ void mach1_head_mm_warp_kernel(
        const uint8_t * __restrict__ qp,
        const half    * __restrict__ gscale,
        const float   * __restrict__ x,
        float         * __restrict__ dst,
        const int n, const int vocab, const int n_tokens) {
    constexpr int WG        = 256;
    constexpr int warp_size = 32;
    // staged x with the tg kernel's padded stride: reading x from global
    // here measured 2886 us at nt=2 - WORSE than the row-per-thread form -
    // because the per-lane j stride is 1024 B, 64 uncoalesced L2 hits per
    // iteration (the r270 census; the same lesson the tg kernel's SP=33
    // comment records for shared banks). n <= 2048 host-guarded: NTT=4 at
    // the 4096 cap would be 67 KB of static shared.
    __shared__ float shx[NTT][(2048/32)*33];

    const int tid  = threadIdx.x;
    const int lane = tid % warp_size;
    const int wid  = tid / warp_size;
    const int r    = blockIdx.x*(WG/warp_size) + wid;
    const int t0   = blockIdx.y*NTT;
    const int ntc  = min(NTT, n_tokens - t0);

    for (int tt = 0; tt < ntc; ++tt) {
        for (int i = tid; i < n; i += WG) {
            shx[tt][(i >> 5)*33 + (i & 31)] = x[(int64_t)(t0 + tt)*n + i];
        }
    }
    __syncthreads();
    if (r >= vocab) {
        return;   // whole warp exits; no further syncs
    }

    const int64_t row_bytes = (int64_t) n/8*5;
    const int     ng        = n/64;
    const uint32_t * pw = (const uint32_t *)(qp + (int64_t) r*row_bytes);

    float acc[NTT];
#pragma unroll
    for (int tt = 0; tt < NTT; ++tt) {
        acc[tt] = 0.0f;
    }
    for (int b4i = lane; b4i < n/32; b4i += warp_size) {
        uint32_t w5[5];
#pragma unroll
        for (int q = 0; q < 5; ++q) {
            w5[q] = pw[b4i*5 + q];
        }
        const float gsc = __half2float(gscale[(int64_t) r*ng + (b4i >> 1)]);
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int bit0 = 5*i;
            const int wi   = bit0 >> 5;
            const int o    = bit0 & 31;
            uint32_t qv = w5[wi] >> o;
            if (o > 27) {
                qv |= w5[wi + 1] << (32 - o);   // straddles a word boundary
            }
            qv &= 31u;
            const float w = (float)((int) qv - 16) * gsc;
#pragma unroll
            for (int tt = 0; tt < NTT; ++tt) {
                if (tt < ntc) {
                    acc[tt] += w * shx[tt][b4i*33 + i];
                }
            }
        }
    }
#pragma unroll
    for (int tt = 0; tt < NTT; ++tt) {
        const float s = warp_reduce_sum<warp_size>(acc[tt]);
        if (lane == 0 && tt < ntc) {
            dst[(int64_t)(t0 + tt)*vocab + r] = s;
        }
    }
}

// p16 high-load output head: decode the existing int5 row stream directly
// into warp-local tensor-core tiles.  A warp owns eight vocabulary rows and
// accumulates all 16 tokens with m16n8k16; eight warps therefore cover 64
// rows per CTA.  There is no decoded global bank or intermediate output.
//
// A 128-column strip is the useful unit for both sides of the kernel:
//   - each packed row contributes exactly 80 B (20 aligned uint32 words),
//   - its two fp16 group scales cover the strip exactly, and
//   - eight k16 MMA steps consume the decoded tile before it is overwritten.
// Four adjacent lanes cooperate on each packed row.  Their first uint4 loads
// cover the first 64 B contiguously; lane zero fetches the final 16 B.  The
// staged word layout is deliberately [row][20]: for a fixed one of the five
// decode words, bank = 20*row + 5*segment + word is a permutation of all 32
// banks.  Decoded halves use mach1_tc_wt_off(), the same compact conflict-free
// layout as the compressed-spine MMA path.
//
// Exact n/nt/vocab gating on the host pins the audited layout and removes all
// tail predicates.  Dynamic shared memory is 29 KiB:
//   activation tile  16 * 128 half                    =  4 KiB
//   decoded weights   8 * (128 * 8) half              = 16 KiB
//   packed strips     8 * (8 * 20) uint32             =  5 KiB
//   row scales        8 * (8 * 32) half               =  4 KiB
// NTOK < 16 zero-fills activation rows [NTOK, 16) once and never restages
// them; a live token row's fragment math is identical to the 16-token form,
// so per-row results match the certified kernel (NTOK=16 is the exact body).
template <int NTOK = 16>
static __global__ __launch_bounds__(256, 3) void mach1_head_mm_mma16_kernel(
        const uint8_t * __restrict__ qp,
        const half    * __restrict__ gscale,
        const float   * __restrict__ x,
        float         * __restrict__ dst,
        const int vocab) {
#ifdef AMPERE_MMA_AVAILABLE
    constexpr int W          = 8;
    constexpr int WG         = W*32;
    constexpr int ROWS       = 8;
    constexpr int NT         = 16;   // staged tile height; NTOK rows are live
    constexpr int N          = 2048;
    constexpr int NG         = N/64;
    constexpr int KT         = 128;
    constexpr int KWORDS     = KT*5/32;
    constexpr int ROW_WORDS  = N*5/32;
    constexpr int WTILE      = KT*ROWS;
    static_assert(NTOK == 16 || NTOK == 8 || NTOK == 4 || NTOK == 2,
                  "token tile heights: 16 (certified) or the NTLOW 8/4/2");

    const int tid     = threadIdx.x;
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int row8    = lane >> 2;
    const int segment = lane & 3;
    const int r0      = ((int) blockIdx.x*W + warp)*ROWS;
    const int r       = r0 + row8;

    extern __shared__ unsigned char smem_raw[];
    half * s_a = (half *) smem_raw;                          // [NT, KT]
    half * s_w = s_a + NT*KT;                                // [W, KT*ROWS]
    uint32_t * s_packed = (uint32_t *) (s_w + W*WTILE);      // [W, ROWS, KWORDS]
    half * s_scale = (half *) (s_packed + W*ROWS*KWORDS);    // [W, ROWS, NG]

    half * wb = s_w + warp*WTILE;
    uint32_t * packed_row = s_packed + (warp*ROWS + row8)*KWORDS;
    half * scale_row = s_scale + (warp*ROWS + row8)*NG;

    // All 32 scales for a row are only 64 B.  Stage them once with four
    // adjacent 16-byte transactions instead of paying one sparse sector per
    // row at each of the 16 K strips.
    *(uint4 *) (scale_row + segment*8) =
        *(const uint4 *) (gscale + (int64_t) r*NG + segment*8);

    // rows [NTOK, NT) of the activation slab are never refilled below, so one
    // zero pass keeps the dead half of the A fragment out of the accumulator
    if constexpr (NTOK < NT) {
        for (int i = tid + NTOK*KT; i < NT*KT; i += WG) {
            s_a[i] = __float2half(0.0f);
        }
    }

    mach1_tc_frag_c acc;
    acc.f[0] = acc.f[1] = acc.f[2] = acc.f[3] = 0.0f;

#pragma unroll 1
    for (int ks = 0; ks < N/KT; ++ks) {
        // The 16x128 activation slab is shared by all eight warps.  It is
        // tiny enough to remain a direct fp32 -> fp16 conversion; no global
        // activation scratch is introduced.
        for (int i = tid; i < NTOK*KT; i += WG) {
            const int tt = i/KT;
            const int k  = i - tt*KT;
            s_a[i] = __float2half_rn(x[(int64_t) tt*N + ks*KT + k]);
        }

        const uint32_t * pw = (const uint32_t *)
            (qp + (int64_t) r*(ROW_WORDS*sizeof(uint32_t)));
        const uint4 p4 = *(const uint4 *) (pw + ks*KWORDS + segment*4);
        *(uint4 *) (packed_row + segment*4) = p4;
        if (segment == 0) {
            *(uint4 *) (packed_row + 16) = *(const uint4 *) (pw + ks*KWORDS + 16);
        }
        __syncthreads();

        uint32_t w5[5];
#pragma unroll
        for (int q = 0; q < 5; ++q) {
            w5[q] = packed_row[segment*5 + q];
        }
        const float gsc = __half2float(scale_row[ks*2 + (segment >> 1)]);

        // One lane expands 32 consecutive codes as four aligned half8
        // chunks.  k0 is always 32-aligned, so each lane stays wholly inside
        // one 64-weight scale group.
#pragma unroll
        for (int h8 = 0; h8 < 4; ++h8) {
            __align__(16) half hv8[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int qi   = h8*8 + i;
                const int bit0 = 5*qi;
                const int wi   = bit0 >> 5;
                const int off  = bit0 & 31;
                uint32_t qv = w5[wi] >> off;
                if (off > 27) {
                    qv |= w5[wi + 1] << (32 - off);
                }
                qv &= 31u;
                hv8[i] = __float2half_rn((float)((int) qv - 16)*gsc);
            }
            const int k8   = segment*32 + h8*8;
            const int kt16 = k8 >> 4;
            const int kg8  = (k8 >> 3) & 1;
            *(uint4 *) (wb + kt16*(16*ROWS) + mach1_tc_wt_off(row8, kg8)) =
                *(const uint4 *) hv8;
        }
        __syncwarp();

#pragma unroll
        for (int kk = 0; kk < KT/16; ++kk) {
            mach1_tc_frag_a a;
            mach1_tc_frag_b b;
            const int ar = lane & 15;
            const int ac = (lane >> 4)*8;
            mach1_tc_ldmatrix_x4(s_a + ar*KT + kk*16 + ac, a);
            mach1_tc_ldmatrix_x2(
                wb + kk*(16*ROWS) + mach1_tc_wt_off(lane & 7, (lane >> 3) & 1), b);
            mach1_tc_mma(acc, a, b);
        }
        // No warp may overwrite its decoded tile, and no thread may overwrite
        // the shared activation tile, until every warp has issued its MMA.
        __syncthreads();
    }

    // Fragment-C mapping: every four adjacent lanes own one token row and
    // eight adjacent vocab columns, so the two scalar stores per lane form a
    // single coalesced 32-byte row transaction.
    const int cr = lane >> 2;
    const int cc = (lane & 3)*2;
    if constexpr (NTOK == NT) {
        dst[(int64_t)(cr + 0)*vocab + r0 + cc + 0] = acc.f[0];
        dst[(int64_t)(cr + 0)*vocab + r0 + cc + 1] = acc.f[1];
        dst[(int64_t)(cr + 8)*vocab + r0 + cc + 0] = acc.f[2];
        dst[(int64_t)(cr + 8)*vocab + r0 + cc + 1] = acc.f[3];
    } else {
        if (cr < NTOK) {
            dst[(int64_t)(cr + 0)*vocab + r0 + cc + 0] = acc.f[0];
            dst[(int64_t)(cr + 0)*vocab + r0 + cc + 1] = acc.f[1];
        }
        if (cr + 8 < NTOK) {
            dst[(int64_t)(cr + 8)*vocab + r0 + cc + 0] = acc.f[2];
            dst[(int64_t)(cr + 8)*vocab + r0 + cc + 1] = acc.f[3];
        }
    }
#else
    GGML_UNUSED(qp); GGML_UNUSED(gscale); GGML_UNUSED(x); GGML_UNUSED(dst);
    GGML_UNUSED(vocab);
    NO_DEVICE_CODE;
#endif
}

// head fold fill: decode the int5 code stream to an fp16 bank [vocab, n]
// once (same extraction and (q - 16) * gscale product as the mm kernels)
static __global__ void mach1_head_fold_decode_kernel(
        const uint8_t * __restrict__ qp,
        const half    * __restrict__ gscale,
        half          * __restrict__ bank,
        const int n, const int64_t vocab) {
    const int64_t gid = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (gid >= vocab*n) {
        return;
    }
    const int64_t r = gid / n;
    const int     j = (int)(gid % n);
    const int64_t row_bytes = (int64_t) n/8*5;
    const int     ng        = n/64;
    const uint32_t * pw = (const uint32_t *)(qp + r*row_bytes);
    const int bit0 = 5*j;
    const int wi   = bit0 >> 5;
    const int o    = bit0 & 31;
    uint32_t qv = pw[wi] >> o;
    if (o > 27) {
        qv |= pw[wi + 1] << (32 - o);
    }
    qv &= 31u;
    const float w = (float)((int) qv - 16) * __half2float(gscale[r*ng + (j >> 6)]);
    bank[gid] = __float2half_rn(w);
}

// decode variant: one WARP per vocab row, lanes stride the 32-code word
// groups. The thread-per-row kernel above strides rows 1280 B apart, so at
// nt = 1 a warp's 32 uint32 loads land in 32 different sectors (~8x read
// amplification on a 318 MB stream — measured 1.4 ms/token on H200, ~21x the
// bandwidth floor). Here lanes read adjacent word groups of the SAME row, x
// is staged in shared once per block, and the row dot ends in a warp
// reduction. Code extraction and the per-weight (q - 16) * gscale product are
// identical; only the dot's accumulation order changes (contractible, like
// every mach1 dot).
// STG: stage the warp's packed row into shared, coalesced, instead of
// reading it with a 20-byte lane stride.
//
// A warp owns one output row of row_bytes = n*5/8. The direct form has lane L
// read pw[5L .. 5L+5], so each of its five LDG.32 spans 640 bytes across the
// warp - 20 sectors moved per instruction to deliver 128 useful bytes. Read
// the row cooperatively instead (lane L takes pw[L], pw[L+32], ...: 32
// consecutive words, 4 sectors per instruction) and take the five words from
// shared. gcd(5, 32) = 1, so srow[5*b4i + q] with b4i = lane puts all 32
// lanes on distinct banks - conflict free. Same words, same order: bit-exact.
template <int SP, int STG>
static __global__ void mach1_head_mm_tg_kernel(
        const uint8_t * __restrict__ qp,
        const half    * __restrict__ gscale,
        const float   * __restrict__ x,
        float         * __restrict__ dst,
        const int n, const int vocab) {
    constexpr int WG        = 256;
    constexpr int warp_size = 32;
    // PADDED STRIDE. The activation is read as shx[b4i*32 + i] with b4i = lane
    // (+ 32k), so with a flat layout every lane of the warp lands on bank
    // (32*lane + i) % 32 = i % 32 - the SAME bank for all 32 lanes, a 32-way
    // conflict on every one of the 32 loads this loop does per iteration.
    // Storing 33 floats per 32-wide group makes the bank (lane + i) % 32, so
    // the warp hits 32 distinct banks. Same values, same order: bit-exact.
    __shared__ float shx[(4096/32)*SP];   // n <= 4096, host-guarded
    // n <= 4096 -> at most 4096*5/32 = 640 uint32 per row, 8 warps per block
    __shared__ uint32_t srow[STG ? 8 : 1][STG ? 640 : 1];

    const int tid  = threadIdx.x;
    const int lane = tid % warp_size;
    const int wid  = tid / warp_size;
    const int r    = blockIdx.x*(WG/warp_size) + wid;

    for (int i = tid; i < n; i += WG) {
        shx[(i >> 5)*SP + (i & 31)] = x[i];
    }
    __syncthreads();
    if (r >= vocab) {
        return;   // whole warp exits; no further syncs
    }

    const int64_t row_bytes = (int64_t) n/8*5;
    const int     ng        = n/64;
    const uint32_t * pw = (const uint32_t *)(qp + (int64_t) r*row_bytes);

    if (STG) {
        const int nw = n*5/32;                 // uint32 per packed row
        for (int i = lane; i < nw; i += warp_size) {
            srow[wid][i] = pw[i];
        }
        __syncwarp();
    }

    float acc = 0.0f;
    for (int b4i = lane; b4i < n/32; b4i += warp_size) {
        uint32_t w5[5];
#pragma unroll
        for (int q = 0; q < 5; ++q) {
            w5[q] = STG ? srow[wid][b4i*5 + q] : pw[b4i*5 + q];
        }
        // j = b4i*32 + i, i < 32  =>  j >> 6 == b4i >> 1
        const float gsc = __half2float(gscale[(int64_t) r*ng + (b4i >> 1)]);
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int bit0 = 5*i;
            const int wi   = bit0 >> 5;
            const int o    = bit0 & 31;
            uint32_t qv = w5[wi] >> o;
            if (o > 27) {
                qv |= w5[wi + 1] << (32 - o);   // straddles a word boundary
            }
            qv &= 31u;
            const float w = (float)((int) qv - 16) * gsc;
            acc += w * shx[b4i*SP + i];
        }
    }
    acc = warp_reduce_sum<warp_size>(acc);
    if (lane == 0) {
        dst[r] = acc;
    }
}

void ggml_cuda_op_mach1_head_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * qp     = dst->src[0];
    const ggml_tensor * gscale = dst->src[1];
    const ggml_tensor * x      = dst->src[2];

    GGML_ASSERT(qp->type     == GGML_TYPE_I8);
    GGML_ASSERT(gscale->type == GGML_TYPE_F16);
    GGML_ASSERT(x->type      == GGML_TYPE_F32);
    GGML_ASSERT(dst->type    == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(qp));
    GGML_ASSERT(ggml_is_contiguous(gscale));
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int n     = (int) x->ne[0];
    const int vocab = (int) qp->ne[1];
    const int nt    = (int)(x->ne[1]*x->ne[2]*x->ne[3]);

    char shp[64];
    snprintf(shp, sizeof(shp), " n=%d vocab=%d nt=%d", n, vocab, nt);

    // Warm the head bank on ANY head call (the untimed nt=16 warmup lands
    // here first) so the ~1 GiB fill never sits inside a timed nt==1 leg -
    // the fill-in-window flip-flop on the hb walls. Fill only; serving is
    // the nt==1 block below the mma16 dispatch.
    static const bool head_bank = mach1_env_int("GGML_MACH1_HEAD_BANK", 0) != 0;
    if (mach1_fold_enabled() || head_bank) {
        const auto warm_fill = [&](half * bdst) {
            const int64_t total = (int64_t) vocab*n;
            mach1_launch(mach1_head_fold_decode_kernel,
                ggml_cuda_kernel_launch_params(dim3((unsigned)((total + 255)/256), 1, 1),
                                               dim3(256, 1, 1), 0, ctx.stream()),
                (const uint8_t *) qp->data, (const half *) gscale->data, bdst, n, (int64_t) vocab);
        };
        (void) mach1_bank_get(qp->data, (size_t) vocab*n, ctx.device, ctx.stream(), warm_fill, head_bank);
    }

    // Default-off p16 tensor-core head.  This exact shape is the only audited
    // layout: it covers the vocabulary without a tail CTA and consumes 29 KiB
    // of per-CTA shared memory, with no persistent or transient global bank.
    static const bool head_mma16_on = mach1_env_int("GGML_MACH1_HEAD_MMA16", 0) != 0;
    const auto & head_mma16_dev = ggml_cuda_info().devices[ctx.device];
    // nt32/ntlow: the audited 16-token kernel serves any admitted batch as
    // {16,8,4,2}-token chunks (x rows are token-major stride n, dst rows
    // token-major stride vocab, so a chunk is a plain pointer offset). The
    // nt == 16 launch is byte-identical.
    const bool head_mma16_nt = mach1_nt_ok(nt);
    if (head_mma16_on && head_mma16_nt && n == 2048 && vocab == 248320 &&
            ampere_mma_available(head_mma16_dev.cc) && head_mma16_dev.warp_size == 32) {
        constexpr int W = 8;
        constexpr int ROWS = 8;
        constexpr int KT = 128;
        constexpr int NG = 2048/64;
        constexpr int KWORDS = KT*5/32;
        constexpr size_t smem =
            16*KT*sizeof(half) +
            W*(KT*ROWS)*sizeof(half) +
            W*(ROWS*KWORDS)*sizeof(uint32_t) +
            W*(ROWS*NG)*sizeof(half);
        static_assert(smem == 29*1024, "p16 head MMA shared-memory budget changed");
        mach1_smem_opt_in((const void *) mach1_head_mm_mma16_kernel<16>, smem);
        if (nt != 16) {
            mach1_smem_opt_in((const void *) mach1_head_mm_mma16_kernel< 8>, smem);
            mach1_smem_opt_in((const void *) mach1_head_mm_mma16_kernel< 4>, smem);
            mach1_smem_opt_in((const void *) mach1_head_mm_mma16_kernel< 2>, smem);
        }
        const ggml_cuda_kernel_launch_params hp =
            ggml_cuda_kernel_launch_params(dim3((unsigned)(vocab/(W*ROWS)), 1, 1),
                                           dim3(W*32, 1, 1), smem, ctx.stream());
        if (!mach1_ablate(MACH1_ABLATE_HEAD)) {
            mach1_timed(ctx.stream(), std::string("head_mm_mma16") + shp, [&]() {
                for (int t0 = 0; t0 < nt; ) {
                    const int nc = mach1_nt_chunk(nt - t0);
                    const uint8_t * qpd = (const uint8_t *) qp->data;
                    const half    * gsd = (const half *) gscale->data;
                    const float   * xc  = (const float *) x->data + (int64_t) t0*n;
                    float         * dc  = (float *) dst->data + (int64_t) t0*vocab;
                    switch (nc) {
                        case 16: mach1_launch(mach1_head_mm_mma16_kernel<16>, hp, qpd, gsd, xc, dc, vocab); break;
                        case 8:  mach1_launch(mach1_head_mm_mma16_kernel< 8>, hp, qpd, gsd, xc, dc, vocab); break;
                        case 4:  mach1_launch(mach1_head_mm_mma16_kernel< 4>, hp, qpd, gsd, xc, dc, vocab); break;
                        default: mach1_launch(mach1_head_mm_mma16_kernel< 2>, hp, qpd, gsd, xc, dc, vocab); break;
                    }
                    t0 += nc;
                }
            });
            if (mach1_debug_on()) {
                static std::once_flag once;
                std::call_once(once, [&]() {
                    fprintf(stderr, "mach1: zero-VRAM compressed head_mma16 nt=%d ENGAGED\n", nt);
                });
            }
        }
        return;
    }

    // FOLD tier: fp16 head bank (~2 B/weight vs the 5-bit stream, but served
    // by the tuned mmv; the tg kernel measured 0.55 ms vs a ~0.25 ms bank read).
    // HEAD_BANK scopes the same bank to the HEAD ONLY - the rest of the stack
    // keeps the certified fused regions (their matchers bail under FOLD=1).
    // Tolerance-class: mmvf fp32 order differs from the warp head; gate with
    // intra-batch AGREE, not the byte sha. ~1.0 GiB VRAM, filled once.
    // The bank is nt-independent (vocab x n fp16), so FILL it on any head
    // call - the untimed nt=16 warmup lands here first, keeping the ~1 GiB
    // decode out of the timed nt==1 legs (the fill-in-window flip-flop on
    // the hb walls). SERVING stays nt==1; the chunked mma16 head owns nt>1
    // (the D-prime GemmEx raced it neutral and was removed - see the ledger).
    // Serving: mmvf at nt==1; ONE GemmEx bank read at nt 2-4 (min8 floored
    // the chunked mma16 out of 2-4, so the 500-670us warp head returned as
    // the fallback there - the bank GEMM beats it by ~2x; rung H).
    if ((mach1_fold_enabled() || head_bank) && nt <= 4) {
        const auto head_fill = [&](half * bdst) {
            const int64_t total = (int64_t) vocab*n;
            mach1_launch(mach1_head_fold_decode_kernel,
                ggml_cuda_kernel_launch_params(dim3((unsigned)((total + 255)/256), 1, 1),
                                               dim3(256, 1, 1), 0, ctx.stream()),
                (const uint8_t *) qp->data, (const half *) gscale->data, bdst, n, (int64_t) vocab);
        };
        half * wb = mach1_bank_get(qp->data, (size_t) vocab*n, ctx.device, ctx.stream(), head_fill, head_bank);
        if (wb != nullptr && nt == 1) {
            if (!mach1_ablate(MACH1_ABLATE_HEAD))
            mach1_timed(ctx.stream(), std::string("head_fold_mv") + shp, [&]() {
            mach1_fold_mmvf(ctx, (const half *) wb, (const float *) x->data, (float *) dst->data, n, vocab);
            });
            return;
        }
        if (wb != nullptr) {
            // nt 2-4: ONE bank read serves every token - the skinny GemmEx
            // pattern (per-token mmvf loops would re-read the ~1 GiB bank).
            // Tolerance-class like the mmvf path; gate with AGREE, not sha.
            if (!mach1_ablate(MACH1_ABLATE_HEAD)) {
            cudaStream_t hstream = ctx.stream();
            ggml_cuda_pool_alloc<half> hx16_alloc(ctx.pool(), (size_t) n*nt);
            half * hx16 = hx16_alloc.get();
            mach1_timed(hstream, std::string("head_bank_cvt") + shp, [&]() {
                const int64_t total = (int64_t) n*nt;
                mach1_launch(mach1_rt_u16_kernel,
                    ggml_cuda_kernel_launch_params(
                        dim3((unsigned) ((total + 255)/256), 1, 1), dim3(256, 1, 1), 0, hstream),
                    (const float *) x->data, hx16, total);
            });
            mach1_timed(hstream, std::string("head_bank_gemm") + shp, [&]() {
                static std::map<int64_t, cublasHandle_t> hbmap;
                static std::mutex hbmu;
                cublasHandle_t cbh;
                {
                    std::lock_guard<std::mutex> lk(hbmu);
                    auto & hh = hbmap[((int64_t) vocab << 20) | (int64_t) nt];
                    if (hh == nullptr) {
                        CUBLAS_CHECK(cublasCreate(&hh));
                        static void * hws = nullptr;
                        static const size_t hwssz = 32u*1024u*1024u;
                        if (hws == nullptr) {
                            CUDA_CHECK(cudaMalloc(&hws, hwssz));
                        }
                        CUBLAS_CHECK(cublasSetWorkspace(hh, hws, hwssz));
                    }
                    cbh = hh;
                }
                CUBLAS_CHECK(cublasSetStream(cbh, hstream));
                const float onef = 1.0f, zerof = 0.0f;
                CUBLAS_CHECK(cublasGemmEx(cbh, CUBLAS_OP_T, CUBLAS_OP_N,
                    (int) vocab, (int) nt, (int) n,
                    &onef,  wb,   CUDA_R_16F, (int) n,
                            hx16, CUDA_R_16F, (int) n,
                    &zerof, dst->data, CUDA_R_32F, (int) vocab,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            });
            if (mach1_debug_on()) {
                static std::once_flag once_hbg;
                std::call_once(once_hbg, [&]() {
                    fprintf(stderr, "mach1: head bank GEMM ENGAGED (nt=%d)\n", nt);
                });
            }
            }
            return;
        }
    }

    if (nt == 1 && n <= 4096 && n % 64 == 0) {
        static const bool hpad = mach1_env_int("GGML_MACH1_HEAD_PAD", 1) != 0;
        // MEASURED NEUTRAL (H200, paired, 3 rounds): 83.99 with staging vs 84.28
        // without. The 5x sector reduction is real but the head stopped being
        // memory-instruction bound once the 32-way bank conflict was fixed.
        // Default OFF; kept for the record.
        static const bool hstg = mach1_env_int("GGML_MACH1_HEAD_STAGE", 0) != 0;
        if (!mach1_ablate(MACH1_ABLATE_HEAD))
        mach1_timed(ctx.stream(), std::string("head_mm_tg") + shp, [&]() {
        const ggml_cuda_kernel_launch_params hp =
            ggml_cuda_kernel_launch_params(dim3((unsigned)((vocab + 7)/8), 1, 1),
                                           dim3(256, 1, 1), 0, ctx.stream());
        const uint8_t * qp_d = (const uint8_t *) qp->data;
        const half    * gs_d = (const half    *) gscale->data;
        const float   * x_d  = (const float   *) x->data;
        float         * d_d  = (float         *) dst->data;
        if (hpad && hstg) {
            mach1_launch(mach1_head_mm_tg_kernel<33, 1>, hp, qp_d, gs_d, x_d, d_d, n, vocab);
        } else if (hpad) {
            mach1_launch(mach1_head_mm_tg_kernel<33, 0>, hp, qp_d, gs_d, x_d, d_d, n, vocab);
        } else {
            mach1_launch(mach1_head_mm_tg_kernel<32, 0>, hp, qp_d, gs_d, x_d, d_d, n, vocab);
        }
        });
        return;
    }

    // small-nt decode batches: the warp-per-row form (see the kernel comment;
    // the row-per-thread fallback below is 3x off roofline here). Default on;
    // GGML_MACH1_HEAD_WARP=0 restores the old form for A/B.
    static const bool head_warp = mach1_env_int("GGML_MACH1_HEAD_WARP", 1) != 0;
    if (head_warp && nt >= 2 && nt <= 8 && n % 64 == 0 && n <= 2048) {
        const int ntt = nt >= 4 ? 4 : 2;
        const ggml_cuda_kernel_launch_params wp =
            ggml_cuda_kernel_launch_params(dim3((unsigned)((vocab + 7)/8), (unsigned)((nt + ntt - 1)/ntt), 1),
                                           dim3(256, 1, 1), 0, ctx.stream());
        if (!mach1_ablate(MACH1_ABLATE_HEAD))
        mach1_timed(ctx.stream(), std::string("head_mm_warp") + shp, [&]() {
        if (ntt == 4) {
            mach1_launch(mach1_head_mm_warp_kernel<4>, wp,
                (const uint8_t *) qp->data, (const half *) gscale->data,
                (const float *) x->data, (float *) dst->data, n, vocab, nt);
        } else {
            mach1_launch(mach1_head_mm_warp_kernel<2>, wp,
                (const uint8_t *) qp->data, (const half *) gscale->data,
                (const float *) x->data, (float *) dst->data, n, vocab, nt);
        }
        });
        return;
    }

    const dim3 grid((unsigned)((vocab + 127)/128), (unsigned)((nt + 7)/8), 1);
    if (!mach1_ablate(MACH1_ABLATE_HEAD))
    mach1_timed(ctx.stream(), std::string("head_mm") + shp, [&]() {
    mach1_launch(mach1_head_mm_kernel,
        ggml_cuda_kernel_launch_params(grid, dim3(128, 1, 1), 0, ctx.stream()),
        (const uint8_t *) qp->data,
        (const half    *) gscale->data,
        (const float   *) x->data,
        (float         *) dst->data,
        n, vocab, nt);
    });
}

//
// NE_MM — one thread per output row x 8-token tile, serial L=12 walk
// (mach1_ne_mm.comp)
//

static __global__ void mach1_ne_mm_kernel(
        const uint8_t * __restrict__ packed,
        const half    * __restrict__ gscale,
        const half    * __restrict__ lut,
        const float   * __restrict__ x,
        float         * __restrict__ dst,
        const int B, const int T, const int k, const int ng,
        const int rows_per_chunk, const int n_tokens) {
    const int r = blockIdx.x*blockDim.x + threadIdx.x;
    if (r >= B) {
        return;
    }
    const int jt0 = blockIdx.y*8;
    const int njt = min(8, n_tokens - jt0);

    const int64_t  row_bytes = (int64_t) T*k/8;
    const int64_t  lut_off   = (int64_t)(r / rows_per_chunk)*4096;
    const uint32_t kmask     = (1u << k) - 1;

    float accs[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        accs[j] = 0.0f;
    }

    const uint8_t * pb = packed + (int64_t) r*row_bytes;
    uint32_t acc   = 0;
    uint32_t state = 0;
    int      nbits = 0;
    for (int g = 0; g < ng; ++g) {
        const float gsc = __half2float(gscale[(int64_t) r*ng + g]);
        for (int t = 0; t < 128; ++t) {
            while (nbits < k) {
                acc = (acc << 8) | *pb++;
                nbits += 8;
            }
            nbits -= k;
            state = ((state << k) | ((acc >> nbits) & kmask)) & 0xFFFu;
            // reference decode: f32(lut) * f32(gscale), one fp32 product
            const float w = __half2float(lut[lut_off + state]) * gsc;
            const int idx = g*128 + t;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (j < njt) {
                    accs[j] += w * x[(int64_t)(jt0 + j)*T + idx];
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (j < njt) {
            dst[(int64_t)(jt0 + j)*B + r] = accs[j];
        }
    }
}

void ggml_cuda_op_mach1_ne_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * packed = dst->src[0];
    const ggml_tensor * gscale = dst->src[1];
    const ggml_tensor * lut    = dst->src[2];
    const ggml_tensor * x      = dst->src[3];

    GGML_ASSERT(packed->type == GGML_TYPE_I8);
    GGML_ASSERT(gscale->type == GGML_TYPE_F16);
    GGML_ASSERT(lut->type    == GGML_TYPE_F16);
    GGML_ASSERT(x->type      == GGML_TYPE_F32);
    GGML_ASSERT(dst->type    == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(packed));
    GGML_ASSERT(ggml_is_contiguous(gscale));
    GGML_ASSERT(ggml_is_contiguous(lut));
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int T   = (int) x->ne[0];
    const int B   = (int) packed->ne[1];
    const int k   = (int)(packed->ne[0]*8 / T);
    const int ng  = (int) gscale->ne[0];
    const int rpc = (int)(B / lut->ne[1]);
    const int nt  = (int)(x->ne[1]*x->ne[2]*x->ne[3]);

    const dim3 grid((unsigned)((B + 127)/128), (unsigned)((nt + 7)/8), 1);
    char shp[64];
    snprintf(shp, sizeof(shp), " B=%d T=%d k=%d nt=%d", B, T, k, nt);
    mach1_timed(ctx.stream(), std::string("ne_mm") + shp, [&]() {
    mach1_launch(mach1_ne_mm_kernel,
        ggml_cuda_kernel_launch_params(grid, dim3(128, 1, 1), 0, ctx.stream()),
        (const uint8_t *) packed->data,
        (const half    *) gscale->data,
        (const half    *) lut->data,
        (const float   *) x->data,
        (float         *) dst->data,
        B, T, k, ng, rpc, nt);
    });
}

//
// RT_MM — 3 kernels: u = H(su ⊙ x) per token -> K4 V2 trellis walk ->
// y = sv ⊙ H(v) (mach1_rt_u/rt_walk/rt_out.comp). Scratch: u [nt, n] then
// v [nt, m] in one pool allocation.
//

// WG is a template knob (GGML_MACH1_FWHT_WG): every loop strides by WG and
// each FWHT pass is elementwise, so the results are bit-identical for any WG.
template <int WG, bool PACKED_F16 = false>
__global__ void mach1_rt_u_kernel(
        const float * __restrict__ su,
        const float * __restrict__ x,
        float       * __restrict__ scr_u,
        const int n) {
    __shared__ float sh[4096];

    const int t   = blockIdx.x;
    const int tid = threadIdx.x;

    for (int i = tid; i < n; i += WG) {
        sh[i] = su[i] * x[(int64_t) t*n + i];
    }
    __syncthreads();
    mach1_fwht_block(sh, n, tid, WG);
    const float sc = __fsqrt_rn((float) n);
    for (int i = tid; i < n; i += WG) {
        const int64_t oi = (int64_t) t*n + i;
        const float v = __fdiv_rn(sh[i], sc);
        if constexpr (PACKED_F16) {
            ((half *) scr_u)[oi] = __float2half(v);
        } else {
            scr_u[oi] = v;
        }
    }
}

// --- the rt integer-lattice walk (GGML_MACH1_ZDP on the spine) ---
//
// The K = 4 spine alphabet is a 64-level ladder (levels = s0*z, z integer).
// zslut packs each tlut row's pair as raw int8 bytes (z1 << 8) | z0, rows
// 512..1023 with z0 negated, so (ph >> 6) & 1023 resolves the row AND the
// sign bit in one mask. A row of 8 states then costs 4 dp4a against the
// thread's activations quantized to q8 once, instead of 16 smem floats,
// 8 selects and 16 fp32 MACs. Scales (s0, the q8 step) associate outside
// the integer sum - the same tolerance class as the exp-tier ZDP.
struct mach1_rt_ztab {
    const uint16_t * ztab;   // device [1024], (z1 << 8) | z0, sign-resolved
    float            zstep;  // s0
};

static mach1_rt_ztab mach1_rt_z_table(const void * tlut_data, cudaStream_t stream) {
    struct entry { const uint16_t * ztab; float zstep; };
    static std::mutex mtx;
    static std::unordered_map<const void *, entry> cache;

    std::lock_guard<std::mutex> lock(mtx);
    const auto it = cache.find(tlut_data);
    if (it != cache.end()) {
        return {it->second.ztab, it->second.zstep};
    }

    std::vector<uint16_t> host(1024);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), tlut_data, host.size()*sizeof(uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float s0 = 0.0f;
    for (int i = 0; i < 1024; ++i) {
        const float a = fabsf(GGML_FP16_TO_FP32((ggml_fp16_t) host[i]));
        if (a > 0.0f && (s0 == 0.0f || a < s0)) {
            s0 = a;
        }
    }
    // one refinement absorbs the fp16 rounding of s0*z at large z
    if (s0 > 0.0f) {
        double num = 0.0, den = 0.0;
        for (int i = 0; i < 1024; ++i) {
            const float v = GGML_FP16_TO_FP32((ggml_fp16_t) host[i]);
            const long  z = lroundf(v/s0);
            if (z != 0) {
                num += fabsf(v);
                den += z < 0 ? -z : z;
            }
        }
        if (den > 0.0) {
            s0 = (float)(num/den);
        }
    }
    bool ladder = s0 > 0.0f;
    int  z[1024];
    for (int i = 0; ladder && i < 1024; ++i) {
        const float v = GGML_FP16_TO_FP32((ggml_fp16_t) host[i]);
        if (v != v) {
            ladder = false;
            break;
        }
        const float r = v/s0;
        z[i] = (int) lroundf(r);
        ladder = fabsf(r - (float) z[i]) <= 0.02f + 0.001f*fabsf(r) && z[i] >= -127 && z[i] <= 127;
    }
    entry e = {nullptr, 0.0f};
    if (ladder) {
        std::vector<uint16_t> zt(1024);
        for (int r = 0; r < 512; ++r) {
            const int z0 = z[2*r + 0], z1 = z[2*r + 1];
            zt[r]       = (uint16_t)(((uint32_t)(uint8_t) z1 << 8) | (uint32_t)(uint8_t) z0);
            zt[512 + r] = (uint16_t)(((uint32_t)(uint8_t) z1 << 8) | (uint32_t)(uint8_t)(-z0));
        }
        void * buf = nullptr;
        CUDA_CHECK(cudaMalloc(&buf, zt.size()*sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpyAsync(buf, zt.data(), zt.size()*sizeof(uint16_t),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        e.ztab  = (const uint16_t *) buf;
        e.zstep = s0;
    }
    cache.emplace(tlut_data, e);
    if (mach1_debug_on() || mach1_time_on()) {
        fprintf(stderr, "mach1: rt z-table %s (s0=%g, tlut=%p)\n",
                ladder ? "ENGAGED" : "FALLBACK (not a ladder)", (double) s0, tlut_data);
    }
    return {e.ztab, e.zstep};
}

// ---------------------------------------------------------------------------
// DEFAULTS NOTE (round 304). The knobs below were raced individually and their
// winning combination lived only in the bench arm `zdpfgt2` - every one of them
// still defaulted OFF, so the shipped build was the UNTUNED configuration.
// Measured on L40S, decode t/s, same container (q4km reproduces at 174.6/175.3
// across runs, so these compare):
//
//   B   default   zdpfgt2   q4km
//   1   138.5     160.4     174.6
//   2   153.2     213.8     277.6
//   4   213.6     323.3     416.0
//
// so the defaults were giving up 16% / 40% / 51%. Promoted here: ZDP,
// TC_WALK, GDN_FULL=2, FORK, MEGA_NT=16, WALK_TT=16, EXP_FUSED_DENSE,
// EXP_ZDP_APPLY=2, GDN_NT=4, TC_NT=16.
//
// NOT promoted: the `zdpfd4` extras (QKVB_WMERGE, FORK_DN, FORK_NT=4,
// QKVB_NT/QKVZ_NT=4). They REGRESS B=1 by 19% (129.7 against 160.4) and are
// still available by env.
//
// ZDP is the single largest of these - 160.4 against 116.6 with it off
// (`exactfg`), i.e. +38% - and it is the one that is NOT bit-exact: it is the
// integer-lattice dp4a walk. Every other flag here is exact restructuring.
// ---------------------------------------------------------------------------
static bool mach1_zdp_on() {
    // ZDP_RT=0 confines the flag to the expert tier for attribution
    static const bool on = mach1_env_int("GGML_MACH1_ZDP", 1) != 0 &&
                           mach1_env_int("GGML_MACH1_ZDP_RT", 1) != 0;
    return on;
}

// quantize a thread's 16 staged activations to q8: qq[k] packs q[4k..4k+3],
// returns the step du (dequant = q*du/127)
static __device__ __forceinline__ float mach1_rt_zdp_quant(
        const float * __restrict__ uu, uint32_t * __restrict__ qq) {
    float du = 0.0f;
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        du = fmaxf(du, fabsf(uu[c]));
    }
    const float qsf = du > 0.0f ? 127.0f/du : 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        uint32_t w = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            w |= (uint32_t)(__float2int_rn(uu[4*k + b]*qsf) & 0xFF) << (8*b);
        }
        qq[k] = w;
    }
    return du;
}

// p16 compressed-spine tensor-core probe. Unlike the dense path, this never
// materializes [m,n] weights: each CTA stages one 128-wide strip of the
// existing trellis codes, expands only its two warp-local 16x32 weight tiles
// in shared memory, and immediately consumes them with m16n8k16 MMA.
//
// The u transform writes its final fp32 result directly as packed f16 rows in
// the front of the already-allocated scr_u region. This removes the separate
// conversion launch and its global round trip while adding no device storage.

// NTOK is the token-tile height: 16 is the certified instantiation, and the
// NTLOW rung adds 8/4/2. The m16n8k16 tile shape, the shared budget and every
// MMA/reduction line are IDENTICAL at all heights - only the global activation
// rows read and the output rows written shrink. Rows [NTOK, 16) of the staged
// activation tile are zeroed once, so their fragments contribute nothing and
// no token outside the batch is ever touched.
template <int NC, int NTOK = 16>
static __global__ __launch_bounds__(128) void mach1_rt_spine_mma_p16_kernel(
        const uint16_t * __restrict__ trellis, // [(m/16)*(n/16), 64]
        const half     * __restrict__ tlut,    // [512, 2], pre-scaled f16
        const half     * __restrict__ u,       // [NTOK, n], packed in scr_u
        float          * __restrict__ out,     // [NTOK, m]
        const int m, const int n) {
#ifdef AMPERE_MMA_AVAILABLE
    constexpr int W       = 4;
    constexpr int NBLK    = 8;
    constexpr int NT      = NC/16;
    constexpr int NF      = NC/8;
    constexpr int KW      = NBLK*16;
    constexpr int WTILE   = NC*16;
    constexpr int WG      = W*32;
    static_assert(NC == 32, "p16 compressed spine is NC32-only");
    static_assert(NTOK == 16 || NTOK == 8 || NTOK == 4 || NTOK == 2,
                  "token tile heights: 16 (certified) or the NTLOW 8/4/2");

    const int nc0  = blockIdx.x*NC;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int NB   = n/16;
    const int mb0  = nc0/16;

    extern __shared__ unsigned char smem_raw[];
    half * s_u = (half *) smem_raw; // [16, KW]
    uint16_t * s_cd = (uint16_t *) (s_u + 16*KW); // [NBLK, NT, 64]
    half * s_w = (half *) (s_cd + NBLK*NT*64);
    half * w_b0 = s_w + warp*2*WTILE;
    half * w_b1 = w_b0 + WTILE;
    float * s_red = (float *) s_w; // overlays weight tiles after the K loop
    half * s_tab = s_w + W*2*WTILE;

    for (int i = threadIdx.x; i < 1024; i += WG) {
        s_tab[i] = tlut[i];
    }
    if constexpr (NTOK < 16) {
        for (int i = threadIdx.x + NTOK*KW; i < 16*KW; i += WG) {
            s_u[i] = __float2half(0.0f);
        }
    }

    mach1_tc_frag_c acc[NF];
#pragma unroll
    for (int i = 0; i < NF; ++i) {
        acc[i].f[0] = 0.0f;
        acc[i].f[1] = 0.0f;
        acc[i].f[2] = 0.0f;
        acc[i].f[3] = 0.0f;
    }

    const int nsteps = NB/NBLK;
    for (int step = 0; step < nsteps; ++step) {
        // 16 rows x 128 halves: 256 aligned 16-byte chunks. The packed u
        // buffer is reused globally and this is the only per-CTA u traffic.
        for (int i = threadIdx.x; i < NTOK*(KW/8); i += WG) {
            const int row = i/(KW/8);
            const int ch  = i%(KW/8);
            ((uint4 *) (s_u + row*KW))[ch] =
                ((const uint4 *) (u + (int64_t) row*n + step*KW))[ch];
        }
        // 128 B per 16x16 trellis tile = eight aligned uint4 chunks.
        for (int i = threadIdx.x; i < NBLK*NT*8; i += WG) {
            const int tile_i = i/8;
            const int ch     = i%8;
            const int kb     = tile_i/NT;
            const int mbi    = tile_i%NT;
            const int64_t tile = (int64_t)(mb0 + mbi)*NB + step*NBLK + kb;
            ((uint4 *) (s_cd + (kb*NT + mbi)*64))[ch] =
                ((const uint4 *) (trellis + tile*64))[ch];
        }
        __syncthreads();

        const auto expand = [&](const int kb, half * wb) {
            const uint16_t * cd0 = s_cd + kb*NT*64;
#pragma unroll
            for (int mbi = 0; mbi < NT; ++mbi) {
                const uint16_t * cd = cd0 + mbi*64;
                __align__(16) half hv8[8];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int s   = lane*4 + i;
                    const int j0  = s*8;
                    const int wa  = j0 >> 4;
                    const int off = j0 & 15;
                    const int wb_i = wa + 1 == 64 ? 0 : wa + 1;
                    const unsigned aw = cd[wa];
                    const unsigned bw = cd[wb_i];
                    const unsigned state = (((aw << 16) | bw) >> (16 - off)) & 0xFFFFu;
                    const unsigned p = state*(state + 1u);
                    const unsigned row = (p >> 6) & 511u;
                    hv8[2*i + 0] = __ushort_as_half(
                        __half_as_ushort(s_tab[2*row + 0]) ^ (uint16_t)(p & 0x8000u));
                    hv8[2*i + 1] = s_tab[2*row + 1];
                }
                *(uint4 *) (wb + mach1_tc_wt_off(mbi*16 + (lane >> 1), lane & 1)) =
                    *(const uint4 *) hv8;
            }
            __syncwarp();
        };

        const auto mma_block = [&](const int kb, const half * wb) {
            mach1_tc_frag_a a;
            const int r  = lane & 15;
            const int c8 = (lane >> 4)*8;
            mach1_tc_ldmatrix_x4(s_u + kb*16 + r*KW + c8, a);
#pragma unroll
            for (int nb8 = 0; nb8 < NF; ++nb8) {
                mach1_tc_frag_b b;
                mach1_tc_ldmatrix_x2(
                    wb + mach1_tc_wt_off(nb8*8 + (lane & 7), (lane >> 3) & 1), b);
                mach1_tc_mma(acc[nb8], a, b);
            }
        };

        const int kb0 = warp*2;
        expand(kb0,     w_b0);
        expand(kb0 + 1, w_b1);
        mma_block(kb0,     w_b0);
        mma_block(kb0 + 1, w_b1);
        __syncthreads();
    }

    // Reuse the weight workspace as a four-warp reduction tile.
    float * red = s_red + warp*16*NC;
    const int rr = lane >> 2;
    const int cc = (lane & 3)*2;
#pragma unroll
    for (int nb8 = 0; nb8 < NF; ++nb8) {
        red[rr*NC + nb8*8 + cc + 0] = acc[nb8].f[0];
        red[rr*NC + nb8*8 + cc + 1] = acc[nb8].f[1];
        red[(rr + 8)*NC + nb8*8 + cc + 0] = acc[nb8].f[2];
        red[(rr + 8)*NC + nb8*8 + cc + 1] = acc[nb8].f[3];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < NTOK*NC; i += WG) {
        const int r = i/NC;
        const int c = i%NC;
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < W; ++w) {
            v += s_red[w*16*NC + i];
        }
        out[(int64_t) r*m + nc0 + c] = v;
    }
#else
    GGML_UNUSED(trellis); GGML_UNUSED(tlut); GGML_UNUSED(u); GGML_UNUSED(out);
    GGML_UNUSED(m); GGML_UNUSED(n);
    NO_DEVICE_CODE;
#endif
}

static void mach1_rt_spine_mma_p16(
        const uint16_t * trellis, const half * tlut, const half * u,
        float * scr_v, const int m, const int n, cudaStream_t stream,
        const int nt = 16) {
    constexpr int NC    = 32;
    constexpr int W     = 4;
    constexpr int NBLK  = 8;
    constexpr int NT    = NC/16;
    constexpr int KW    = NBLK*16;
    constexpr int WTILE = NC*16;
    constexpr size_t smem =
        16*KW*sizeof(half) +
        NBLK*NT*64*sizeof(uint16_t) +
        W*2*WTILE*sizeof(half) +
        1024*sizeof(half);
    static_assert(smem == 16*1024, "p16 compressed-spine shared-memory budget changed");

    const ggml_cuda_kernel_launch_params p =
        ggml_cuda_kernel_launch_params(dim3(m/NC, 1, 1), dim3(W*32, 1, 1), smem, stream);
    // nt == 16 is one chunk at zero offset - the certified launch. Wider or
    // narrower batches issue the same kernel per {16,8,4,2}-token chunk; u
    // rows and out rows are token-major, so a chunk is a pointer offset.
    for (int t0 = 0; t0 < nt; ) {
        const int nc = mach1_nt_chunk(nt - t0);
        const half * uc = u + (int64_t) t0*n;
        float * oc = scr_v + (int64_t) t0*m;
        switch (nc) {
            case 16: mach1_launch(mach1_rt_spine_mma_p16_kernel<NC, 16>, p, trellis, tlut, uc, oc, m, n); break;
            case 8:  mach1_launch(mach1_rt_spine_mma_p16_kernel<NC,  8>, p, trellis, tlut, uc, oc, m, n); break;
            case 4:  mach1_launch(mach1_rt_spine_mma_p16_kernel<NC,  4>, p, trellis, tlut, uc, oc, m, n); break;
            default: mach1_launch(mach1_rt_spine_mma_p16_kernel<NC,  2>, p, trellis, tlut, uc, oc, m, n); break;
        }
        t0 += nc;
    }
}

// Integer-lattice counterpart of the compressed fp16 spine. The transform
// stage has already written each token as q8 K16 groups plus its combined
// activation/lattice scale inside the ordinary scr_u allocation. Each warp
// decodes its two K16 blocks straight into signed-int8 B fragments and applies
// one m16n8k16 IMMA per NC8 output fragment. No dense or quantized weight bank
// is materialized and no additional global storage is allocated.
// Exact restructurings of the native int8 spine, selected by an OPT bitmask.
// Neither changes a value: with OPT == 0 every instantiation is the certified
// source.
//   bit 0 (WALK8) - the codec window is word-aligned, so the funnel shift and
//                   one of the two code loads per state are dead work.
//   bit 1 (QPAD)  - the staged q8 rows sit 128 B apart, so all eight A-fragment
//                   rows of a warp land in the same four shared banks; a 16 B
//                   row pad spreads them and keeps every copy 16 B aligned.
static __host__ __device__ constexpr int mach1_rt_spine_qpad(const int opt) {
    return (opt & 2) != 0 ? 16 : 0;
}

// WALK8 is an exact rewrite of the same two states, not a different walk.
// This fragment's state index is s = ri*8 + ti*2 + h, so s and h always have
// the same parity and the bit offset j0 = 8*s satisfies
//     j0 >> 4  == ri*4 + ti   (both h)
//     j0 & 15  == 8*h
// The window is therefore word-aligned at h == 0 and byte-shifted at h == 1:
// the funnel shift, the offset arithmetic and one of the two code loads per
// state are dead work. Values are bit-identical to the generic form.
template <int NT, bool WALK8 = false>
static __device__ __forceinline__ mach1_tc_frag_bi8 mach1_rt_spine_direct_b_i8(
        const uint16_t * __restrict__ s_cd, const uint16_t * __restrict__ s_ztab,
        const int kb, const int nb8, const int lane) {
    const int out_row = nb8*8 + (lane >> 2);
    const int mbi     = out_row >> 4;
    const int ri      = out_row & 15;
    const int ti      = lane & 3;
    const uint16_t * cd = s_cd + (kb*NT + mbi)*64;

    mach1_tc_frag_bi8 b = {};
    if constexpr (WALK8) {
        const int wa = ri*4 + ti;
        const unsigned aw = cd[wa];
        const unsigned bw = cd[wa + 1 == 64 ? 0 : wa + 1];
        const unsigned s1 = ((aw << 8) | (bw >> 8)) & 0xFFFFu;
        const unsigned p0 = aw*(aw + 1u);
        const unsigned p1 = s1*(s1 + 1u);
        b.r = (uint32_t) s_ztab[(p0 >> 6) & 1023u] |
              ((uint32_t) s_ztab[(p1 >> 6) & 1023u] << 16);
        return b;
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int s    = ri*8 + ti*2 + h;
        const int j0   = s*8;
        const int wa   = j0 >> 4;
        const int off  = j0 & 15;
        const int wb_i = wa + 1 == 64 ? 0 : wa + 1;
        const unsigned aw = cd[wa];
        const unsigned bw = cd[wb_i];
        const unsigned state = (((aw << 16) | bw) >> (16 - off)) & 0xFFFFu;
        const unsigned p = state*(state + 1u);
        // ztab's upper half resolves the codec's first-value sign flip.
        b.r |= (uint32_t) s_ztab[(p >> 6) & 1023u] << (16*h);
    }
    return b;
}

template <int NC, int NTOK = 16, int OPT = 0>
static __global__ __launch_bounds__(128) void mach1_rt_spine_imma8_p16_kernel(
        const uint16_t      * __restrict__ trellis,
        const uint16_t      * __restrict__ ztab,
        const unsigned char * __restrict__ qslots,
        float               * __restrict__ out,
        const int m, const int n) {
#ifdef AMPERE_MMA_AVAILABLE
    constexpr int W       = 4;
    constexpr int NBLK    = 8;
    constexpr int NT      = NC/16;
    constexpr int NF      = NC/8;
    constexpr int KW      = NBLK*16;
    constexpr int WG      = W*32;
    constexpr bool WALK8  = (OPT & 1) != 0;
    constexpr int QROW    = KW + mach1_rt_spine_qpad(OPT);
    static_assert(NC == 32, "p16 compressed int8 spine is NC32-only");
    static_assert(NTOK == 16 || NTOK == 8 || NTOK == 4 || NTOK == 2,
                  "token tile heights: 16 (certified) or the NTLOW 8/4/2");

    const int nc0  = blockIdx.x*NC;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int NB   = n/16;
    const int mb0  = nc0/16;

    extern __shared__ unsigned char smem_raw[];
    int8_t   * s_q    = (int8_t *) smem_raw;             // [16, QROW]
    float    * s_sc   = (float *) (s_q + 16*QROW);       // [16, NBLK]
    uint16_t * s_cd   = (uint16_t *) (s_sc + 16*NBLK);   // [NBLK, NT, 64]
    float    * s_red  = (float *) (s_cd + NBLK*NT*64);   // [W, 16, NC]
    uint16_t * s_ztab = (uint16_t *) (s_red + W*16*NC);  // [1024]

    // The 2 KiB integer table is the existing ZDP cache; staging it once per
    // CTA turns the data-dependent codec gathers into shared-memory traffic.
    for (int i = threadIdx.x; i < 1024/8; i += WG) {
        ((uint4 *) s_ztab)[i] = ((const uint4 *) ztab)[i];
    }
    if constexpr (NTOK < 16) {
        for (int i = threadIdx.x + NTOK*QROW; i < 16*QROW; i += WG) {
            s_q[i] = 0;
        }
        for (int i = threadIdx.x + NTOK*NBLK; i < 16*NBLK; i += WG) {
            s_sc[i] = 0.0f;
        }
    }

    mach1_tc_frag_c acc[NF];
#pragma unroll
    for (int i = 0; i < NF; ++i) {
        acc[i].f[0] = 0.0f;
        acc[i].f[1] = 0.0f;
        acc[i].f[2] = 0.0f;
        acc[i].f[3] = 0.0f;
    }

    const int nsteps = NB/NBLK;
    for (int step = 0; step < nsteps; ++step) {
        // qslots reserves the original n-float stride for each of 16 tokens:
        // n qbytes followed by n/16 float combined scales.
        for (int i = threadIdx.x; i < NTOK*(KW/16); i += WG) {
            const int row = i/(KW/16);
            const int ch  = i%(KW/16);
            const unsigned char * slot = qslots + (int64_t) row*4*n;
            ((uint4 *) (s_q + row*QROW))[ch] =
                ((const uint4 *) (slot + step*KW))[ch];
        }
        for (int i = threadIdx.x; i < NTOK*NBLK; i += WG) {
            const int row = i/NBLK;
            const int kb  = i%NBLK;
            const unsigned char * slot = qslots + (int64_t) row*4*n;
            const float * scales = (const float *) (slot + n);
            s_sc[i] = scales[step*NBLK + kb];
        }
        for (int i = threadIdx.x; i < NBLK*NT*8; i += WG) {
            const int tile_i = i/8;
            const int ch     = i%8;
            const int kb     = tile_i/NT;
            const int mbi    = tile_i%NT;
            const int64_t tile = (int64_t)(mb0 + mbi)*NB + step*NBLK + kb;
            ((uint4 *) (s_cd + (kb*NT + mbi)*64))[ch] =
                ((const uint4 *) (trellis + tile*64))[ch];
        }
        __syncthreads();

        const auto mma_block = [&](const int kb) {
            const int ar  = lane >> 2;
            const int ac4 = (lane & 3)*4;
            mach1_tc_frag_ai8 a;
            a.r[0] = *(const uint32_t *) (s_q + ar*QROW + kb*16 + ac4);
            a.r[1] = *(const uint32_t *) (s_q + (ar + 8)*QROW + kb*16 + ac4);
#pragma unroll
            for (int nb8 = 0; nb8 < NF; ++nb8) {
                const mach1_tc_frag_bi8 b =
                    mach1_rt_spine_direct_b_i8<NT, WALK8>(s_cd, s_ztab, kb, nb8, lane);
                mach1_tc_frag_ci8 ci = {};
                mach1_tc_mma_i8(ci, a, b);
                const float sc0 = s_sc[ar*NBLK + kb];
                const float sc8 = s_sc[(ar + 8)*NBLK + kb];
                acc[nb8].f[0] += (float) ci.r[0]*sc0;
                acc[nb8].f[1] += (float) ci.r[1]*sc0;
                acc[nb8].f[2] += (float) ci.r[2]*sc8;
                acc[nb8].f[3] += (float) ci.r[3]*sc8;
            }
        };

        const int kb0 = warp*2;
        mma_block(kb0);
        mma_block(kb0 + 1);
        __syncthreads();
    }

    float * red = s_red + warp*16*NC;
    const int rr = lane >> 2;
    const int cc = (lane & 3)*2;
#pragma unroll
    for (int nb8 = 0; nb8 < NF; ++nb8) {
        red[rr*NC + nb8*8 + cc + 0] = acc[nb8].f[0];
        red[rr*NC + nb8*8 + cc + 1] = acc[nb8].f[1];
        red[(rr + 8)*NC + nb8*8 + cc + 0] = acc[nb8].f[2];
        red[(rr + 8)*NC + nb8*8 + cc + 1] = acc[nb8].f[3];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < NTOK*NC; i += WG) {
        const int r = i/NC;
        const int c = i%NC;
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < W; ++w) {
            v += s_red[w*16*NC + i];
        }
        out[(int64_t) r*m + nc0 + c] = v;
    }
#else
    GGML_UNUSED(trellis); GGML_UNUSED(ztab); GGML_UNUSED(qslots); GGML_UNUSED(out);
    GGML_UNUSED(m); GGML_UNUSED(n);
    NO_DEVICE_CODE;
#endif
}

// Same-warp asynchronous pipeline for the native 4-bpw p16 trellis. Stage 0
// keeps the measured IMMA8 layout. Stage 1 aliases the first 4.5 KiB of the
// 8 KiB reduction tile, which is dead until every K step has completed. Thus
// q8 activations, scales, and trellis codes can be prefetched while the four
// original warps decode/MMA the other parity without adding shared or global
// storage, changing the grid, or introducing a producer warp cohort.
static __device__ __forceinline__ void mach1_cp_async_commit_group() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;" ::: "memory");
#else
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void mach1_cp_async_wait_group_0() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#else
    NO_DEVICE_CODE;
#endif
}

template <int NTOK = 16, int OPT = 0>
static __global__ __launch_bounds__(128, 6)
void mach1_rt_spine_imma8_cpasync_p16_kernel(
        const uint16_t      * __restrict__ trellis,
        const uint16_t      * __restrict__ ztab,
        const unsigned char * __restrict__ qslots,
        float               * __restrict__ out,
        const int m, const int n) {
#if defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
    constexpr int W           = 4;
    constexpr int NC          = 32;
    constexpr int NT          = NC/16;
    constexpr int NBLK        = 8;
    constexpr int NF          = NC/8;
    constexpr int KW          = NBLK*16;
    constexpr int WG          = W*32;
    constexpr bool WALK8      = (OPT & 1) != 0;
    constexpr int QROW        = KW + mach1_rt_spine_qpad(OPT);
    constexpr int Q_BYTES     = 16*QROW*sizeof(int8_t);
    constexpr int SC_BYTES    = 16*NBLK*sizeof(float);
    constexpr int CD_BYTES    = NBLK*NT*64*sizeof(uint16_t);
    constexpr int STAGE_BYTES = Q_BYTES + SC_BYTES + CD_BYTES;
    static_assert(STAGE_BYTES == 9*512 + 16*mach1_rt_spine_qpad(OPT),
                  "native cp.async p16 stage size changed");
    static_assert(NTOK == 16 || NTOK == 8 || NTOK == 4 || NTOK == 2,
                  "token tile heights: 16 (certified) or the NTLOW 8/4/2");

    const int nc0  = blockIdx.x*NC;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int NB   = n/16;
    const int mb0  = nc0/16;

    extern __shared__ unsigned char smem_raw[];
    // Original layout: [stage 0, reduction, ztab]. Stage 1 deliberately
    // begins at the same address as reduction and is dead before reduction.
    float    * s_red  = (float *) (smem_raw + STAGE_BYTES);  // [W, 16, NC]
    uint16_t * s_ztab = (uint16_t *) (s_red + W*16*NC);      // [1024]

    // Every copy is 16-byte aligned. In particular, each row's eight K16
    // scales occupy two uint4 records instead of 128 exposed scalar loads.
    const auto prefetch = [&](const int step, const int parity) {
        unsigned char * stage = smem_raw + parity*STAGE_BYTES;
        int8_t   * sq  = (int8_t *) stage;
        float    * ssc = (float *) (stage + Q_BYTES);
        uint16_t * scd = (uint16_t *) (stage + Q_BYTES + SC_BYTES);

        for (int i = threadIdx.x; i < NTOK*(KW/16); i += WG) {
            const int row = i/(KW/16);
            const int ch  = i%(KW/16);
            const unsigned char * slot = qslots + (int64_t) row*4*n;
            cp_async_cg_16<0>(
                ggml_cuda_cvta_generic_to_shared(sq + row*QROW + ch*16),
                slot + step*KW + ch*16);
        }
        for (int i = threadIdx.x; i < NTOK*(NBLK/4); i += WG) {
            const int row = i/(NBLK/4);
            const int ch  = i%(NBLK/4);
            const unsigned char * slot = qslots + (int64_t) row*4*n;
            const float * scales = (const float *) (slot + n);
            cp_async_cg_16<0>(
                ggml_cuda_cvta_generic_to_shared(ssc + row*NBLK + ch*4),
                scales + step*NBLK + ch*4);
        }
        for (int i = threadIdx.x; i < NBLK*NT*(64/8); i += WG) {
            const int tile_i = i/(64/8);
            const int ch     = i%(64/8);
            const int kb     = tile_i/NT;
            const int mbi    = tile_i%NT;
            const int64_t tile =
                (int64_t)(mb0 + mbi)*NB + step*NBLK + kb;
            cp_async_cg_16<0>(
                ggml_cuda_cvta_generic_to_shared(
                    scd + (kb*NT + mbi)*64 + ch*8),
                trellis + tile*64 + ch*8);
        }
        mach1_cp_async_commit_group();
    };

    // The immutable table remains in its measured 2 KiB CTA cache. Issue the
    // first asynchronous stage before its synchronous copy so those global
    // transactions can overlap.
    prefetch(0, 0);
    for (int i = threadIdx.x; i < 1024/8; i += WG) {
        ((uint4 *) s_ztab)[i] = ((const uint4 *) ztab)[i];
    }
    // Both stages' rows [NTOK, 16) are never prefetched; zero them once so the
    // dead half of every A fragment contributes exactly nothing. Stage 1
    // aliases s_red, which is dead until after the K loop.
    if constexpr (NTOK < 16) {
        for (int parity = 0; parity < 2; ++parity) {
            unsigned char * stage = smem_raw + parity*STAGE_BYTES;
            int8_t * sq  = (int8_t *) stage;
            float  * ssc = (float *) (stage + Q_BYTES);
            for (int i = threadIdx.x + NTOK*QROW; i < 16*QROW; i += WG) {
                sq[i] = 0;
            }
            for (int i = threadIdx.x + NTOK*NBLK; i < 16*NBLK; i += WG) {
                ssc[i] = 0.0f;
            }
        }
    }
    mach1_cp_async_wait_group_0();
    __syncthreads();

    mach1_tc_frag_c acc[NF];
#pragma unroll
    for (int i = 0; i < NF; ++i) {
        acc[i].f[0] = 0.0f;
        acc[i].f[1] = 0.0f;
        acc[i].f[2] = 0.0f;
        acc[i].f[3] = 0.0f;
    }

    const int nsteps = NB/NBLK;
    for (int step = 0; step < nsteps; ++step) {
        const int parity = step & 1;
        unsigned char * stage = smem_raw + parity*STAGE_BYTES;
        const int8_t   * sq  = (const int8_t *) stage;
        const float    * ssc = (const float *) (stage + Q_BYTES);
        const uint16_t * scd =
            (const uint16_t *) (stage + Q_BYTES + SC_BYTES);

        const bool has_next = step + 1 < nsteps;
        if (has_next) {
            prefetch(step + 1, parity ^ 1);
        }

        const auto mma_block = [&](const int kb) {
            const int ar  = lane >> 2;
            const int ac4 = (lane & 3)*4;
            mach1_tc_frag_ai8 a;
            a.r[0] = *(const uint32_t *) (sq + ar*QROW + kb*16 + ac4);
            a.r[1] = *(const uint32_t *) (sq + (ar + 8)*QROW + kb*16 + ac4);
            const float sc0 = ssc[ar*NBLK + kb];
            const float sc8 = ssc[(ar + 8)*NBLK + kb];
#pragma unroll
            for (int nb8 = 0; nb8 < NF; ++nb8) {
                const mach1_tc_frag_bi8 b =
                    mach1_rt_spine_direct_b_i8<NT, WALK8>(
                        scd, s_ztab, kb, nb8, lane);
                mach1_tc_frag_ci8 ci = {};
                mach1_tc_mma_i8(ci, a, b);
                acc[nb8].f[0] += (float) ci.r[0]*sc0;
                acc[nb8].f[1] += (float) ci.r[1]*sc0;
                acc[nb8].f[2] += (float) ci.r[2]*sc8;
                acc[nb8].f[3] += (float) ci.r[3]*sc8;
            }
        };

        const int kb0 = warp*2;
        mma_block(kb0);
        mma_block(kb0 + 1);

        if (has_next) {
            mach1_cp_async_wait_group_0();
        }
        // Publishes the next parity and, after the final step, prevents any
        // lane from overwriting the aliased reduction tile while it is read.
        __syncthreads();
    }

    float * red = s_red + warp*16*NC;
    const int rr = lane >> 2;
    const int cc = (lane & 3)*2;
#pragma unroll
    for (int nb8 = 0; nb8 < NF; ++nb8) {
        red[rr*NC + nb8*8 + cc + 0] = acc[nb8].f[0];
        red[rr*NC + nb8*8 + cc + 1] = acc[nb8].f[1];
        red[(rr + 8)*NC + nb8*8 + cc + 0] = acc[nb8].f[2];
        red[(rr + 8)*NC + nb8*8 + cc + 1] = acc[nb8].f[3];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < NTOK*NC; i += WG) {
        const int r = i/NC;
        const int c = i%NC;
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < W; ++w) {
            v += s_red[w*16*NC + i];
        }
        out[(int64_t) r*m + nc0 + c] = v;
    }
#else
    GGML_UNUSED(trellis); GGML_UNUSED(ztab); GGML_UNUSED(qslots); GGML_UNUSED(out);
    GGML_UNUSED(m); GGML_UNUSED(n);
    NO_DEVICE_CODE;
#endif
}

// Work-centric exact-shape companion for the underfilled B16 out projection.
// The ordinary KS1 kernel above remains source-identical. This grid gives each
// NC32 output tile four independent contiguous eight-step K ranges, raising
// 64 CTAs to 256 on L40S. All four deterministic FP32 partials occupy the
// candidate's token-major [nt, 4, m] transient scratch and are folded by the
// existing out launch.
template <int N = 4096, int NTOK = 16, int OPT = 0>
static __global__ __launch_bounds__(128, 6)
void mach1_rt_spine_imma8_cpasync_split4_p16_kernel(
        const uint16_t      * __restrict__ trellis,
        const uint16_t      * __restrict__ ztab,
        const unsigned char * __restrict__ qslots,
        float               * __restrict__ partials) {
#if defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
    constexpr int M           = 2048;
    constexpr int N_TOK       = 16;   // staged tile height; NTOK rows are live
    constexpr int SPLITS      = 4;
    constexpr int W           = 4;
    constexpr int NC          = 32;
    constexpr int NT          = NC/16;
    constexpr int NBLK        = 8;
    constexpr int NF          = NC/8;
    constexpr int KW          = NBLK*16;
    constexpr int WG          = W*32;
    constexpr int NB          = N/16;
    constexpr int NSTEPS      = (NB/NBLK)/SPLITS;
    constexpr bool WALK8      = (OPT & 1) != 0;
    constexpr int QROW        = KW + mach1_rt_spine_qpad(OPT);
    constexpr int Q_BYTES     = N_TOK*QROW*sizeof(int8_t);
    constexpr int SC_BYTES    = N_TOK*NBLK*sizeof(float);
    constexpr int CD_BYTES    = NBLK*NT*64*sizeof(uint16_t);
    constexpr int STAGE_BYTES = Q_BYTES + SC_BYTES + CD_BYTES;
    static_assert(N == 4096 ? NSTEPS == 8 : (N == 512 && NSTEPS == 1),
                  "native p16 split-K step count changed");
    static_assert(STAGE_BYTES == 9*512 + 16*mach1_rt_spine_qpad(OPT),
                  "native p16 split-K stage size changed");
    static_assert(NTOK == 16 || NTOK == 8 || NTOK == 4 || NTOK == 2,
                  "token tile heights: 16 (certified) or the NTLOW 8/4/2");

    const int split = (int) blockIdx.x % SPLITS;
    const int tile  = (int) blockIdx.x/SPLITS;
    const int nc0   = tile*NC;
    const int mb0   = nc0/16;
    const int warp  = threadIdx.x >> 5;
    const int lane  = threadIdx.x & 31;
    const int step0 = split*NSTEPS;

    extern __shared__ unsigned char smem_raw[];
    float    * s_red  = (float *) (smem_raw + STAGE_BYTES);
    uint16_t * s_ztab = (uint16_t *) (s_red + W*N_TOK*NC);

    const auto prefetch = [&](const int step, const int parity) {
        unsigned char * stage = smem_raw + parity*STAGE_BYTES;
        int8_t   * sq  = (int8_t *) stage;
        float    * ssc = (float *) (stage + Q_BYTES);
        uint16_t * scd = (uint16_t *) (stage + Q_BYTES + SC_BYTES);

        for (int i = threadIdx.x; i < NTOK*(KW/16); i += WG) {
            const int row = i/(KW/16);
            const int ch  = i%(KW/16);
            const unsigned char * slot = qslots + (int64_t) row*4*N;
            cp_async_cg_16<0>(
                ggml_cuda_cvta_generic_to_shared(sq + row*QROW + ch*16),
                slot + step*KW + ch*16);
        }
        for (int i = threadIdx.x; i < NTOK*(NBLK/4); i += WG) {
            const int row = i/(NBLK/4);
            const int ch  = i%(NBLK/4);
            const unsigned char * slot = qslots + (int64_t) row*4*N;
            const float * scales = (const float *) (slot + N);
            cp_async_cg_16<0>(
                ggml_cuda_cvta_generic_to_shared(ssc + row*NBLK + ch*4),
                scales + step*NBLK + ch*4);
        }
        for (int i = threadIdx.x; i < NBLK*NT*(64/8); i += WG) {
            const int tile_i = i/(64/8);
            const int ch     = i%(64/8);
            const int kb     = tile_i/NT;
            const int mbi    = tile_i%NT;
            const int64_t ti = (int64_t)(mb0 + mbi)*NB + step*NBLK + kb;
            cp_async_cg_16<0>(
                ggml_cuda_cvta_generic_to_shared(
                    scd + (kb*NT + mbi)*64 + ch*8),
                trellis + ti*64 + ch*8);
        }
        mach1_cp_async_commit_group();
    };

    prefetch(step0, 0);
    for (int i = threadIdx.x; i < 1024/8; i += WG) {
        ((uint4 *) s_ztab)[i] = ((const uint4 *) ztab)[i];
    }
    if constexpr (NTOK < 16) {
        for (int parity = 0; parity < 2; ++parity) {
            unsigned char * stage = smem_raw + parity*STAGE_BYTES;
            int8_t * sq  = (int8_t *) stage;
            float  * ssc = (float *) (stage + Q_BYTES);
            for (int i = threadIdx.x + NTOK*QROW; i < N_TOK*QROW; i += WG) {
                sq[i] = 0;
            }
            for (int i = threadIdx.x + NTOK*NBLK; i < N_TOK*NBLK; i += WG) {
                ssc[i] = 0.0f;
            }
        }
    }
    mach1_cp_async_wait_group_0();
    __syncthreads();

    mach1_tc_frag_c acc[NF];
#pragma unroll
    for (int i = 0; i < NF; ++i) {
        acc[i].f[0] = 0.0f;
        acc[i].f[1] = 0.0f;
        acc[i].f[2] = 0.0f;
        acc[i].f[3] = 0.0f;
    }

#pragma unroll 1
    for (int local_step = 0; local_step < NSTEPS; ++local_step) {
        const int step   = step0 + local_step;
        const int parity = local_step & 1;
        unsigned char * stage = smem_raw + parity*STAGE_BYTES;
        const int8_t   * sq   = (const int8_t *) stage;
        const float    * ssc  = (const float *) (stage + Q_BYTES);
        const uint16_t * scd  =
            (const uint16_t *) (stage + Q_BYTES + SC_BYTES);

        const bool has_next = local_step + 1 < NSTEPS;
        if (has_next) {
            prefetch(step + 1, parity ^ 1);
        }

        const auto mma_block = [&](const int kb) {
            const int ar  = lane >> 2;
            const int ac4 = (lane & 3)*4;
            mach1_tc_frag_ai8 a;
            a.r[0] = *(const uint32_t *) (sq + ar*QROW + kb*16 + ac4);
            a.r[1] = *(const uint32_t *) (sq + (ar + 8)*QROW + kb*16 + ac4);
            const float sc0 = ssc[ar*NBLK + kb];
            const float sc8 = ssc[(ar + 8)*NBLK + kb];
#pragma unroll
            for (int nb8 = 0; nb8 < NF; ++nb8) {
                const mach1_tc_frag_bi8 b =
                    mach1_rt_spine_direct_b_i8<NT, WALK8>(
                        scd, s_ztab, kb, nb8, lane);
                mach1_tc_frag_ci8 ci = {};
                mach1_tc_mma_i8(ci, a, b);
                acc[nb8].f[0] += (float) ci.r[0]*sc0;
                acc[nb8].f[1] += (float) ci.r[1]*sc0;
                acc[nb8].f[2] += (float) ci.r[2]*sc8;
                acc[nb8].f[3] += (float) ci.r[3]*sc8;
            }
        };

        const int kb0 = warp*2;
        mma_block(kb0);
        mma_block(kb0 + 1);
        if (has_next) {
            mach1_cp_async_wait_group_0();
        }
        __syncthreads();
    }

    float * red = s_red + warp*N_TOK*NC;
    const int rr = lane >> 2;
    const int cc = (lane & 3)*2;
#pragma unroll
    for (int nb8 = 0; nb8 < NF; ++nb8) {
        red[rr*NC + nb8*8 + cc + 0] = acc[nb8].f[0];
        red[rr*NC + nb8*8 + cc + 1] = acc[nb8].f[1];
        red[(rr + 8)*NC + nb8*8 + cc + 0] = acc[nb8].f[2];
        red[(rr + 8)*NC + nb8*8 + cc + 1] = acc[nb8].f[3];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < NTOK*NC; i += WG) {
        const int t = i/NC;
        const int c = i%NC;
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < W; ++w) {
            v += s_red[w*N_TOK*NC + i];
        }
        const int64_t row = (int64_t) t*SPLITS + split;
        partials[row*M + nc0 + c] = v;
    }
#else
    GGML_UNUSED(trellis); GGML_UNUSED(ztab); GGML_UNUSED(qslots);
    GGML_UNUSED(partials);
    NO_DEVICE_CODE;
#endif
}

static bool mach1_rt_imma8_cpasync_enabled() {
    static const bool on = mach1_env_int("GGML_MACH1_RT_IMMA8_CPASYNC", 0) != 0;
    return on;
}

// GGML_MACH1_RT_SPINE_OPT selects the exact restructurings above; 0 is the
// certified source and the default.
static int mach1_rt_spine_opt() {
    static const int v = mach1_env_int("GGML_MACH1_RT_SPINE_OPT", 0) & 3;
    return v;
}

template <int OPT>
static void mach1_rt_spine_imma8_p16_opt(
        const uint16_t * trellis, const uint16_t * ztab,
        const unsigned char * qslots, float * scr_v,
        const int m, const int n, cudaStream_t stream,
        const bool split4, const int nt, const bool cpasync_on) {
    constexpr int NC    = 32;
    constexpr int W     = 4;
    constexpr int NBLK  = 8;
    constexpr int NT    = NC/16;
    constexpr int KW    = NBLK*16 + mach1_rt_spine_qpad(OPT);
    constexpr size_t smem =
        16*KW*sizeof(int8_t) +
        16*NBLK*sizeof(float) +
        NBLK*NT*64*sizeof(uint16_t) +
        W*16*NC*sizeof(float) +
        1024*sizeof(uint16_t);
    static_assert(smem == 29*512 + 16*mach1_rt_spine_qpad(OPT),
                  "p16 compressed int8 spine shared-memory budget changed");

    // nt == 16 is one chunk at zero offset - the certified launch. Other
    // admitted widths issue the same kernel per {16,8,4,2}-token chunk: q8
    // slot records are token-major with a 4n-byte stride, scr_v rows with an
    // m stride, split-K partials with a 4m stride, so a chunk is an offset.
    const ggml_cuda_kernel_launch_params p =
        ggml_cuda_kernel_launch_params(dim3(m/NC, 1, 1), dim3(W*32, 1, 1), smem, stream);
    const ggml_cuda_kernel_launch_params p4 =
        ggml_cuda_kernel_launch_params(dim3((m/NC)*4, 1, 1), dim3(W*32, 1, 1), smem, stream);
    for (int t0 = 0; t0 < nt; ) {
        const int nc = mach1_nt_chunk(nt - t0);
        const unsigned char * qc = qslots + (int64_t) t0*4*n;
        float * oc = scr_v + (int64_t)(split4 ? 4 : 1)*t0*m;
        if (split4) {
            GGML_ASSERT(cpasync_on && m == 2048 && (n == 4096 || n == 512));
            if (n == 4096) {
                switch (nc) {
                    case 16: mach1_launch(mach1_rt_spine_imma8_cpasync_split4_p16_kernel<4096, 16, OPT>, p4, trellis, ztab, qc, oc); break;
                    case 8:  mach1_launch(mach1_rt_spine_imma8_cpasync_split4_p16_kernel<4096,  8, OPT>, p4, trellis, ztab, qc, oc); break;
                    case 4:  mach1_launch(mach1_rt_spine_imma8_cpasync_split4_p16_kernel<4096,  4, OPT>, p4, trellis, ztab, qc, oc); break;
                    default: mach1_launch(mach1_rt_spine_imma8_cpasync_split4_p16_kernel<4096,  2, OPT>, p4, trellis, ztab, qc, oc); break;
                }
            } else {
                switch (nc) {
                    case 16: mach1_launch(mach1_rt_spine_imma8_cpasync_split4_p16_kernel<512, 16, OPT>, p4, trellis, ztab, qc, oc); break;
                    case 8:  mach1_launch(mach1_rt_spine_imma8_cpasync_split4_p16_kernel<512,  8, OPT>, p4, trellis, ztab, qc, oc); break;
                    case 4:  mach1_launch(mach1_rt_spine_imma8_cpasync_split4_p16_kernel<512,  4, OPT>, p4, trellis, ztab, qc, oc); break;
                    default: mach1_launch(mach1_rt_spine_imma8_cpasync_split4_p16_kernel<512,  2, OPT>, p4, trellis, ztab, qc, oc); break;
                }
            }
        } else if (cpasync_on) {
            switch (nc) {
                case 16: mach1_launch(mach1_rt_spine_imma8_cpasync_p16_kernel<16, OPT>, p, trellis, ztab, qc, oc, m, n); break;
                case 8:  mach1_launch(mach1_rt_spine_imma8_cpasync_p16_kernel< 8, OPT>, p, trellis, ztab, qc, oc, m, n); break;
                case 4:  mach1_launch(mach1_rt_spine_imma8_cpasync_p16_kernel< 4, OPT>, p, trellis, ztab, qc, oc, m, n); break;
                default: mach1_launch(mach1_rt_spine_imma8_cpasync_p16_kernel< 2, OPT>, p, trellis, ztab, qc, oc, m, n); break;
            }
        } else {
            switch (nc) {
                case 16: mach1_launch(mach1_rt_spine_imma8_p16_kernel<NC, 16, OPT>, p, trellis, ztab, qc, oc, m, n); break;
                case 8:  mach1_launch(mach1_rt_spine_imma8_p16_kernel<NC,  8, OPT>, p, trellis, ztab, qc, oc, m, n); break;
                case 4:  mach1_launch(mach1_rt_spine_imma8_p16_kernel<NC,  4, OPT>, p, trellis, ztab, qc, oc, m, n); break;
                default: mach1_launch(mach1_rt_spine_imma8_p16_kernel<NC,  2, OPT>, p, trellis, ztab, qc, oc, m, n); break;
            }
        }
        t0 += nc;
    }
}

static void mach1_rt_spine_imma8_p16(
        const uint16_t * trellis, const uint16_t * ztab,
        const float * qslots, float * scr_v,
        const int m, const int n, cudaStream_t stream,
        const bool split4 = false, const int nt = 16) {
    const bool cpasync_on = mach1_rt_imma8_cpasync_enabled();
    if (cpasync_on && mach1_debug_on()) {
        static std::once_flag once;
        std::call_once(once, []() {
            fprintf(stderr,
                    "mach1: zero-VRAM native rt_imma8 cp.async double-buffer "
                    "ENGAGED (s_red alias)\n");
        });
    }
    const unsigned char * qb = (const unsigned char *) qslots;
    switch (mach1_rt_spine_opt()) {
        case 1:  mach1_rt_spine_imma8_p16_opt<1>(trellis, ztab, qb, scr_v, m, n, stream, split4, nt, cpasync_on); break;
        case 2:  mach1_rt_spine_imma8_p16_opt<2>(trellis, ztab, qb, scr_v, m, n, stream, split4, nt, cpasync_on); break;
        case 3:  mach1_rt_spine_imma8_p16_opt<3>(trellis, ztab, qb, scr_v, m, n, stream, split4, nt, cpasync_on); break;
        default: mach1_rt_spine_imma8_p16_opt<0>(trellis, ztab, qb, scr_v, m, n, stream, split4, nt, cpasync_on); break;
    }
    if (split4 && mach1_debug_on()) {
        static std::once_flag once;
        std::call_once(once, []() {
            fprintf(stderr,
                    "mach1: native rt_imma8 split-K4 m2048 n4096 nt16 "
                    "ENGAGED (384 KiB transient, 0 persistent)\n");
        });
    }
}

// one 16-weight row: 8 states, 4 dp4a; ph carries the sign in bit 15.
// Entries are widened to u32 and replicated REP ways so the bank a lane
// touches is mostly a function of the lane (the r220 probes put the
// data-dependent serialization at 13% of the batch walk). REP is 2 in the
// batch kernel - 4 plus the TC fold workspace exceeds Ada's 99 KB block
// budget - and 4 where the shared budget is free.
template <int REP>
static __device__ __forceinline__ int mach1_rt_zdp_row_rep(
        const uint32_t * __restrict__ ph,
        const uint32_t *              zslutr,   // [1024*REP]
        const uint32_t * __restrict__ qq, const int sel) {
    int iacc = 0;
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
        const uint32_t z01 = zslutr[((ph[j]     >> 6) & 1023u)*REP + sel];
        const uint32_t z23 = zslutr[((ph[j + 1] >> 6) & 1023u)*REP + sel];
        iacc = ggml_cuda_dp4a((int) __byte_perm(z01, z23, 0x5410), (int) qq[j >> 1], iacc);
    }
    return iacc;
}

// --- the exp integer-lattice walk (GGML_MACH1_ZDP, GGML_MACH1_EXP_DP4A) ---
//
// The expert alphabet is s0*z and mach1_exp_p4_table stores the nibble z + 8,
// so a state's eight weights are the eight nibbles of one u32 and its dot is
// two dp4a against the column tile's quantized activations. HI2 carries the
// activations at 15 bits as two int8 planes (q = 128*qh + ql) for 1/256 the
// quantization noise at four dp4a - round 46's arm P7. The weights are exact
// either way; only the activations are approximated.
//
// The nibble split yields (n0,n2,n4,n6) and (n1,n3,n5,n7), so the planes are
// packed in that interleave. qc0 keeps lane 0 at full precision for the sign
// fix, and the z + 8 bias comes off once per row as 8*sumq.
template <bool HI2>
static __device__ __forceinline__ float mach1_exp_zdp_planes(
        const float * __restrict__ uu, uint32_t * qe, uint32_t * qo,
        uint32_t * qle, uint32_t * qlo, int * qc0, int & sumq) {
    constexpr int QMAX = HI2 ? 16319 : 127;   // HI2: qh stays int8, ql in [-64,63]
    float du = 0.0f;
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        du = fmaxf(du, fabsf(uu[c]));
    }
    const float qsf = du > 0.0f ? (float) QMAX/du : 0.0f;
    int q[16];
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        q[c] = __float2int_rn(uu[c]*qsf);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        if constexpr (HI2) {
            int qh[8], ql[8];
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                qh[c] = (q[8*h + c] + 64) >> 7;
                ql[c] = q[8*h + c] - 128*qh[c];
            }
            qe[h]  = (uint32_t)(qh[0] & 0xFF) | ((uint32_t)(qh[2] & 0xFF) << 8)
                   | ((uint32_t)(qh[4] & 0xFF) << 16) | ((uint32_t)(qh[6] & 0xFF) << 24);
            qo[h]  = (uint32_t)(qh[1] & 0xFF) | ((uint32_t)(qh[3] & 0xFF) << 8)
                   | ((uint32_t)(qh[5] & 0xFF) << 16) | ((uint32_t)(qh[7] & 0xFF) << 24);
            qle[h] = (uint32_t)(ql[0] & 0xFF) | ((uint32_t)(ql[2] & 0xFF) << 8)
                   | ((uint32_t)(ql[4] & 0xFF) << 16) | ((uint32_t)(ql[6] & 0xFF) << 24);
            qlo[h] = (uint32_t)(ql[1] & 0xFF) | ((uint32_t)(ql[3] & 0xFF) << 8)
                   | ((uint32_t)(ql[5] & 0xFF) << 16) | ((uint32_t)(ql[7] & 0xFF) << 24);
        } else {
            qe[h] = (uint32_t)(q[8*h + 0] & 0xFF) | ((uint32_t)(q[8*h + 2] & 0xFF) << 8)
                  | ((uint32_t)(q[8*h + 4] & 0xFF) << 16) | ((uint32_t)(q[8*h + 6] & 0xFF) << 24);
            qo[h] = (uint32_t)(q[8*h + 1] & 0xFF) | ((uint32_t)(q[8*h + 3] & 0xFF) << 8)
                  | ((uint32_t)(q[8*h + 5] & 0xFF) << 16) | ((uint32_t)(q[8*h + 7] & 0xFF) << 24);
        }
        qc0[h] = q[8*h];
    }
    sumq = 0;
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        sumq += q[c];
    }
    return du;
}

// one state: zk is the row's eight z + 8 nibbles, ph carries the sign in bit
// 15, h selects the column half the state's ci lands in
template <bool HI2>
static __device__ __forceinline__ int mach1_exp_zdp_state(
        const uint32_t zk, const uint32_t ph,
        const uint32_t * qe, const uint32_t * qo,
        const uint32_t * qle, const uint32_t * qlo, const int * qc0, const int h) {
    int d = ggml_cuda_dp4a((int)(zk & 0x0F0F0F0Fu), (int) qe[h],
            ggml_cuda_dp4a((int)((zk >> 4) & 0x0F0F0F0Fu), (int) qo[h], 0));
    if constexpr (HI2) {
        const int dl = ggml_cuda_dp4a((int)(zk & 0x0F0F0F0Fu), (int) qle[h],
                       ggml_cuda_dp4a((int)((zk >> 4) & 0x0F0F0F0Fu), (int) qlo[h], 0));
        d = 128*d + dl;
    }
    if (ph & 0x8000u) {
        d -= 2*((int)(zk & 0xFu) - 8)*qc0[h];   // the codec flips weight 0 only
    }
    return d;
}

// FWHT over m floats in GLOBAL memory with a 2048-float shared workspace:
// stages with stride >= 2048 run as block-wide global passes, the rest as
// independent shared-memory chunks. Same sums, different association -
// contractible, the same class as the dense-path split. Reads go through
// __ldcg: the values were written by other blocks behind a counter fence.
template <int WG>
static __device__ void mach1_fwht_chunked(
        float * __restrict__ v, float * sh, const int m, const int tid) {
    for (int len = m >> 1; len >= 2048; len >>= 1) {
        for (int i = tid; i < m/2; i += WG) {
            const int a = (i/len)*(2*len) + (i%len);
            const float x = __ldcg(&v[a]);
            const float y = __ldcg(&v[a + len]);
            v[a]       = x + y;
            v[a + len] = x - y;
        }
        __syncthreads();
    }
    const int chunk = m < 2048 ? m : 2048;
    for (int c0 = 0; c0 < m; c0 += chunk) {
        for (int i = tid; i < chunk; i += WG) {
            sh[i] = __ldcg(&v[c0 + i]);
        }
        __syncthreads();
        mach1_fwht_block(sh, chunk, tid, WG);
        for (int i = tid; i < chunk; i += WG) {
            v[c0 + i] = sh[i];
        }
        __syncthreads();
    }
}

// WG covers one thread per column tile (tid < tiles_y walk); WG=128 avoids
// launching idle warps when tiles_y <= 128 (n <= 2048). The dropped warps
// only ever contributed exact zeros to the cross-warp reduction.
template <int WG>
__global__ void mach1_rt_walk_kernel(
        const uint16_t * __restrict__ trellis,
        const half     * __restrict__ tlut,     // pre-rounded fp16
        const float    * __restrict__ scr_u,
        float          * __restrict__ scr_v,
        const int m, const int n) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = WG / warp_size;
    __shared__ float red[16][8];   // >= n_warps for warp_size 32 (and 64 on HIP)
    __shared__ float slut[1024];   // full F16 [2,512] tlut, staged as fp32

    const int t   = blockIdx.z;
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;

    constexpr int words = 64;      // K = 4: 64 int16 words per 16x16 tile
    const int tiles_y = n / 16;
    const bool have   = tid < tiles_y;

    // 2 KB table: gather from shared instead of global (same fp32 values —
    // the __half2float just moves ahead of the gather)
    for (int i = tid; i < 1024; i += WG) {
        slut[i] = __half2float(tlut[i]);
    }
    __syncthreads();

    float partial[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        partial[i] = 0.0f;
    }

    if (have) {
        const int64_t tw = ((int64_t) tr*tiles_y + tid)*words;
        const float * ub = scr_u + (int64_t) t*n + tid*16;

        // the tile column's 16 u values are shared by all 16 tile rows
        float uu[16];
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            uu[c] = ub[c];
        }

        for (int ri = 0; ri < 16; ++ri) {
            // row ri covers stream bytes [8*ri, 8*ri+8] -> words [4*ri, 4*ri+4]
            uint32_t w[5];
#pragma unroll
            for (int q = 0; q < 5; ++q) {
                w[q] = trellis[tw + ((4*ri + q) & 63)];   // only q=4,ri=15 wraps
            }
            // 8 independent byte-aligned direct-window states
            uint32_t ph[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t hi = w[j >> 1];
                const uint32_t st = (j & 1) == 0 ? hi
                    : (((hi << 8) | (w[(j >> 1) + 1] >> 8)) & 0xFFFFu);
                ph[j] = st*(st + 1u);
            }
            // same accumulation order as the serial walk: per state, w0 then w1
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t row = (ph[j] >> 6) & 511u;
                float a0 = slut[2*row + 0];
                const float a1 = slut[2*row + 1];
                if (ph[j] & 0x8000u) {
                    a0 = -a0;                             // exact in fp16 (sign bit)
                }
                acc += a0*uu[2*j] + a1*uu[2*j + 1];
            }
            partial[ri] = acc;
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

// Row-split walk: same math, 4x the threads, 4x fewer accumulators.
//
// The original walk gives one thread all 16 tile-rows of a column tile: for
// [m 8192, n 2048] that is 512 blocks x 128 threads = 65k threads on a part
// that wants ~270k, with 16 live accumulators plus the per-row word loads,
// so occupancy - not bandwidth - sets the pace (measured ~200 GB/s on an
// 8 MB read). Here a block is RG x tiles_y threads: thread (rg, ct) computes
// tile-rows [4rg, 4rg+4) of column tile ct.
//
// BIT-EXACT vs mach1_rt_walk_kernel by construction: each (tile-row, column
// tile) partial is the same 8 states accumulated in the same order, and the
// cross-tile reduction keeps the same shape - each 128-thread row group is
// four warps reduced by the same shuffle butterfly, then the same sequential
// warp combine. Only the thread that owns a partial changes.
// PROBE (GGML_MACH1_WALK_PROBE, measurement only - output is garbage when
// non-zero): both GPU profilers are unavailable on this fleet (nsys dies in
// clock-sync conversion, ncu cannot load counters), so the walk's internal
// cost split is measured by ablating pieces of the kernel itself.
//   1 = no LUT gather (state math + MAC remain)
//   2 = no state arithmetic (gather + MAC remain, index is uniform)
//   3 = neither (pure activation MAC + trellis staging)
template <int WG, int RG, int PROBE>
__global__ void mach1_rt_walk_rows_kernel(
        const uint16_t * __restrict__ trellis,
        const half     * __restrict__ tlut,     // pre-rounded fp16
        const float    * __restrict__ scr_u,
        float          * __restrict__ scr_v,
        const int m, const int n) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int CT        = WG / RG;            // column tiles per block
    constexpr int n_warps   = CT / warp_size;     // warps per row group
    __shared__ float red[RG][16/RG][CT/32];
    // De-interleaved tlut: the packed [row][2] layout puts every a0 on an EVEN
    // bank, so half the banks sit idle and every gather pays a 2x conflict
    // before random row collisions even start. Two arrays make the bank index
    // row % 32. Same values, same read order, bit-exact.
    __shared__ float slutA[512];
    __shared__ float slutB[512];
    // Coalesced trellis staging. Read straight from global, a thread takes 10
    // of the 128 bytes of its OWN tile, so a 32-lane warp touches 32 distinct
    // 128 B lines - the hardware moves ~4 KB to deliver 320 B. That ~12x read
    // amplification is why the walk measures ~168 GB/s effective on an 8.4 MB
    // stream, and why neither more threads (row-split: paired neutral) nor
    // fewer bank conflicts (de-interleaved tlut: paired neutral) moved it.
    // The block's whole tile row is 16 KB; staging it cooperatively makes
    // every global read fully coalesced and every later access shared.
    __shared__ uint16_t stre[CT*64];

    const int t   = blockIdx.z;
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;
    const int ct  = tid % CT;                     // column tile (lane layout as before)
    const int rg  = tid / CT;                     // row group

    for (int i = tid; i < 512; i += WG) {
        slutA[i] = __half2float(tlut[2*i + 0]);
        slutB[i] = __half2float(tlut[2*i + 1]);
    }
    {
        const int      tiles   = n / 16;
        const int64_t  tbase_w = (int64_t) tr*tiles*64;
        for (int i = tid; i < tiles*64; i += WG) {
            stre[i] = trellis[tbase_w + i];
        }
    }
    __syncthreads();

    constexpr int words   = 64;                   // K = 4
    constexpr int ROWS    = 16/RG;
    const int     tiles_y = n / 16;
    const bool    have    = ct < tiles_y;

    float partial[ROWS];
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
        partial[i] = 0.0f;
    }

    if (have) {
        const int     tw = ct*words;   // into the staged tile row
        const float * ub = scr_u + (int64_t) t*n + ct*16;

        float uu[16];
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            uu[c] = ub[c];
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int ri = rg*ROWS + r;
            uint32_t w[5];
#pragma unroll
            for (int q = 0; q < 5; ++q) {
                w[q] = stre[tw + ((4*ri + q) & 63)];
            }
            uint32_t ph[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t hi = w[j >> 1];
                const uint32_t st = (j & 1) == 0 ? hi
                    : (((hi << 8) | (w[(j >> 1) + 1] >> 8)) & 0xFFFFu);
                // PROBE&2 removes the state arithmetic (the multiply that
                // makes the LUT index data-dependent), keeping the gather.
                ph[j] = (PROBE & 2) ? (uint32_t)(j*64) : st*(st + 1u);
            }
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t row = (ph[j] >> 6) & 511u;
                // PROBE&1 removes the LUT gather, keeping state math and MAC.
                float a0 = (PROBE & 1) ? 1.0f : slutA[row];
                const float a1 = (PROBE & 1) ? 1.0f : slutB[row];
                if (ph[j] & 0x8000u) {
                    a0 = -a0;
                }
                acc += a0*uu[2*j] + a1*uu[2*j + 1];
            }
            partial[r] = acc;
        }
    }

    const int lane = ct % warp_size;
    const int wid  = ct / warp_size;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const float s = warp_reduce_sum<warp_size>(partial[r]);
        if (lane == 0) {
            red[rg][r][wid] = s;
        }
    }
    __syncthreads();
    if (tid < 16) {
        const int g = tid / ROWS;
        const int r = tid % ROWS;
        float sum = 0.0f;
#pragma unroll
        for (int wj = 0; wj < n_warps; ++wj) {
            sum += red[g][r][wj];
        }
        scr_v[(int64_t) t*m + tr*16 + g*ROWS + r] = sum;
    }
}

// Vector-load walk (GGML_MACH1_WALK_VEC=1). Same math, two fewer memory
// pathologies. MEASURED in isolation (mb3.cu, the rt decode shape census on
// an H200, cold-L2 rotation, us/op):
//
//   shape        orig   rows(default)   vec     fp16 GEMV
//   8192x2048   29.00      34.28       19.27      10.96
//   2048x4096   15.76      17.86       10.23       5.87
//   4096x2048   15.84      17.90       10.33       5.36
//   512x2048     9.97      10.12        6.40       3.65
//   2048x512     8.43       5.27        4.36       2.83
//   TOKEN ms     3.600      3.845       2.304      1.312
//
// Two changes, both value-identical:
//
// 1. VECTOR TRELLIS LOAD. A row consumes 5 uint16 - 10 bytes inside its
//    128-byte tile - so the scalar form issues 20 LDG.U16 per thread, and
//    because consecutive threads are 128 bytes apart every one of those
//    instructions touches 32 separate sectors to deliver 64 useful bytes.
//    A thread's four rows need words [16rg, 16rg+16], which is covered by
//    THREE uint4 at base[(2rg+q) & 7] - the & 7 reproducing exactly the & 63
//    word wrap the scalar form does for the last row group. 20 loads -> 3.
//
// 2. u IN SHARED. ub[0..15] is read once per (row group, column tile), so at
//    RG 4 the activation is pulled four times per block out of L1. Staging
//    the whole u vector into shared once per block removes that entirely.
//    (This is why the earlier RG 8 arm regressed: it doubled these reads.)
//
// No trellis staging: with the vector load the global reads are already
// well-formed, and the staged variant measured slower (26.36 vs 19.27).
//
// BIT-EXACT: same words, same states, same order, same reduction tree.
// UFUSE folds the u stage in: the block computes u = H(su .* x)/sqrt(n)
// itself instead of reading what a separate one-block rt_u kernel wrote.
//
// The earlier all-shapes GGML_MACH1_FUSE_RT was catastrophic (-30%) because at
// m = 8192 it made 512 blocks each redo the transform. But the redundancy
// costs WORK, not TIME - the blocks run concurrently - and a single-block FWHT
// kernel measures ~4.2 us almost independently of its size (mb3.cu), i.e. it
// is nearly all launch and block latency. So on a shape whose grid is small
// the duplicate transform hides and an entire kernel launch plus a graph node
// disappear. 100 of the 250 rt decode ops have m = 512, i.e. 32 blocks.
//
// Bit-exact: same mach1_fwht_block, same inputs, same pinned round - the
// values are identical to what rt_u would have written, they are simply never
// round-tripped through global memory.
//
// TCF = 1 (GGML_MACH1_TC_FWHT, decode only, NOT bit-exact - KL-gated): the
// UFUSE prologue and the OFUSE epilogue run as the tensor-core Kronecker
// two-dot instead of the butterfly; the dynamic shared carries the TC
// workspace instead of the m-float obuf, and htab points at the +-1 H tiles.
// Replication factor for the lane-addressed codebook below. 0 keeps the
// two-fp32-array layout; 4/8/16 select a packed half2 table replicated that
// many times. Anything else falls back to 0.
//
// 32 - the fully conflict-free factor the canonical kernels use - is not
// offered: 512*32 words is 64 KB of STATIC shared, over the 48 KB a static
// __shared__ array may occupy, and this walk already spends 8 KB on shu.
// 16 is the largest that fits (32 KB + 8 KB), and it leaves at most a 2-way
// collision where 32 would leave none.
static int mach1_lutx() {
    static const int v = mach1_env_int("GGML_MACH1_LUTX", 0);
    return (v == 4 || v == 8 || v == 16) ? v : 0;
}

// LUTX > 0 (GGML_MACH1_LUTX): the codebook is stored as one packed half2 per
// row, replicated LUTX times, and addressed as row*LUTX + (lane & (LUTX-1)).
//
// The default layout takes a data-dependent bank conflict on every lookup:
// slutA[row] with row pseudorandom per lane means 32 lanes draw 32 unrelated
// indices out of 512, and shared memory serialises whatever collides - about
// 3-4 ways on average, twice, because a0 and a1 are separate fp32 arrays.
// Folding the lane into the ADDRESS makes the bank a function of the lane
// instead of the data, so the access is conflict-free by construction, and
// packing the pair as half2 makes it one 4-byte load instead of two.
//
// Replication is what buys that, and it is not free: LUTX = 32 is fully
// conflict-free but stages 16384 words per block. The canonical kernels can
// afford that because they run a persistent grid (one block per SM, looping
// over the whole matrix); this walk launches m/16 * nt short-lived blocks -
// 512 at m = 8192 - so the staging is paid per block and small LUTX is the
// operating point. LUTX = 8 removes most of the collisions for 8 stores per
// thread against the current 2.
//
// The sign is the same trick: bit 15 of the packed word IS the sign bit of
// the low half, so XOR-ing it with the phase bit replaces the branch.
template <int WG, int RG, int UFUSE, int OFUSE = 0, int TCF = 0, int LUTX = 0, int ZDP = 0>
__global__ void __launch_bounds__(WG) mach1_rt_walk_rows_v_kernel(
        const uint16_t * __restrict__ trellis,
        const half     * __restrict__ tlut,     // pre-rounded fp16
        const float    * __restrict__ scr_u,    // UFUSE = 0
        const float    * __restrict__ su,       // UFUSE = 1
        const float    * __restrict__ x,        // UFUSE = 1
        float          * __restrict__ scr_v,
        const int m, const int n,
        const float    * __restrict__ sv,       // OFUSE = 1
        float          * __restrict__ dst,      // OFUSE = 1
        int            * __restrict__ counter,  // OFUSE = 1
        const int      * __restrict__ operm,    // OFUSE = 1, optional row scatter
        const half     * __restrict__ htab,     // TCF = 1
        const uint16_t * __restrict__ ztab,     // ZDP = 1
        const float zs0,                        // ZDP = 1
        const int      * __restrict__ xperm) {  // XPERM: gather x through a row map (or nullptr)
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int CT        = WG / RG;
    constexpr int n_warps   = CT / warp_size;
    constexpr int ROWS      = 16 / RG;
    __shared__ float red[RG][16/RG][CT/32];
    __shared__ float slutA[(LUTX || ZDP) ? 1 : 512];
    __shared__ float slutB[(LUTX || ZDP) ? 1 : 512];
    __shared__ uint32_t slutX[LUTX ? 512*LUTX : 1];
    __shared__ uint32_t zslutr[ZDP ? 4096 : 1];   // 4-way replicated
    __shared__ float shu[CT*16];
    extern __shared__ float obuf[];   // OFUSE: [m]; TCF: the TC workspace

    const int t   = blockIdx.z;
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;
    const int ct  = tid % CT;
    const int rg  = tid / CT;

    if (ZDP) {
        for (int i = tid; i < 1024; i += WG) {
            const uint32_t w = ztab[i];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                zslutr[(i << 2) + r] = w;
            }
        }
    } else if (LUTX) {
        for (int i = tid; i < 512; i += WG) {
            const uint32_t w = (uint32_t) __half_as_ushort(tlut[2*i + 0])
                | ((uint32_t) __half_as_ushort(tlut[2*i + 1]) << 16);
#pragma unroll
            for (int r = 0; r < (LUTX ? LUTX : 1); ++r) {
                slutX[i*(LUTX ? LUTX : 1) + r] = w;
            }
        }
    } else {
        for (int i = tid; i < 512; i += WG) {
            slutA[i] = __half2float(tlut[2*i + 0]);
            slutB[i] = __half2float(tlut[2*i + 1]);
        }
    }
    if (UFUSE && TCF) {
        mach1_tc_fwht_dyn<WG, false>(n, x + (int64_t) t*n, su, shu, nullptr, nullptr,
                                     htab, (half *) obuf, tid);
    } else if (UFUSE) {
        // exactly mach1_rt_u_kernel's body, into shared instead of scr_u.
        // XPERM: the row map moves the load's ADDRESS only - the values are
        // the ones the skipped cont would have materialized, bit for bit.
        for (int i = tid; i < n; i += WG) {
            shu[i] = su[i] * x[(int64_t) t*n + (xperm != nullptr ? xperm[i] : i)];
        }
        __syncthreads();
        mach1_fwht_block(shu, n, tid, WG);
        const float sc = __fsqrt_rn((float) n);
        for (int i = tid; i < n; i += WG) {
            shu[i] = __fdiv_rn(shu[i], sc);
        }
    } else {
        for (int i = tid; i < n; i += WG) {
            shu[i] = scr_u[(int64_t) t*n + i];
        }
    }
    __syncthreads();

    const int  tiles_y = n / 16;
    const bool have    = ct < tiles_y;

    float partial[ROWS];
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
        partial[i] = 0.0f;
    }

    if (have) {
        const float * ub = shu + ct*16;
        float uu[16];
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            uu[c] = ub[c];
        }
        uint32_t qq[4];
        float    fsc = 0.0f;
        if (ZDP) {
            fsc = zs0*mach1_rt_zdp_quant(uu, qq)*(1.0f/127.0f);
        }
        // 3 x 128-bit covers words [16rg, 16rg+24), and (2rg+q) & 7 wraps to
        // the tile's word 0 exactly where the scalar form's & 63 does.
        const uint4 * base = (const uint4 *) (trellis + ((int64_t) tr*tiles_y + ct)*64);
        uint4 v[3];
#pragma unroll
        for (int q = 0; q < 3; ++q) {
            v[q] = base[(2*rg + q) & 7];
        }
        const uint32_t * V = (const uint32_t *) v;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            // The eight 16-bit windows of a row out of three uint32, with two
            // PRMT and four funnel shifts and NO mask.
            //
            // V[2r+k] packs words (4r+2k, 4r+2k+1) as (w_hi << 16) | w_lo, so
            // A, B already carry w0, w2 in their low halves, and byte_perm
            // 0x5432 rebuilds the (w2<<16)|w1 and (w4<<16)|w3 views that carry
            // w1, w3. A rotate-left-8 of any of those four words puts the odd
            // window - (w_k << 8) | (w_k+1 >> 8) - in its low half.
            //
            // The mask the scalar form applies is dead work: bit i of a
            // product depends only on bits 0..i of the operands, and the only
            // bits of ph ever read are 6..15 (the tlut row) and 15 (the sign),
            // so leaving the high half dirty cannot change the result. Proven
            // rather than argued - mb3.cu compares this against the reference
            // walk on random trellis bytes over all five decode shapes and
            // reports 0/8192, 0/2048, 0/512, 0/2048, 0/4096 mismatches.
            const uint32_t A  = V[2*r + 0];
            const uint32_t B  = V[2*r + 1];
            const uint32_t C  = V[2*r + 2];
            const uint32_t Ap = __byte_perm(A, B, 0x5432);
            const uint32_t Bp = __byte_perm(B, C, 0x5432);
            uint32_t st[8];
            st[0] = A;   st[1] = __funnelshift_l(A,  A,  8);
            st[2] = Ap;  st[3] = __funnelshift_l(Ap, Ap, 8);
            st[4] = B;   st[5] = __funnelshift_l(B,  B,  8);
            st[6] = Bp;  st[7] = __funnelshift_l(Bp, Bp, 8);
            uint32_t ph[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                ph[j] = st[j]*(st[j] + 1u);
            }
            if (ZDP) {
                partial[r] = (float) mach1_rt_zdp_row_rep<4>(ph, zslutr, qq, ct & 3)*fsc;
            } else if (LUTX) {
                constexpr int REP = LUTX ? LUTX : 1;
                const int     sel = ct & (REP - 1);
                float acc = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint32_t row = (ph[j] >> 6) & 511u;
                    // one conflict-free 4-byte load, and the sign is an XOR of
                    // bit 15 - the low half's sign bit - instead of a branch
                    const uint32_t w = slutX[row*REP + sel] ^ (ph[j] & 0x8000u);
                    const float2   a = __half22float2(*(const half2 *) &w);
                    acc += a.x*uu[2*j] + a.y*uu[2*j + 1];
                }
                partial[r] = acc;
            } else {
                float acc = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint32_t row = (ph[j] >> 6) & 511u;
                    float a0 = slutA[row];
                    const float a1 = slutB[row];
                    if (ph[j] & 0x8000u) {
                        a0 = -a0;
                    }
                    acc += a0*uu[2*j] + a1*uu[2*j + 1];
                }
                partial[r] = acc;
            }
        }
    }

    const int lane = ct % warp_size;
    const int wid  = ct / warp_size;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const float s = warp_reduce_sum<warp_size>(partial[r]);
        if (lane == 0) {
            red[rg][r][wid] = s;
        }
    }
    __syncthreads();
    if (tid < 16) {
        const int g = tid / ROWS;
        const int r = tid % ROWS;
        float sum = 0.0f;
#pragma unroll
        for (int wj = 0; wj < n_warps; ++wj) {
            sum += red[g][r][wj];
        }
        scr_v[(int64_t) t*m + tr*16 + g*ROWS + r] = sum;
    }

    // OFUSE: the last block to publish its tile row runs the out stage
    // (y = sv .* H_m(v)/sqrt(m), mach1_rt_out_kernel's body) itself - one
    // single-block launch and its ~4 us fixed cost deleted per op. FWHT
    // values do not depend on WG and the scale order is unchanged: bit-exact.
    // Requires nt == 1 (enforced at the launch site).
    if constexpr (OFUSE) {
        __shared__ int is_last;
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            is_last = atomicAdd(counter, 1) == gridDim.x - 1;
        }
        __syncthreads();
        if (!is_last) {
            return;
        }
        if (tid == 0) {
            *counter = 0;   // self-reset for the next op on this stream
        }
        __threadfence();
        if (TCF) {
            mach1_tc_fwht_dyn<WG, true>(m, scr_v, nullptr, dst, sv, operm,
                                        htab, (half *) obuf, tid);
            return;
        }
        for (int i = tid; i < m; i += WG) {
            obuf[i] = __ldcg(&scr_v[i]);
        }
        __syncthreads();
        mach1_fwht_block(obuf, m, tid, WG);
        const float osc = __fsqrt_rn((float) m);
        if (operm == nullptr) {
            for (int i = tid; i < m; i += WG) {
                dst[i] = __fdiv_rn(obuf[i], osc) * sv[i];
            }
        } else {
            // vtiled fold: identical values stored at the permuted row
            for (int i = tid; i < m; i += WG) {
                dst[operm[i]] = __fdiv_rn(obuf[i], osc) * sv[i];
            }
        }
    } else {
        (void) sv; (void) dst; (void) counter; (void) operm; (void) htab;
    }
}

// TT-token walk (GGML_MACH1_WALK_TT, nt >= 2): decode once, MAC TT tokens.
//
// The nt >= 2 walk above re-runs the WHOLE decode - trellis loads, the
// ph = st*(st+1) chain, the codebook gathers - once per blockIdx.z token,
// and the r258 census prices that redundancy at 4.08 ms/step of the nt=2
// decode (the fattest single line left in the serving regime). But the
// decode depends only on the ROW, not the token: this form runs the decode
// phase once per thread, caches the decoded row windows in registers
// (fp16: the LUTX packed half2 with the sign XORed in; ZDP: the dp4a
// operand words), then MACs TT staged u vectors against the cache. Decode
// cost divides by TT; only the MAC scales with tokens.
//
// Per token this is BIT-IDENTICAL to the walk above: same windows, same
// j-order, same adds (fp16: the LUTX equivalence, proven in mb3.cu; ZDP:
// the same per-token quantization and dp4a order).
//
// Shared budget (static 48 KB): shu is TT*n floats, so TT = 4 fits n <=
// 2048 (32 KB) beside an REP = 2 z-table (8 KB), and n = 4096 caps at
// TT = 2 under WG 1024. The table replication drops from 4 to 2 at TT = 4
// for the same reason the batch kernel runs REP 2 - the budget, not the
// banks.
// OFUSE = 1 (single-chunk grids only, nt <= TT): the rt_out stage - the
// mach1_rt_out_kernel body per token - runs in the walk's LAST block instead
// of its own launch. rt_out sits on the data-dependency chain (its output is
// the next op's input), so unlike the sibling transforms it cannot pipeline
// under replay; folding it removes one serial link per rt op. Values are
// mach1_rt_out_kernel's bit for bit: same staging, same FWHT, same fdiv and
// sv order.
// RG = 8 (WG 1024, CT still 128): the latency-bound answer for small-m
// shapes at nt >= 2 - twice the threads on the same tiles, half the rows
// per thread. The uint4 window start moves from 2*rg to rg (each group now
// spans 2 rows = 4 uint32 = 1 uint4) and the in-window row indexing is
// unchanged; same states, same order, bit-identical (the rows_v RG-split
// argument, and shexp_down's RGT=16 precedent for the window shift).
// UF = 1: the u stage folds into this kernel - each block computes the
// FWHT(su .* x) of its chunk's tokens into shu directly (bit-identical to
// mach1_rt_u_kernel's op order) and scr_u is never read. The ufuse
// measurement (redundancy cheaper than the launch to 256-block grids)
// carries: the prologue costs ~1.2 us/token/block in parallel, the u launch
// it removes is a ~8 us serial stage on the op's dependency chain.
// the body, callable from the plain kernel and the pair-select wrapper (the
// bnt_split region merges independent walks into one grid); tr_in replaces
// blockIdx.x so a wrapper can rebase the row-tile index per op
template <int WG, int TT, int ZDP, int OFUSE = 0, int RG = 4, int UF = 0>
static __device__ __forceinline__ void mach1_rt_walk_tt_body(
        const uint16_t * __restrict__ trellis,
        const half     * __restrict__ tlut,     // pre-rounded fp16
        const float    * __restrict__ scr_u,    // [nt, n] (UF = 0)
        float          * __restrict__ scr_v,    // [nt, m]
        const int m, const int n, const int nt,
        const uint16_t * __restrict__ ztab,     // ZDP = 1
        const float zs0,
        const float    * __restrict__ sv,       // OFUSE = 1
        float          * __restrict__ dst,      // OFUSE = 1
        int            * __restrict__ counter,  // OFUSE = 1
        const float    * __restrict__ uf_su,    // UF = 1
        const float    * __restrict__ uf_x,     // UF = 1: x rows [nt, n]
        const int tr_in) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int CT        = WG / RG;
    constexpr int n_warps   = CT / warp_size;
    constexpr int ROWS      = 16 / RG;
    static_assert(RG == 4 || RG == 8, "window derivation covers RG 4 and 8");
    // replication shrinks where shu grows - the 48 KB static budget, not banks
    constexpr int REP       = (TT >= 4 || WG >= 1024) ? 2 : 4;   // z-table
    constexpr int XREP      = (TT >= 4 || WG >= 1024) ? 4 : 8;   // fp16 table
    __shared__ float    red[TT][RG][16/RG][CT/32];
    __shared__ uint32_t slutX[ZDP ? 1 : 512*XREP];
    __shared__ uint32_t zslutr[ZDP ? 1024*REP : 1];
    __shared__ float    shu[TT*CT*16];
    extern __shared__ float obuf[];   // OFUSE: [m]

    const int t0  = blockIdx.z*TT;
    const int ntc = min(TT, nt - t0);   // tokens in this chunk
    const int tr  = tr_in;
    const int tid = threadIdx.x;
    const int ct  = tid % CT;
    const int rg  = tid / CT;

    if (ZDP) {
        (void) tlut;
    } else {
        (void) ztab; (void) zs0;
    }

    if (ZDP) {
        for (int i = tid; i < 1024; i += WG) {
            const uint32_t w = ztab[i];
#pragma unroll
            for (int r = 0; r < REP; ++r) {
                zslutr[i*REP + r] = w;
            }
        }
    } else {
        for (int i = tid; i < 512; i += WG) {
            const uint32_t w = (uint32_t) __half_as_ushort(tlut[2*i + 0])
                | ((uint32_t) __half_as_ushort(tlut[2*i + 1]) << 16);
#pragma unroll
            for (int r = 0; r < XREP; ++r) {
                slutX[i*XREP + r] = w;
            }
        }
    }
    if (UF) {
        // exactly mach1_rt_u_kernel's body per token, into shu (the
        // shexp_gu prologue pattern: load, sync, fwht, scale, sync)
        for (int tt = 0; tt < ntc; ++tt) {
            float       * sh = shu + tt*CT*16;
            const float * xr = uf_x + (int64_t)(t0 + tt)*n;
            for (int i = tid; i < n; i += WG) {
                sh[i] = uf_su[i]*xr[i];
            }
        }
        __syncthreads();
        const float usc = __fsqrt_rn((float) n);
        for (int tt = 0; tt < ntc; ++tt) {
            mach1_fwht_block(shu + tt*CT*16, n, tid, WG);
            for (int i = tid; i < n; i += WG) {
                shu[tt*CT*16 + i] = __fdiv_rn(shu[tt*CT*16 + i], usc);
            }
            __syncthreads();
        }
    } else {
        for (int tt = 0; tt < ntc; ++tt) {
            for (int i = tid; i < n; i += WG) {
                shu[tt*CT*16 + i] = scr_u[(int64_t)(t0 + tt)*n + i];
            }
        }
        __syncthreads();
    }

    const int  tiles_y = n / 16;
    const bool have    = ct < tiles_y;

    float partial[TT][ROWS];
#pragma unroll
    for (int tt = 0; tt < TT; ++tt) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            partial[tt][r] = 0.0f;
        }
    }

    if (have) {
        // decode phase: the row windows once, cached for every token
        const uint4 * base = (const uint4 *) (trellis + ((int64_t) tr*tiles_y + ct)*64);
        uint4 v[3];
#pragma unroll
        for (int q = 0; q < 3; ++q) {
            v[q] = base[((RG == 8 ? rg : 2*rg) + q) & 7];
        }
        const uint32_t * V = (const uint32_t *) v;
        uint32_t wc[ROWS][ZDP ? 4 : 8];   // ZDP: dp4a operands; else signed half2
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const uint32_t A  = V[2*r + 0];
            const uint32_t B  = V[2*r + 1];
            const uint32_t C  = V[2*r + 2];
            const uint32_t Ap = __byte_perm(A, B, 0x5432);
            const uint32_t Bp = __byte_perm(B, C, 0x5432);
            uint32_t st[8];
            st[0] = A;   st[1] = __funnelshift_l(A,  A,  8);
            st[2] = Ap;  st[3] = __funnelshift_l(Ap, Ap, 8);
            st[4] = B;   st[5] = __funnelshift_l(B,  B,  8);
            st[6] = Bp;  st[7] = __funnelshift_l(Bp, Bp, 8);
            if (ZDP) {
                const int sel = ct & (REP - 1);
#pragma unroll
                for (int j = 0; j < 8; j += 2) {
                    const uint32_t p0  = st[j]*(st[j] + 1u);
                    const uint32_t p1  = st[j + 1]*(st[j + 1] + 1u);
                    const uint32_t z01 = zslutr[((p0 >> 6) & 1023u)*REP + sel];
                    const uint32_t z23 = zslutr[((p1 >> 6) & 1023u)*REP + sel];
                    wc[r][j >> 1] = __byte_perm(z01, z23, 0x5410);
                }
            } else {
                const int sel = ct & (XREP - 1);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint32_t ph = st[j]*(st[j] + 1u);
                    wc[r][j] = slutX[((ph >> 6) & 511u)*XREP + sel] ^ (ph & 0x8000u);
                }
            }
        }

        // MAC phase: each token reads the cache, never the trellis
        for (int tt = 0; tt < ntc; ++tt) {
            const float * ub = shu + tt*CT*16 + ct*16;
            if (ZDP) {
                float uu[16];
#pragma unroll
                for (int c = 0; c < 16; ++c) {
                    uu[c] = ub[c];
                }
                uint32_t qq[4];
                const float fsc = zs0*mach1_rt_zdp_quant(uu, qq)*(1.0f/127.0f);
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    int iacc = 0;
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        iacc = ggml_cuda_dp4a((int) wc[r][k], (int) qq[k], iacc);
                    }
                    partial[tt][r] = (float) iacc*fsc;
                }
            } else {
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    float acc = 0.0f;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const uint32_t w = wc[r][j];
                        const float2   a = __half22float2(*(const half2 *) &w);
                        acc += a.x*ub[2*j] + a.y*ub[2*j + 1];
                    }
                    partial[tt][r] = acc;
                }
            }
        }
    }

    const int lane = ct % warp_size;
    const int wid  = ct / warp_size;
#pragma unroll
    for (int tt = 0; tt < TT; ++tt) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const float s = warp_reduce_sum<warp_size>(partial[tt][r]);
            if (lane == 0) {
                red[tt][rg][r][wid] = s;
            }
        }
    }
    __syncthreads();
    if (tid < 16) {
        const int g = tid / ROWS;
        const int r = tid % ROWS;
        for (int tt = 0; tt < ntc; ++tt) {
            float sum = 0.0f;
#pragma unroll
            for (int wj = 0; wj < n_warps; ++wj) {
                sum += red[tt][g][r][wj];
            }
            scr_v[(int64_t)(t0 + tt)*m + tr*16 + g*ROWS + r] = sum;
        }
    }

    if constexpr (OFUSE) {
        // single chunk by the dispatch gate, so gridDim.x is the whole grid
        __shared__ int is_last;
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            is_last = atomicAdd(counter, 1) == gridDim.x - 1;
        }
        __syncthreads();
        if (!is_last) {
            return;
        }
        if (tid == 0) {
            *counter = 0;   // self-reset for the next op on this stream
        }
        __threadfence();
        const float osc = __fsqrt_rn((float) m);
        for (int tt = 0; tt < ntc; ++tt) {
            for (int i = tid; i < m; i += WG) {
                obuf[i] = __ldcg(&scr_v[(int64_t)(t0 + tt)*m + i]);
            }
            __syncthreads();
            mach1_fwht_block(obuf, m, tid, WG);
            for (int i = tid; i < m; i += WG) {
                dst[(int64_t)(t0 + tt)*m + i] = __fdiv_rn(obuf[i], osc) * sv[i];
            }
            __syncthreads();
        }
    } else {
        (void) sv; (void) dst; (void) counter;
    }
}

// MINB = 2 pins two blocks per SM (the r238 lesson: the register file, not
// the workspace, caps this walk's residency - telling ptxas the budget is
// what converts). At TT = 2 the kernel already sits at 64 regs so the pin
// costs nothing; at TT = 4 it forces ~96 -> 64 with spill - the race judges.
template <int WG, int TT, int ZDP, int OFUSE = 0, int RG = 4, int UF = 0, int MINB = 1>
__global__ void __launch_bounds__(WG, MINB) mach1_rt_walk_tt_kernel(
        const uint16_t * __restrict__ trellis,
        const half     * __restrict__ tlut,
        const float    * __restrict__ scr_u,
        float          * __restrict__ scr_v,
        const int m, const int n, const int nt,
        const uint16_t * __restrict__ ztab,
        const float zs0,
        const float    * __restrict__ sv,
        float          * __restrict__ dst,
        int            * __restrict__ counter,
        const float    * __restrict__ uf_su = nullptr,
        const float    * __restrict__ uf_x  = nullptr) {
    mach1_rt_walk_tt_body<WG, TT, ZDP, OFUSE, RG, UF>(trellis, tlut, scr_u, scr_v,
        m, n, nt, ztab, zs0, sv, dst, counter, uf_su, uf_x, (int) blockIdx.x);
}

// Signed 1K-LUT walk (GGML_MACH1_LUT1K=1). Same walk, one shared gather per
// state instead of two, and no per-state sign work.
//
// Only bits 6..15 of ph = st*(st+1) are ever consumed: bits 6..14 pick the
// tlut row and bit 15 negates a0. That is 10 bits, so the whole state->pair
// map fits in 1024 entries with the sign already folded in. Per state this
// drops the &511 mask, the sign test and the negate, and merges slutA/slutB
// (two LDS.32 at random rows) into one LDS.64 of a float2.
//
// BIT-EXACT: slutS[i] holds exactly (+/-)slutA[i & 511] and slutB[i & 511],
// negation of a float is exact, and the accumulation order is unchanged.
// Shared cost is 8 KB instead of 4 KB, which does not change occupancy here
// (the 16 KB trellis staging dominates).
template <int WG, int RG>
__global__ void mach1_rt_walk_rows_s_kernel(
        const uint16_t * __restrict__ trellis,
        const half     * __restrict__ tlut,     // pre-rounded fp16
        const float    * __restrict__ scr_u,
        float          * __restrict__ scr_v,
        const int m, const int n) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int CT        = WG / RG;
    constexpr int n_warps   = CT / warp_size;
    __shared__ float  red[RG][16/RG][CT/32];
    __shared__ float2 slutS[1024];
    __shared__ uint16_t stre[CT*64];

    const int t   = blockIdx.z;
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;
    const int ct  = tid % CT;
    const int rg  = tid / CT;

    for (int i = tid; i < 1024; i += WG) {
        const int   row = i & 511;
        const float a0  = __half2float(tlut[2*row + 0]);
        const float a1  = __half2float(tlut[2*row + 1]);
        slutS[i] = make_float2((i & 512) ? -a0 : a0, a1);
    }
    {
        const int      tiles   = n / 16;
        const int64_t  tbase_w = (int64_t) tr*tiles*64;
        for (int i = tid; i < tiles*64; i += WG) {
            stre[i] = trellis[tbase_w + i];
        }
    }
    __syncthreads();

    constexpr int words   = 64;                   // K = 4
    constexpr int ROWS    = 16/RG;
    const int     tiles_y = n / 16;
    const bool    have    = ct < tiles_y;

    float partial[ROWS];
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
        partial[i] = 0.0f;
    }

    if (have) {
        const int     tw = ct*words;
        const float * ub = scr_u + (int64_t) t*n + ct*16;

        float uu[16];
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            uu[c] = ub[c];
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int ri = rg*ROWS + r;
            uint32_t w[5];
#pragma unroll
            for (int q = 0; q < 5; ++q) {
                w[q] = stre[tw + ((4*ri + q) & 63)];
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
                const float2 a = slutS[(ph[j] >> 6) & 1023u];
                acc += a.x*uu[2*j] + a.y*uu[2*j + 1];
            }
            partial[r] = acc;
        }
    }

    const int lane = ct % warp_size;
    const int wid  = ct / warp_size;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const float s = warp_reduce_sum<warp_size>(partial[r]);
        if (lane == 0) {
            red[rg][r][wid] = s;
        }
    }
    __syncthreads();
    if (tid < 16) {
        const int g = tid / ROWS;
        const int r = tid % ROWS;
        float sum = 0.0f;
#pragma unroll
        for (int wj = 0; wj < n_warps; ++wj) {
            sum += red[g][r][wj];
        }
        scr_v[(int64_t) t*m + tr*16 + g*ROWS + r] = sum;
    }
}

// Cooperative fused rt op: u-FWHT, walk and out-FWHT in ONE launch.
//
// Measured motivation (round 19 probes): the trellis decode is only 18% of
// the step, while probe3 (walk with the decode removed) -> fold (one GEMV per
// op) is a further ~10% - that gap is purely 3 kernels per op versus 1. The
// earlier naive fusion regressed (55.2 vs 68.3) because every block recomputed
// the u stage; a grid barrier computes it ONCE and lets the rest read it.
//
// Residency: the largest shape is m/16 = 512 blocks x 512 threads = 262144
// threads (H200 allows 270336), 4 blocks/SM at 32 KB shared each = 128 KB of
// the SM's 228 KB. The launch is checked with the occupancy API and falls
// back to the split chain if the grid would not be co-resident.
//
// Bit-exact: identical stage bodies, identical order. sh is reused across
// phases, which is safe because each phase is separated by a grid barrier.
template <int WG>
__global__ void mach1_rt_coop_kernel(
        const uint16_t * __restrict__ trellis,
        const half     * __restrict__ tlut,
        const float    * __restrict__ su,
        const float    * __restrict__ sv,
        const float    * __restrict__ x,
        float          * __restrict__ scr_u,
        float          * __restrict__ scr_v,
        float          * __restrict__ dst,
        const int m, const int n) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int RG   = 4;
    constexpr int CT   = WG / RG;
    constexpr int ROWS = 16 / RG;
    constexpr int n_warps = CT / warp_size;

    __shared__ float sh[8192];          // FWHT scratch, then the tlut
    __shared__ float red[RG][ROWS][CT/32];   // CT/32 warps per row group

    const int tid = threadIdx.x;
    const int tr  = blockIdx.x;

    // phase 1: u = H(su .* x)/sqrt(n)  (same body as mach1_rt_u_kernel)
    if (tr == 0) {
        for (int i = tid; i < n; i += WG) {
            sh[i] = su[i]*x[i];
        }
        __syncthreads();
        mach1_fwht_block(sh, n, tid, WG);
        const float sc = __fsqrt_rn((float) n);
        for (int i = tid; i < n; i += WG) {
            scr_u[i] = __fdiv_rn(sh[i], sc);
        }
    }
    grid.sync();

    // phase 2: walk (row-split layout, bit-identical partials/reduction)
    {
        float * slutA = sh;
        float * slutB = sh + 512;
        float * shu   = sh + 1024;          // n <= 2048 here (launch guard)
        for (int i = tid; i < 512; i += WG) {
            slutA[i] = __half2float(tlut[2*i + 0]);
            slutB[i] = __half2float(tlut[2*i + 1]);
        }
        // u once per block instead of once per (row group, column tile)
        for (int i = tid; i < n; i += WG) {
            shu[i] = scr_u[i];
        }
        __syncthreads();

        const int  ct      = tid % CT;
        const int  rg      = tid / CT;
        const int  tiles_y = n / 16;
        const bool have    = ct < tiles_y;

        float partial[ROWS];
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            partial[i] = 0.0f;
        }
        if (have) {
            const float * ub = shu + ct*16;
            float uu[16];
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                uu[c] = ub[c];
            }
            // same vector load + byte_perm window build as the split kernel
            const uint4 * base = (const uint4 *) (trellis + ((int64_t) tr*tiles_y + ct)*64);
            uint4 v[3];
#pragma unroll
            for (int q = 0; q < 3; ++q) {
                v[q] = base[(2*rg + q) & 7];
            }
            const uint32_t * V = (const uint32_t *) v;
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const uint32_t A  = V[2*r + 0];
                const uint32_t B  = V[2*r + 1];
                const uint32_t C  = V[2*r + 2];
                const uint32_t Ap = __byte_perm(A, B, 0x5432);
                const uint32_t Bp = __byte_perm(B, C, 0x5432);
                uint32_t st[8];
                st[0] = A;   st[1] = __funnelshift_l(A,  A,  8);
                st[2] = Ap;  st[3] = __funnelshift_l(Ap, Ap, 8);
                st[4] = B;   st[5] = __funnelshift_l(B,  B,  8);
                st[6] = Bp;  st[7] = __funnelshift_l(Bp, Bp, 8);
                uint32_t ph[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    ph[j] = st[j]*(st[j] + 1u);
                }
                float acc = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint32_t row = (ph[j] >> 6) & 511u;
                    float a0 = slutA[row];
                    const float a1 = slutB[row];
                    if (ph[j] & 0x8000u) {
                        a0 = -a0;
                    }
                    acc += a0*uu[2*j] + a1*uu[2*j + 1];
                }
                partial[r] = acc;
            }
        }
        const int lane = ct % warp_size;
        const int wid  = ct / warp_size;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const float s = warp_reduce_sum<warp_size>(partial[r]);
            if (lane == 0) {
                red[rg][r][wid] = s;
            }
        }
        __syncthreads();
        if (tid < 16) {
            const int g = tid / ROWS;
            const int r = tid % ROWS;
            float sum = 0.0f;
#pragma unroll
            for (int wj = 0; wj < n_warps; ++wj) {
                sum += red[g][r][wj];
            }
            scr_v[tr*16 + g*ROWS + r] = sum;
        }
    }
    grid.sync();

    // phase 3: y = sv .* H(v)/sqrt(m)  (same body as mach1_rt_out_kernel)
    if (tr == 0) {
        __syncthreads();
        for (int i = tid; i < m; i += WG) {
            sh[i] = scr_v[i];
        }
        __syncthreads();
        mach1_fwht_block(sh, m, tid, WG);
        const float sc = __fsqrt_rn((float) m);
        for (int i = tid; i < m; i += WG) {
            dst[i] = __fdiv_rn(sh[i], sc) * sv[i];
        }
    }
}

// dense prefill: decode the K4/V2 trellis ONCE into an fp16 bank [m, n] —
// one thread per (tile, tile-row), same window/state math as rt_walk, same
// pre-rounded tlut halves (sign flip is exact), then apply the bank with a
// plain fp32 GEMM-ish kernel. Replaces only the walk stage; rt_u/rt_out are
// untouched.

static __global__ void mach1_rt_dense_decode_kernel(
        const uint16_t * __restrict__ trellis,
        const half     * __restrict__ tlut,     // pre-rounded fp16
        half           * __restrict__ bank,     // [m, n]
        const int m, const int n) {
    constexpr int WG = 256;
    __shared__ half slut[1024];

    const int tid = threadIdx.x;

    for (int i = tid; i < 1024; i += WG) {
        slut[i] = tlut[i];
    }
    __syncthreads();

    const int tiles_y = n / 16;
    const int gx = blockIdx.x*WG + tid;
    if (gx >= (m/16)*16*tiles_y) {   // one thread per (tile, tile-row)
        return;
    }
    // column tile fastest: a warp writes 16 consecutive 16-half row chunks
    const int tc     = gx % tiles_y;
    const int rowall = gx / tiles_y;
    const int tr     = rowall >> 4;
    const int rr     = rowall & 15;

    const int64_t tw = ((int64_t) tr*tiles_y + tc)*64;   // K = 4: 64 words/tile
    uint32_t w[5];
#pragma unroll
    for (int q = 0; q < 5; ++q) {
        w[q] = trellis[tw + ((4*rr + q) & 63)];   // only q=4,rr=15 wraps
    }
    half * dst = bank + (int64_t)(tr*16 + rr)*n + tc*16;
    half tmp[16];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t hi = w[j >> 1];
        const uint32_t st = (j & 1) == 0 ? hi
            : (((hi << 8) | (w[(j >> 1) + 1] >> 8)) & 0xFFFFu);
        const uint32_t ph  = st*(st + 1u);
        const uint32_t row = (ph >> 6) & 511u;
        const float a0 = __half2float(slut[2*row + 0]);
        // negate-then-round-trip is bit-exact for fp16 values (sign bit)
        tmp[2*j + 0] = __float2half_rn((ph & 0x8000u) ? -a0 : a0);
        tmp[2*j + 1] = slut[2*row + 1];
    }
    mach1_store_half16(dst, tmp);   // same bits, 2x16B instead of 16x2B
}

// grid (m/16, 1, ceil(nt/TC)): each block owns 16 output rows and a TC-token
// chunk; per column-stride iteration the 16-row W slice is held in registers
// and reused across the chunk's tokens, so bank traffic is amortized TC-fold
// (per-chunk L2 re-reads keep grid-level parallelism, which a single token
// loop over grid (m/16) blocks would not). TC is an A/B knob
// (GGML_MACH1_RT_TC, 4 or 8); numerics are TC-independent (per-token order
// unchanged, tail clamps are write-guarded).
template <int TC>
__global__ void mach1_rt_apply_kernel(
        const half  * __restrict__ bank,     // [m, n]
        const float * __restrict__ scr_u,    // [nt, n]
        float       * __restrict__ scr_v,    // [nt, m]
        const int m, const int n, const int nt) {
    constexpr int WG        = 128;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = WG / warp_size;
    __shared__ float red[TC][16][4];   // >= n_warps for warp_size 32 (and 64 on HIP)

    const int tr  = blockIdx.x;
    const int t0  = blockIdx.z*TC;
    const int tid = threadIdx.x;

    const half * W = bank + (int64_t) tr*16*n;

    float acc[TC][16];
#pragma unroll
    for (int t = 0; t < TC; ++t) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            acc[t][i] = 0.0f;
        }
    }

    // half2/float2 loads: each thread covers column pair (c0, c0+1) — halves
    // the load instruction count. The two columns' terms are accumulated into
    // the same per-thread partial instead of two lanes' partials, a
    // contractible fp32 reassociation (the apply path's summation order is
    // already declared contractible).
    for (int c0 = 2*tid; c0 < n; c0 += 2*WG) {
        float2 wc[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            wc[i] = __half22float2(*(const half2 *)(W + (int64_t) i*n + c0));
        }
#pragma unroll
        for (int t = 0; t < TC; ++t) {
            // tail chunk: clamp (duplicate compute, write-guarded below)
            const int tt = min(t0 + t, nt - 1);
            const float2 uc = *(const float2 *)(scr_u + (int64_t) tt*n + c0);
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                acc[t][i] += wc[i].x*uc.x + wc[i].y*uc.y;
            }
        }
    }

    const int lane = tid % warp_size;
    const int wid  = tid / warp_size;
#pragma unroll
    for (int t = 0; t < TC; ++t) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float s = warp_reduce_sum<warp_size>(acc[t][i]);
            if (lane == 0) {
                red[t][i][wid] = s;
            }
        }
    }
    __syncthreads();
    if (tid < TC*16) {
        const int t = tid >> 4;
        const int i = tid & 15;
        if (t0 + t < nt) {
            float sum = 0.0f;
#pragma unroll
            for (int wj = 0; wj < n_warps; ++wj) {
                sum += red[t][i][wj];
            }
            scr_v[(int64_t)(t0 + t)*m + tr*16 + i] = sum;
        }
    }
}

// PP lane (GGML_MACH1_RT_APPLY_TC): round scr_u to fp16 so the bank apply can
// run as a tensor-core GEMM. Quality class: the weight side is the same fp16
// bank the scalar apply reads; the activation side gains one fp32->fp16
// rounding (rn). Accumulation is cuBLAS 32F - the apply path's summation
// order is already declared contractible (see mach1_rt_apply_kernel).
static __global__ void mach1_rt_u16_kernel(
        const float * __restrict__ u, half * __restrict__ u16, const int64_t k) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < k) {
        u16[i] = __float2half_rn(u[i]);
    }
}

// Skinny stock-op GEMM lane (GGML_MACH1_SKINNY_MM=1). The mach1 checkpoint
// keeps the GDN beta/alpha projections and the router gate as raw F32
// tensors, and at prefill their mul_mats fall through to the sgemm fallback:
// the prefill2k census reads 673 us/call for s0=2048x32 d=32x2048 against
// q4_K's 128 for the same shape (~40 ms per 2048-token ubatch over 60
// calls, plus the 256-row router at 327 us x40). Route them through the
// same per-shape-handle GemmEx path as the rt TC apply: mirror the weight
// to f16 once per process, convert the activations per call, 32F
// accumulate. The nt >= mach1_pp_min() floor (256; 128 under PPLOW, never
// below 64) keeps decode bitwise-untouched; quality rides the
// prefill-shaped KLD gate like the rest of the PP stack.

bool ggml_cuda_mach1_skinny_mm(ggml_backend_cuda_context & ctx,
                               const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    static const bool on = mach1_env_int("GGML_MACH1_SKINNY_MM", 0) != 0;
    if (!on) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    const int64_t k  = src0->ne[0];
    const int64_t m  = src0->ne[1];
    const int64_t nt = src1->ne[1];
    if (m > 256 || nt < mach1_pp_min() || src1->ne[0] != k) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    cudaStream_t stream = ctx.stream();

    // one-time f16 mirror of the weight, keyed by its device address (weights
    // are load-time-stable; a reload drops the whole process in every lane
    // this runs in)
    static std::map<const void *, const half *> wmap;
    static std::mutex wmu;
    const half * w16 = nullptr;
    {
        std::lock_guard<std::mutex> lk(wmu);
        auto it = wmap.find(src0->data);
        if (it != wmap.end()) {
            w16 = it->second;
        }
    }
    if (w16 == nullptr) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) {
            cudaGetLastError();
            return false;   // first sight mid-capture: fall through, mirror later
        }
        half * w = nullptr;
        if (cudaMalloc(&w, (size_t) k*m*sizeof(half)) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        const int64_t total = k*m;
        mach1_launch(mach1_rt_u16_kernel,
            ggml_cuda_kernel_launch_params(
                dim3((unsigned) ((total + 255)/256), 1, 1), dim3(256, 1, 1), 0, stream),
            (const float *) src0->data, w, total);
        std::lock_guard<std::mutex> lk(wmu);
        wmap[src0->data] = w;
        w16 = w;
    }

    mach1_pplow_mark("skinny mm", (int) nt);
    if (mach1_debug_on()) {
        static std::once_flag once;
        std::call_once(once, []() {
            fprintf(stderr, "mach1: PP skinny f32 mm -> f16 GemmEx ENGAGED\n");
        });
    }

    ggml_cuda_pool_alloc<half> b16_alloc(ctx.pool(), (size_t) k*nt);
    half * b16 = b16_alloc.get();
    mach1_timed(stream, "skinny_b16cvt", [&]() {
        const int64_t total = k*nt;
        mach1_launch(mach1_rt_u16_kernel,
            ggml_cuda_kernel_launch_params(
                dim3((unsigned) ((total + 255)/256), 1, 1), dim3(256, 1, 1), 0, stream),
            (const float *) src1->data, b16, total);
    });
    mach1_timed(stream, "skinny_gemm", [&]() {
        static std::map<int64_t, cublasHandle_t> hmap;
        static std::mutex hmu;
        cublasHandle_t cbh;
        {
            std::lock_guard<std::mutex> lk(hmu);
            auto & hh = hmap[((int64_t) m << 40) | (nt << 20) | k];
            if (hh == nullptr) {
                CUBLAS_CHECK(cublasCreate(&hh));
                static void * ws = nullptr;
                static const size_t wssz = 32u*1024u*1024u;
                if (ws == nullptr) {
                    CUDA_CHECK(cudaMalloc(&ws, wssz));
                }
                CUBLAS_CHECK(cublasSetWorkspace(hh, ws, wssz));
            }
            cbh = hh;
        }
        CUBLAS_CHECK(cublasSetStream(cbh, stream));
        const float onef = 1.0f, zerof = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(cbh, CUBLAS_OP_T, CUBLAS_OP_N,
            (int) m, (int) nt, (int) k,
            &onef,  w16, CUDA_R_16F, (int) k,
                    b16, CUDA_R_16F, (int) k,
            &zerof, dst->data, CUDA_R_32F, (int) m,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    });
    return true;
}

// Fused decode-path RT op (nt == 1): one kernel for u -> walk/apply -> out.
// Measured (H200, GGML_MACH1_TIME, 127 decode tokens): 250 RT ops per token,
// and the separate rt_u/rt_out stages average 12.6/14.5 us per call for ~2 us
// of single-block FWHT work — per-kernel latency, not compute. The fusion
// removes both stages from the launch chain:
//   stage 1: every block computes u = H(su .* x)/sqrt(n) redundantly in shared
//            memory — bit-identical to mach1_rt_u_kernel (same fwht_block, same
//            pinned final round, identical inputs), and the redundancy is free
//            because the u blocks ran on one SM while the rest idled anyway.
//   stage 2: the block's 16 output rows, exactly the rt_walk (or rt_apply
//            TC=1, when a persistent bank exists) body, reading u from shared.
//   stage 3: the LAST block to finish (self-resetting atomic counter, plain
//            threadfence pattern — graph-capture safe) runs the rt_out body
//            unchanged: same single-block FWHT, same order, same rounds.
// scr_v is still written globally so stage 3 sees every block's rows; numerics
// per stage are the stage kernels' own (walk arm bit-identical; apply arm the
// bank contract's contractible fp32 order).
template <int WG, bool BANKED>
__global__ void mach1_rt_fused_kernel(
        const uint16_t * __restrict__ trellis,
        const half     * __restrict__ tlut,     // pre-rounded fp16 (walk arm)
        const half     * __restrict__ bank,     // [m, n] (apply arm)
        const float    * __restrict__ su,
        const float    * __restrict__ sv,
        const float    * __restrict__ x,
        float          * __restrict__ scr_v,    // [m]
        float          * __restrict__ dst,      // [m]
        int            * __restrict__ counter,
        const int m, const int n) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = WG / warp_size;
    __shared__ float smem[8192];   // u [0, n) + slut [n, n+1024); out stage: H(v) [0, m)
    __shared__ float red[16][8];   // >= n_warps
    __shared__ int   is_last;

    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;

    float * shu = smem;
    for (int i = tid; i < n; i += WG) {
        shu[i] = su[i] * x[i];
    }
    if (!BANKED) {
        float * slut = smem + n;
        for (int i = tid; i < 1024; i += WG) {
            slut[i] = __half2float(tlut[i]);
        }
    }
    __syncthreads();
    mach1_fwht_block(shu, n, tid, WG);
    const float scn = __fsqrt_rn((float) n);
    for (int i = tid; i < n; i += WG) {
        shu[i] = __fdiv_rn(shu[i], scn);
    }
    __syncthreads();

    const int lane = tid % warp_size;
    const int wid  = tid / warp_size;

    if (BANKED) {
        const half * W = bank + (int64_t) tr*16*n;
        float acc[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            acc[i] = 0.0f;
        }
        for (int c0 = 2*tid; c0 < n; c0 += 2*WG) {
            const float2 uc = *(const float2 *)(shu + c0);
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const float2 wc = __half22float2(*(const half2 *)(W + (int64_t) i*n + c0));
                acc[i] += wc.x*uc.x + wc.y*uc.y;
            }
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float s = warp_reduce_sum<warp_size>(acc[i]);
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
            scr_v[tr*16 + tid] = sum;
        }
    } else {
        const float * slut = smem + n;
        const int  tiles_y = n / 16;
        const bool have    = tid < tiles_y;

        float partial[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            partial[i] = 0.0f;
        }
        if (have) {
            constexpr int words = 64;   // K = 4: 64 int16 words per 16x16 tile
            const int64_t tw = ((int64_t) tr*tiles_y + tid)*words;
            const float * ub = shu + tid*16;

            float uu[16];
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                uu[c] = ub[c];
            }
            for (int ri = 0; ri < 16; ++ri) {
                uint32_t w[5];
#pragma unroll
                for (int q = 0; q < 5; ++q) {
                    w[q] = trellis[tw + ((4*ri + q) & 63)];   // only q=4,ri=15 wraps
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
                        a0 = -a0;                             // exact in fp16 (sign bit)
                    }
                    acc += a0*uu[2*j] + a1*uu[2*j + 1];
                }
                partial[ri] = acc;
            }
        }
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
            scr_v[tr*16 + tid] = sum;
        }
    }

    __threadfence();
    if (tid == 0) {
        is_last = atomicAdd(counter, 1) == (int) gridDim.x - 1;
        if (is_last) {
            *counter = 0;   // self-resetting: ops are stream-serialized
        }
    }
    __syncthreads();
    if (!is_last) {
        return;
    }

    for (int i = tid; i < m; i += WG) {
        smem[i] = scr_v[i];
    }
    __syncthreads();
    mach1_fwht_block(smem, m, tid, WG);
    const float scm = __fsqrt_rn((float) m);
    for (int i = tid; i < m; i += WG) {
        dst[i] = __fdiv_rn(smem[i], scm) * sv[i];
    }
}

// --- folded weight banks (GGML_MACH1_FOLD=1, UNCERTIFIED tier) ---
//
// The rt chain y = sv .* H(W_hat H(su .* x)/sqrt(n))/sqrt(m) is one linear
// operator, so W_eff = D_sv (H_m/sqrt(m)) W_hat (H_n/sqrt(n)) D_su can be
// computed once at bank build and decode becomes a single fp16 GEMV - no
// per-token FWHT stages, no walk, no scratch, one kernel per op: the same
// per-token byte traffic as a bf16 checkpoint. NOT bit-exact to the certified
// chain: W_eff carries one extra fp16 rounding (the runtime fp32 dot is
// otherwise the apply path's contractible order), so this tier is opt-in and
// needs its own flip-gate before it can ship as a default.
//
// Fill: decode W_hat to the fp16 bank (exact), widen to an fp32 scratch, FWHT
// every row over n and fold su[c]/sqrt(n), FWHT every column over m and fold
// sv[r]/sqrt(m), round once to fp16 in place of the bank.

static __global__ void mach1_fold_widen_kernel(const half * __restrict__ w,
                                               float * __restrict__ f, const int64_t total) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < total) {
        f[i] = __half2float(w[i]);
    }
}

// one block per row: FWHT over the row's n columns, then scale by s[c]*rsq
template <int WG>
__global__ void mach1_fold_rows_kernel(float * __restrict__ f,
                                       const float * __restrict__ s,
                                       const int n, const float rsq) {
    __shared__ float sh[4096];
    const int64_t r  = blockIdx.x;
    const int    tid = threadIdx.x;
    for (int i = tid; i < n; i += WG) {
        sh[i] = f[r*n + i];
    }
    __syncthreads();
    mach1_fwht_block(sh, n, tid, WG);
    for (int i = tid; i < n; i += WG) {
        f[r*n + i] = sh[i]*rsq*s[i];
    }
}

// one block per column: FWHT over the column's m rows, then scale by s[r]*rsq
// and round the result to the fp16 bank
template <int WG>
__global__ void mach1_fold_cols_kernel(const float * __restrict__ f,
                                       const float * __restrict__ s,
                                       half * __restrict__ bank,
                                       const int m, const int n, const float rsq) {
    __shared__ float sh[8192];
    const int c   = blockIdx.x;
    const int tid = threadIdx.x;
    for (int i = tid; i < m; i += WG) {
        sh[i] = f[(int64_t) i*n + c];
    }
    __syncthreads();
    mach1_fwht_block(sh, m, tid, WG);
    for (int i = tid; i < m; i += WG) {
        bank[(int64_t) i*n + c] = __float2half_rn(sh[i]*rsq*s[i]);
    }
}

// runtime: y = W_eff x. One WARP per output row, 16-byte vectorized loads —
// the llama.cpp mmv shape. The first fold cut ran a 16-rows-per-128-thread
// block grid with half2 loads: MEASURED 13-15% of bandwidth (25 us real for a
// 16 MB row read, floor 3.3 us) and the whole fold arm landed at base speed.
static __global__ void mach1_rt_fold_mv_kernel(
        const half  * __restrict__ bank,     // [m, n] W_eff
        const float * __restrict__ x,
        float       * __restrict__ dst,
        const int m, const int n) {
    constexpr int warp_size = 32;
    const int lane = threadIdx.x % warp_size;
    const int r    = blockIdx.x*(blockDim.x/warp_size) + threadIdx.x/warp_size;
    if (r >= m) {
        return;
    }
    const half * W = bank + (int64_t) r*n;
    float acc = 0.0f;
    // 8 halves per lane-iteration; n is a multiple of 16, tail is n % 256
    const int n8 = n & ~(8*warp_size - 1);
    for (int c0 = 8*lane; c0 < n8; c0 += 8*warp_size) {
        const uint4  wv = *(const uint4 *)(W + c0);
        const float4 x0 = *(const float4 *)(x + c0);
        const float4 x1 = *(const float4 *)(x + c0 + 4);
        const half2 * wh = (const half2 *) &wv;
        float2 w01 = __half22float2(wh[0]);
        float2 w23 = __half22float2(wh[1]);
        acc += w01.x*x0.x + w01.y*x0.y + w23.x*x0.z + w23.y*x0.w;
        w01 = __half22float2(wh[2]);
        w23 = __half22float2(wh[3]);
        acc += w01.x*x1.x + w01.y*x1.y + w23.x*x1.z + w23.y*x1.w;
    }
    for (int c = n8 + 2*lane; c < n; c += 2*warp_size) {
        const float2 wc = __half22float2(*(const half2 *)(W + c));
        acc += wc.x*x[c] + wc.y*x[c + 1];
    }
    acc = warp_reduce_sum<warp_size>(acc);
    if (lane == 0) {
        dst[r] = acc;
    }
}

static bool mach1_fold_enabled() {
    static const bool on = mach1_env_int("GGML_MACH1_FOLD", 0) != 0;
    return on;
}

// --- z-bank tier (GGML_MACH1_ZBANK=1, UNCERTIFIED) ---
//
// The decoded weights are exact integer lattices (GGML_MACH1_DEBUG=2 on the
// shipped tables: spine 64 levels z = -32..31, experts 10 levels z = -5..4),
// so a bank can hold the integers themselves in ggml's own Q8_0 block layout
// and the GEMV becomes a stock ggml_cuda_mul_mat_vec_q call - llama.cpp's
// dp4a kernels, the same ones the bf16 baseline's MUL_MATs run on, at half
// the bytes of an fp16 bank. Measured ceiling for that path on these shapes:
// 223 tok/s decode (Q8_0) vs the bf16 baseline's 193 and our fp16 fold's 103.
//
// Applies to the SPINE only. The expert payload carries wave_gamma, a
// per-16x16-tile scale, and a 32-wide block spans two tiles - no standard
// block layout can express that without a second quantization. Spine reads
// are 2.8 GB of the 5.8 GB per token, so this is the majority of the
// addressable pool anyway.
//
// NOT bit-exact: the tables store fp16(z*u) and a Q8_0 bank serves
// z*fp16(u), ~4e-4 relative. Opt-in, own flip gate, like the fold tier.

// the lattice unit of a tlut, computed once per table on the host
static float mach1_lattice_unit(const void * tlut_data, size_t n_elem, cudaStream_t stream) {
    static std::mutex mtx;
    static std::unordered_map<const void *, float> cache;

    std::lock_guard<std::mutex> lock(mtx);
    const auto it = cache.find(tlut_data);
    if (it != cache.end()) {
        return it->second;
    }
    std::vector<uint16_t> host(n_elem);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), tlut_data, n_elem*sizeof(uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float amin = 0.0f;
    for (size_t i = 0; i < n_elem; ++i) {
        const float a = fabsf(GGML_FP16_TO_FP32((ggml_fp16_t) host[i]));
        if (a > 0.0f && (amin == 0.0f || a < amin)) {
            amin = a;
        }
    }
    // reject anything that is not an integer lattice: every value must sit
    // within a quarter step of a multiple of the unit
    float resid = 0.0f;
    for (size_t i = 0; i < n_elem; ++i) {
        const float q = GGML_FP16_TO_FP32((ggml_fp16_t) host[i])/amin;
        resid = std::max(resid, fabsf(q - roundf(q)));
    }
    const float unit = resid < 0.25f ? amin : 0.0f;
    cache.emplace(tlut_data, unit);
    if (mach1_debug_on()) {
        fprintf(stderr, "mach1: zbank unit %.9g (resid %.3g) for tlut %p\n", unit, resid, tlut_data);
    }
    return unit;
}

// fp16 bank -> ggml Q8_0 blocks, one thread per 32-weight block. The stored
// halves are fp16(z*unit), so v/unit recovers z to well within half a step
// (measured worst 0.014) and the round is exact.
static __global__ void mach1_zbank_pack_q8_0_kernel(
        const half * __restrict__ src,     // [m, n]
        block_q8_0 * __restrict__ dst,
        const float unit, const int n, const int64_t n_blocks) {
    const int64_t b = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (b >= n_blocks) {
        return;
    }
    const int64_t base = b*QK8_0;
    block_q8_0 * o = dst + b;
    o->d = __float2half(unit);
#pragma unroll
    for (int i = 0; i < QK8_0; ++i) {
        const float z = __half2float(src[base + i])/unit;
        const int   q = (int) roundf(z);
        o->qs[i] = (int8_t) min(127, max(-127, q));
    }
    GGML_UNUSED(n);
}

static bool mach1_zbank_enabled() {
    static const bool on = mach1_env_int("GGML_MACH1_ZBANK", 0) != 0;
    return on;
}

// GGML_MACH1_FOLDQ=1 (UNCERTIFIED, lossy): fold AND quantize. The fold tier
// removes the FWHT stages (1 kernel per op instead of 3-4) but leaves an fp16
// bank served by our own GEMV; the z-bank tier keeps exact integers but must
// keep the stages, which costs more launches than it saves (measured: zbank
// 61.3 vs fold 91.6 tok/s). The measured 223 tok/s Q8_0 ceiling was taken on
// a plain dense model with NO Hadamard stages, so reaching it requires BOTH:
// fold the codec into W_eff, then store W_eff as standard Q8_0 blocks and
// serve it with llama.cpp's dp4a mul_mat_vec_q.
//
// This is a second quantization (W_eff is not a lattice - the Hadamards
// destroy integer structure), ~0.4% of each block's max on top of the codec's
// own error. That is the failure mode this repo rejected for NE affine
// requant, so it is opt-in and MUST clear a flip gate before it is anything
// else. It exists to measure what the ceiling actually costs in quality.
static bool mach1_foldq_enabled() {
    static const bool on = mach1_env_int("GGML_MACH1_FOLDQ", 0) != 0;
    return on;
}

// fp32 W_eff -> ggml Q8_0 blocks, one thread per block, standard per-block
// scale (max|v|/127) since folded values are not a lattice
static __global__ void mach1_foldq_pack_kernel(
        const float * __restrict__ src,
        block_q8_0  * __restrict__ dst,
        const int64_t n_blocks) {
    const int64_t b = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (b >= n_blocks) {
        return;
    }
    const float * v = src + b*QK8_0;
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < QK8_0; ++i) {
        amax = fmaxf(amax, fabsf(v[i]));
    }
    const float d  = amax/127.0f;
    const float id = d > 0.0f ? 1.0f/d : 0.0f;
    block_q8_0 * o = dst + b;
    o->d = __float2half(d);
#pragma unroll
    for (int i = 0; i < QK8_0; ++i) {
        o->qs[i] = (int8_t) __float2int_rn(v[i]*id);
    }
}

// serve a Q8_0 bank through llama.cpp's own mul_mat_vec_q (dp4a), via
// fabricated tensor metadata - the ggml_cuda_mul_mat_id pattern
static void mach1_zbank_mmvq(ggml_backend_cuda_context & ctx, const void * W,
                             const float * y, float * out,
                             const int64_t cols, const int64_t rows) {
    const size_t ts  = ggml_type_size(GGML_TYPE_Q8_0);
    const size_t row = (size_t)(cols/QK8_0)*ts;

    // mmvq reads src0->buffer to decide whether to clear compute-buffer
    // padding; banks are fully packed, so present them as weights
    static ggml_backend_buffer wbuf = { {}, nullptr, nullptr, 0, GGML_BACKEND_BUFFER_USAGE_WEIGHTS };

    ggml_tensor src0 = {};
    src0.buffer = &wbuf;
    src0.type  = GGML_TYPE_Q8_0;
    src0.ne[0] = cols; src0.ne[1] = rows; src0.ne[2] = 1; src0.ne[3] = 1;
    src0.nb[0] = ts;   src0.nb[1] = row;  src0.nb[2] = row*rows; src0.nb[3] = src0.nb[2];
    src0.data  = (void *) W;

    ggml_tensor src1 = {};
    src1.type  = GGML_TYPE_F32;
    src1.ne[0] = cols; src1.ne[1] = 1;      src1.ne[2] = 1; src1.ne[3] = 1;
    src1.nb[0] = 4;    src1.nb[1] = 4*cols; src1.nb[2] = src1.nb[1]; src1.nb[3] = src1.nb[1];
    src1.data  = (void *) y;

    ggml_tensor dst = {};
    dst.type  = GGML_TYPE_F32;
    dst.ne[0] = rows;  dst.ne[1] = 1;      dst.ne[2] = 1; dst.ne[3] = 1;
    dst.nb[0] = 4;     dst.nb[1] = 4*rows; dst.nb[2] = dst.nb[1]; dst.nb[3] = dst.nb[1];
    dst.data  = out;
    dst.src[0] = &src0;
    dst.src[1] = &src1;

    ggml_cuda_mul_mat_vec_q(ctx, &src0, &src1, nullptr, &dst);
}

// hand a bank GEMV to llama.cpp's tuned f16 mmv (multi-warp rows, vectorized
// loads, Hopper paths) through fabricated tensor metadata — the same pattern
// ggml_cuda_mul_mat_id uses for its internal slices. W is row-major
// [rows, cols]; y is [cols]; out is [rows].
static void mach1_fold_mmvf(ggml_backend_cuda_context & ctx, const half * W,
                            const float * y, float * out,
                            const int64_t cols, const int64_t rows) {
    ggml_tensor src0 = {};
    src0.type  = GGML_TYPE_F16;
    src0.ne[0] = cols; src0.ne[1] = rows;       src0.ne[2] = 1; src0.ne[3] = 1;
    src0.nb[0] = 2;    src0.nb[1] = 2*cols;     src0.nb[2] = 2*cols*rows; src0.nb[3] = src0.nb[2];
    src0.data  = (void *) W;

    ggml_tensor src1 = {};
    src1.type  = GGML_TYPE_F32;
    src1.ne[0] = cols; src1.ne[1] = 1;          src1.ne[2] = 1; src1.ne[3] = 1;
    src1.nb[0] = 4;    src1.nb[1] = 4*cols;     src1.nb[2] = src1.nb[1]; src1.nb[3] = src1.nb[1];
    src1.data  = (void *) y;

    ggml_tensor dst = {};
    dst.type  = GGML_TYPE_F32;
    dst.ne[0] = rows;  dst.ne[1] = 1;           dst.ne[2] = 1; dst.ne[3] = 1;
    dst.nb[0] = 4;     dst.nb[1] = 4*rows;      dst.nb[2] = dst.nb[1]; dst.nb[3] = dst.nb[1];
    dst.data  = out;
    dst.op_params[0] = GGML_PREC_DEFAULT;
    dst.src[0] = &src0;
    dst.src[1] = &src1;

    ggml_cuda_mul_mat_vec_f(ctx, &src0, &src1, nullptr, &dst);
}

// exp fold: per storage group g, W_eff_g = D_sv_e (H_m/sqrt(m)) (gamma .* W_hat_g)
// (H_n/sqrt(n)) D_su_e with e the group's ORIGINAL expert id (su/sv/gamma are
// expert-indexed). inv[g] = e is scattered from remap once per tensor.

static __global__ void mach1_fold_inv_remap_kernel(const int32_t * __restrict__ remap,
                                                   int32_t * __restrict__ inv,
                                                   const int n_experts, const int n_kept) {
    const int e = blockIdx.x*blockDim.x + threadIdx.x;
    if (e >= n_experts) {
        return;
    }
    const uint32_t rm = (uint32_t) remap[e];
    const uint32_t g  = (rm & GGML_CUDA_MACH1_DEM_FLAG) != 0u
        ? (uint32_t) n_kept + (rm & ~GGML_CUDA_MACH1_DEM_FLAG) : rm;
    inv[g] = e;
}

// widen one group's exact fp16 slice to fp32 with the wave gamma folded in
static __global__ void mach1_fold_widen_gamma_kernel(
        const half * __restrict__ w,      // [m, n] group slice
        const half * __restrict__ gamma,  // [Mb + tiles_y] for expert e
        float      * __restrict__ f,
        const int m, const int n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) m*n) {
        return;
    }
    const int r = (int)(i / n);
    const int c = (int)(i % n);
    const int tiles_y = n / 16;
    const int Mb      = m / 16;
    const int tr      = r >> 4;
    const int tc      = c >> 4;
    const int wv = (tr + tc <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + tc)
                                            : Mb + tiles_y - 2 - (tr + tc);
    f[i] = __half2float(w[i]) * __half2float(gamma[wv]);
}

// fold_rows/fold_cols with fp16 scale vectors (exp su/sv are F16)
template <int WG>
__global__ void mach1_fold_rows_h_kernel(float * __restrict__ f,
                                         const half * __restrict__ s,
                                         const int n, const float rsq) {
    __shared__ float sh[4096];
    const int64_t r  = blockIdx.x;
    const int    tid = threadIdx.x;
    for (int i = tid; i < n; i += WG) {
        sh[i] = f[r*n + i];
    }
    __syncthreads();
    mach1_fwht_block(sh, n, tid, WG);
    for (int i = tid; i < n; i += WG) {
        f[r*n + i] = sh[i]*rsq*__half2float(s[i]);
    }
}

template <int WG>
__global__ void mach1_fold_cols_h_kernel(const float * __restrict__ f,
                                         const half * __restrict__ s,
                                         half * __restrict__ bank,
                                         const int m, const int n, const float rsq) {
    __shared__ float sh[8192];
    const int c   = blockIdx.x;
    const int tid = threadIdx.x;
    for (int i = tid; i < m; i += WG) {
        sh[i] = f[(int64_t) i*n + c];
    }
    __syncthreads();
    mach1_fwht_block(sh, m, tid, WG);
    for (int i = tid; i < m; i += WG) {
        bank[(int64_t) i*n + c] = __float2half_rn(sh[i]*rsq*__half2float(s[i]));
    }
}

// runtime: whole EXP op in one kernel - per used slot, y_s = W_eff_g x_s
// written straight to dst (the exp_out stage is folded into the bank). Same
// warp-per-row 16-byte-load shape as the rt fold GEMV.
static __global__ void mach1_exp_fold_mv_kernel(
        const half    * __restrict__ bank,     // [n_groups, m, n]
        const int32_t * __restrict__ remap,
        const int32_t * __restrict__ ids,
        const float   * __restrict__ x,
        float         * __restrict__ dst,
        const int m, const int n, const int xne1,
        const int n_used, const int n_kept, const int ids_s0, const int ids_s1) {
    constexpr int warp_size = 32;
    const int lane = threadIdx.x % warp_size;
    const int s    = blockIdx.z;
    const int r    = blockIdx.x*(blockDim.x/warp_size) + threadIdx.x/warp_size;
    if (r >= m) {
        return;
    }
    const uint32_t g = mach1_group_of(remap, ids, s, n_used, n_kept, ids_s0, ids_s1);
    const half  * W  = bank + ((int64_t) g*m + r)*n;
    const float * xs = x + (int64_t)(xne1 == 1 ? 0 : s*n);

    float acc = 0.0f;
    const int n8 = n & ~(8*warp_size - 1);
    for (int c0 = 8*lane; c0 < n8; c0 += 8*warp_size) {
        const uint4  wv = *(const uint4 *)(W + c0);
        const float4 x0 = *(const float4 *)(xs + c0);
        const float4 x1 = *(const float4 *)(xs + c0 + 4);
        const half2 * wh = (const half2 *) &wv;
        float2 w01 = __half22float2(wh[0]);
        float2 w23 = __half22float2(wh[1]);
        acc += w01.x*x0.x + w01.y*x0.y + w23.x*x0.z + w23.y*x0.w;
        w01 = __half22float2(wh[2]);
        w23 = __half22float2(wh[3]);
        acc += w01.x*x1.x + w01.y*x1.y + w23.x*x1.z + w23.y*x1.w;
    }
    for (int c = n8 + 2*lane; c < n; c += 2*warp_size) {
        const float2 wc = __half22float2(*(const half2 *)(W + c));
        acc += wc.x*xs[c] + wc.y*xs[c + 1];
    }
    acc = warp_reduce_sum<warp_size>(acc);
    if (lane == 0) {
        dst[(int64_t) s*m + r] = acc;
    }
}

// last-block gate for the fused kernels: one device int per GPU, zeroed at
// alloc and self-resetting per launch. Allocation is illegal mid-capture, so
// a capture-time miss falls back to the split kernels for that graph.
// per-(device, stream) completion counter for in-kernel out-stage folds. The
// last block self-resets it, so ops serialized on one stream can share it;
// concurrent streams get their own. nullptr under graph capture on first
// sight of a stream - the caller keeps the separate out launch then.
static int * mach1_stream_counter(int device, cudaStream_t stream) {
    static std::map<std::pair<int, cudaStream_t>, int *> counters;
    static std::mutex mtx;

    std::lock_guard<std::mutex> lock(mtx);
    const auto key = std::make_pair(device, stream);
    const auto it  = counters.find(key);
    if (it != counters.end()) {
        return it->second;
    }
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) {
        cudaGetLastError();
        return nullptr;
    }
    int * c = nullptr;
    if (cudaMalloc(&c, sizeof(int)) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    CUDA_CHECK(cudaMemsetAsync(c, 0, sizeof(int), stream));
    counters[key] = c;
    return c;
}

static int * mach1_fused_counter(int device, cudaStream_t stream) {
    static int * counters[GGML_CUDA_MAX_DEVICES] = {nullptr};
    static std::mutex mtx;

    std::lock_guard<std::mutex> lock(mtx);
    if (counters[device] != nullptr) {
        return counters[device];
    }
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) {
        cudaGetLastError();
        return nullptr;
    }
    int * c = nullptr;
    if (cudaMalloc(&c, sizeof(int)) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    CUDA_CHECK(cudaMemsetAsync(c, 0, sizeof(int), stream));
    counters[device] = c;
    return c;
}

// counter bank for the whole-layer expert FFN fusion. The megakernel's
// general layout spans (token, slot) pairs (see MACH1_EXP_MEGA_CNT_*, up to
// 32 pairs at ntok <= 4); the two-launch form keeps its original [0..7]
// per-slot + [8] sum layout in the same bank. A single int cannot decode
// per-slot completion (arrival order across slots is arbitrary), so the
// slot counters are an array; all forms share the bank because same-stream
// kernels serialize and every counter self-resets before its kernel exits.
// Same capture guard as mach1_stream_counter: nullptr on first sight of a
// capturing stream.
static int * mach1_ffn_counters(int device, cudaStream_t stream) {
    static std::map<std::pair<int, cudaStream_t>, int *> counters;
    static std::mutex mtx;

    std::lock_guard<std::mutex> lock(mtx);
    const auto key = std::make_pair(device, stream);
    const auto it  = counters.find(key);
    if (it != counters.end()) {
        return it->second;
    }
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) {
        cudaGetLastError();
        return nullptr;
    }
    // 1040 ints: the certified 128-pair layout ends at slot 388; the nt32
    // wide (PCAP=256) QONCE layout ends at 3*256+4 = 772. Extra ints are
    // idle margin, never read.
    int * c = nullptr;
    if (cudaMalloc(&c, 1040*sizeof(int)) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    CUDA_CHECK(cudaMemsetAsync(c, 0, 1040*sizeof(int), stream));
    counters[key] = c;
    return c;
}

// row scatter map for the vtiled fold (see v_tiled in src/models/mach1.cpp):
// rows [0, qk_off) keep their index, row qk_off + (k*r + v)*hd + d moves to
// qk_off + (v*K + k)*hd + d. One small device buffer per shape, cached for
// the process like the weight banks. Same capture guard as
// mach1_stream_counter: nullptr on first sight of a capturing stream.
static const int * mach1_vtiled_perm(int device, cudaStream_t stream,
                                     int m, int qk_off, int hd, int K, int r) {
    static std::map<std::array<int, 6>, int *> cache;
    static std::mutex mtx;

    std::lock_guard<std::mutex> lock(mtx);
    const std::array<int, 6> key = { device, m, qk_off, hd, K, r };
    const auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) {
        cudaGetLastError();
        return nullptr;
    }
    std::vector<int> p(m);
    for (int i = 0; i < qk_off; ++i) {
        p[i] = i;
    }
    for (int i = qk_off; i < m; ++i) {
        const int d = (i - qk_off) % hd;
        const int g = (i - qk_off) / hd;
        p[i] = qk_off + ((g % r)*K + g/r)*hd + d;
    }
    int * dp = nullptr;
    if (cudaMalloc(&dp, m*sizeof(int)) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    CUDA_CHECK(cudaMemcpyAsync(dp, p.data(), m*sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cache.emplace(key, dp);
    return dp;
}

template <int WG>
__global__ void mach1_rt_out_kernel(
        const float * __restrict__ sv,
        const float * __restrict__ scr_v,
        float       * __restrict__ dst,
        const int m) {
    __shared__ float sh[8192];

    const int t   = blockIdx.x;
    const int tid = threadIdx.x;

    for (int i = tid; i < m; i += WG) {
        sh[i] = scr_v[(int64_t) t*m + i];
    }
    __syncthreads();
    mach1_fwht_block(sh, m, tid, WG);
    const float sc = __fsqrt_rn((float) m);
    for (int i = tid; i < m; i += WG) {
        dst[(int64_t) t*m + i] = __fdiv_rn(sh[i], sc) * sv[i];
    }
}

void ggml_cuda_op_mach1_rt_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                              const void * x_data_override, const int * xperm,
                              const mach1_rt_tail * tail) {
    const ggml_tensor * trellis = dst->src[0];
    const ggml_tensor * su      = dst->src[1];
    const ggml_tensor * sv      = dst->src[2];
    const ggml_tensor * tlut    = dst->src[3];
    const ggml_tensor * x       = dst->src[4];

    GGML_ASSERT(trellis->type == GGML_TYPE_I16);
    GGML_ASSERT(su->type   == GGML_TYPE_F32);
    GGML_ASSERT(sv->type   == GGML_TYPE_F32);
    GGML_ASSERT(tlut->type == GGML_TYPE_F16);
    GGML_ASSERT(x->type    == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(trellis));
    GGML_ASSERT(ggml_is_contiguous(su));
    GGML_ASSERT(ggml_is_contiguous(sv));
    GGML_ASSERT(ggml_is_contiguous(tlut));
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(trellis->ne[0] == 64);   // K = 4 (the walk kernel hardcodes the rate)

    if (mach1_env_int("GGML_MACH1_DEBUG", 0) >= 2) {
        mach1_lattice_report("rt_tlut", tlut->data, ggml_nelements(tlut), true, ctx.stream());
    }

    const int n  = (int) su->ne[0];
    const int m  = (int) sv->ne[0];
    const int nt = (int)(x->ne[1]*x->ne[2]*x->ne[3]);
    GGML_ASSERT(n <= 4096);   // rt_u shared bound
    GGML_ASSERT(m <= 8192);   // rt_out shared bound

    // KS4 needs four logical partial rows per token. Co-allocate them with the
    // existing scr_v row so the core carries one 64-bit destination base just
    // like the zero-spill chainbench kernel. Flag-off retains the exact
    // allocation size and layout. A later format gate may leave the extra
    // rows unused, but they remain transient and exist only in the opt-in arm.
    static const bool rt_imma8_split4_requested =
        mach1_env_int("GGML_MACH1_RT_IMMA8_SPLITK", 0) != 0;
    // N512: the same work-centric split for the 64-CTA m2048 n512 shape (the
    // pure-stream probe caps 64-CTA grids at ~320 GB/s; 256 CTAs reach ~978)
    static const bool rt_imma8_split4_n512 =
        mach1_env_int("GGML_MACH1_RT_IMMA8_SPLITK_N512", 0) != 0;
    // the split-K storage admission stays EXACTLY nt == 16 unless NTLOW is on:
    // the nt32 rung measured its B24/B32 rows without it, so widening it must
    // not ride in on that arm's env
    const bool split4_nt = nt == 16 || (mach1_ntlow_on() && mach1_nt_ok(nt));
    const bool split4_storage =
        rt_imma8_split4_requested && mach1_rt_imma8_cpasync_enabled() &&
        m == 2048 && (n == 4096 || (n == 512 && rt_imma8_split4_n512)) &&
        split4_nt && xperm == nullptr;
    // Declared here (not at the dispatch below) so the scratch capacity can
    // see the admission; the rt_mma16 gate further down reuses these.
    static const bool rt_mma16_on = mach1_env_int("GGML_MACH1_RT_MMA16", 0) != 0;
    static const bool rt_mma16_m2_on = mach1_env_int("GGML_MACH1_RT_MMA16_M2", 0) != 0;
    const bool rt_mma16_shape = (n == 2048 && (m == 4096 || m == 8192)) ||
                                (rt_mma16_m2_on && m == 2048 &&
                                 (n == 512 || n == 2048 || n == 4096));
    const size_t scr_v_rows = split4_storage ? (size_t) 4*nt : (size_t) nt;
    ggml_cuda_pool_alloc<float> scr(
        ctx.pool(), (size_t) nt*n + scr_v_rows*m);
    float * scr_u = scr.get();
    float * scr_v = scr_u + (size_t) nt*n;

    cudaStream_t stream = ctx.stream();

    char shp[64];
    snprintf(shp, sizeof(shp), " m=%d n=%d nt=%d", m, n, nt);

    const auto decode_bank = [&](half * bdst) {
        mach1_timed(stream, std::string("rt_dense_decode") + shp, [&]() {
        mach1_launch(mach1_rt_dense_decode_kernel,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            (const uint16_t *) trellis->data, (const half *) tlut->data, bdst, m, n);
        });
    };

    // GGML_MACH1_FOLD=1 (UNCERTIFIED): serve decode from a W_eff bank - one
    // fp16 GEMV per op. Prefill keeps the exact paths; a fold-on process never
    // touches the plain banks (same cache key, different contents).
    if (mach1_fold_enabled() && nt == 1) {
        const bool foldq = mach1_foldq_enabled();
        // FOLDQ stores Q8_0 blocks in place of fp16 halves: 1.0625 B/weight.
        // The cache is half-typed, so ask for bytes/2 (block size is even).
        const size_t bank_elems = foldq
            ? (size_t) m*(n/QK8_0)*sizeof(block_q8_0)/2
            : (size_t) m*n;
        const auto fold_fill = [&](half * bdst) {
            ggml_cuda_pool_alloc<half>  hb(ctx.pool());
            half * dec = foldq ? hb.alloc((size_t) m*n) : bdst;
            decode_bank(dec);
            ggml_cuda_pool_alloc<float> f32(ctx.pool(), (size_t) m*n);
            const int64_t total = (int64_t) m*n;
            mach1_launch(mach1_fold_widen_kernel,
                ggml_cuda_kernel_launch_params(dim3((unsigned)((total + 255)/256), 1, 1),
                                               dim3(256, 1, 1), 0, stream),
                (const half *) dec, f32.get(), total);
            mach1_launch(mach1_fold_rows_kernel<256>,
                ggml_cuda_kernel_launch_params(dim3(m, 1, 1), dim3(256, 1, 1), 0, stream),
                f32.get(), (const float *) su->data, n, 1.0f/sqrtf((float) n));
            if (foldq) {
                // fold_cols writes fp16; run it into the scratch decode buffer,
                // then pack fp32 W_eff blocks. Reuse f32 as the W_eff store by
                // widening again after the column pass.
                mach1_launch(mach1_fold_cols_kernel<256>,
                    ggml_cuda_kernel_launch_params(dim3(n, 1, 1), dim3(256, 1, 1), 0, stream),
                    (const float *) f32.get(), (const float *) sv->data, dec, m, n, 1.0f/sqrtf((float) m));
                mach1_launch(mach1_fold_widen_kernel,
                    ggml_cuda_kernel_launch_params(dim3((unsigned)((total + 255)/256), 1, 1),
                                                   dim3(256, 1, 1), 0, stream),
                    (const half *) dec, f32.get(), total);
                const int64_t nb = (int64_t) m*(n/QK8_0);
                mach1_launch(mach1_foldq_pack_kernel,
                    ggml_cuda_kernel_launch_params(dim3((unsigned)((nb + 255)/256), 1, 1),
                                                   dim3(256, 1, 1), 0, stream),
                    (const float *) f32.get(), (block_q8_0 *) bdst, nb);
            } else {
                mach1_launch(mach1_fold_cols_kernel<256>,
                    ggml_cuda_kernel_launch_params(dim3(n, 1, 1), dim3(256, 1, 1), 0, stream),
                    (const float *) f32.get(), (const float *) sv->data, bdst, m, n, 1.0f/sqrtf((float) m));
            }
        };
        half * wb = mach1_bank_get(trellis->data, bank_elems, ctx.device, stream, fold_fill);
        if (wb != nullptr && foldq) {
            mach1_timed(stream, std::string("rt_foldq_mv") + shp, [&]() {
            mach1_zbank_mmvq(ctx, (const void *) wb, (const float *) x->data, (float *) dst->data, n, m);
            });
            return;
        }
        if (wb != nullptr) {
            mach1_timed(stream, std::string("rt_fold_mv") + shp, [&]() {
            static const bool own_mv = mach1_env_int("GGML_MACH1_FOLD_OWN_MV", 0) != 0;
            if (own_mv) {
                mach1_launch(mach1_rt_fold_mv_kernel,
                    ggml_cuda_kernel_launch_params(dim3((unsigned)((m + 7)/8), 1, 1), dim3(256, 1, 1), 0, stream),
                    (const half *) wb, (const float *) x->data, (float *) dst->data, m, n);
            } else {
                mach1_fold_mmvf(ctx, (const half *) wb, (const float *) x->data, (float *) dst->data, n, m);
            }
            });
            return;
        }
        // fold on but out of budget: this tensor stays on the exact walk path
    }

    // decode: one fused kernel per op. MEASURED REGRESSION (H200, paired, 3
    // rounds): 55.2 vs base 68.3 tok/s - the per-block redundant u-FWHT
    // multiplies the u work by grid size (512 blocks at m = 8192) and the
    // 32 KB static shared cuts walk occupancy. Default OFF; kept for A/B
    // (GGML_MACH1_FUSE_RT=1) since the shape is right for small-m ops.
    static const bool fuse_on = mach1_env_int("GGML_MACH1_FUSE_RT", 0) != 0 &&
                                mach1_env_int("GGML_MACH1_ABLATE", 0) == 0;
    if (fuse_on && nt == 1 && !mach1_fold_enabled()) {
        int * counter = mach1_fused_counter(ctx.device, stream);
        if (counter != nullptr) {
            half * fbank = mach1_bank_get(trellis->data, (size_t) m*n, ctx.device, stream, decode_bank);
            const ggml_cuda_kernel_launch_params fp_128 =
                ggml_cuda_kernel_launch_params(dim3(m/16, 1, 1), dim3(128, 1, 1), 0, stream);
            const ggml_cuda_kernel_launch_params fp_256 =
                ggml_cuda_kernel_launch_params(dim3(m/16, 1, 1), dim3(256, 1, 1), 0, stream);
            mach1_timed(stream, std::string("rt_fused") + shp, [&]() {
            if (fbank != nullptr) {
                if (n/16 <= 128) {
                    mach1_launch(mach1_rt_fused_kernel<128, true>, fp_128,
                        (const uint16_t *) trellis->data, (const half *) tlut->data, (const half *) fbank,
                        (const float *) su->data, (const float *) sv->data, (const float *) x->data,
                        scr_v, (float *) dst->data, counter, m, n);
                } else {
                    mach1_launch(mach1_rt_fused_kernel<256, true>, fp_256,
                        (const uint16_t *) trellis->data, (const half *) tlut->data, (const half *) fbank,
                        (const float *) su->data, (const float *) sv->data, (const float *) x->data,
                        scr_v, (float *) dst->data, counter, m, n);
                }
            } else {
                if (n/16 <= 128) {
                    mach1_launch(mach1_rt_fused_kernel<128, false>, fp_128,
                        (const uint16_t *) trellis->data, (const half *) tlut->data, (const half *) nullptr,
                        (const float *) su->data, (const float *) sv->data, (const float *) x->data,
                        scr_v, (float *) dst->data, counter, m, n);
                } else {
                    mach1_launch(mach1_rt_fused_kernel<256, false>, fp_256,
                        (const uint16_t *) trellis->data, (const half *) tlut->data, (const half *) nullptr,
                        (const float *) su->data, (const float *) sv->data, (const float *) x->data,
                        scr_v, (float *) dst->data, counter, m, n);
                }
            }
            });
            return;
        }
    }

    // GGML_MACH1_COOP=1: one cooperative launch replaces rt_u + walk + rt_out
    if (mach1_env_int("GGML_MACH1_COOP", 0) != 0 && nt == 1 &&
        n % 16 == 0 && n/16 <= 256 && mach1_env_int("GGML_MACH1_ABLATE", 0) == 0 &&
        !mach1_fold_enabled() && !mach1_zbank_enabled()) {
        // CT must cover tiles_y: WG 512 (CT 128) up to n = 2048, WG 1024
        // (CT 256) for n = 4096. The wide form keeps RG at 4, so the warp
        // butterfly and the per-row-group combine are the same ones the
        // split path runs.
        const bool wide = n/16 > 128;
        const int  wg   = wide ? 1024 : 512;
        int blocks_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, wide ? (const void *) mach1_rt_coop_kernel<1024>
                                 : (const void *) mach1_rt_coop_kernel<512>, wg, 0));
        const int n_sm   = ggml_cuda_info().devices[ctx.device].nsm;
        const int blocks = m/16;
        if (mach1_debug_on()) {
            static std::once_flag once;
            std::call_once(once, [&]() {
                fprintf(stderr, "mach1-coop: wg=%d blocks_per_sm=%d n_sm=%d\n",
                        wg, blocks_per_sm, n_sm);
            });
        }
        if (blocks_per_sm > 0 && blocks <= blocks_per_sm*n_sm) {
            const uint16_t * tre_d = (const uint16_t *) trellis->data;
            const half     * tl_d  = (const half     *) tlut->data;
            const float    * su_d  = (const float    *) su->data;
            const float    * sv_d  = (const float    *) sv->data;
            const float    * x_d   = (const float    *) x->data;
            float          * dst_d = (float          *) dst->data;
            int mm = m, nn = n;
            void * args[] = { &tre_d, &tl_d, &su_d, &sv_d, &x_d,
                              &scr_u, &scr_v, &dst_d, &mm, &nn };
            mach1_timed(stream, std::string("rt_coop") + shp, [&]() {
            CUDA_CHECK(cudaLaunchCooperativeKernel(
                wide ? (const void *) mach1_rt_coop_kernel<1024>
                     : (const void *) mach1_rt_coop_kernel<512>,
                dim3(blocks, 1, 1), dim3(wg, 1, 1), args, 0, stream));
            });
            return;
        }
    }

    // GGML_MACH1_UFUSE_MAXB: fold the u stage into the walk when the walk's
    // grid is small enough that redoing it per block is hidden. 0 disables.
    // MEASURED (H200, paired, 3 rounds, token-identical): 80.99 at 256 blocks,
    // 79.63 at 64, 78.63 off. The redundancy is cheaper than the launch it
    // replaces far further up than expected - only m = 8192 (512 blocks) is
    // still excluded.
    static const int ufuse_maxb = mach1_env_int("GGML_MACH1_UFUSE_MAXB", 256);
    static const bool wvec0     = mach1_env_int("GGML_MACH1_WALK_VEC", 1) != 0;
    static const int  walk_rg0  = mach1_env_int("GGML_MACH1_WALK_RG", 4);
    static const int  probe00   = mach1_env_int("GGML_MACH1_WALK_PROBE", 0);
    static const bool lut1k00   = mach1_env_int("GGML_MACH1_LUT1K", 0) != 0;
    const bool ufuse = ufuse_maxb > 0 && m/16 <= ufuse_maxb && nt == 1 &&
                       wvec0 && walk_rg0 == 4 && probe00 == 0 && !lut1k00 &&
                       n % 16 == 0 && n/16 <= 256 &&
                       !mach1_ablate(MACH1_ABLATE_RT_U) &&
                       !mach1_ablate(MACH1_ABLATE_RT_WALK) &&
                       mach1_env_int("GGML_MACH1_COOP", 0) == 0;

    // GGML_MACH1_UFUSE_TT: the nt >= 2 form of the same fold - the TT walk
    // computes FWHT(su .* x) per chunk token into shu and the u launch is
    // skipped. Mirrors every condition on the path to the TT walk branch
    // (the ufuse precedent); if a bank path takes the op anyway, the u stage
    // is issued late from there, so scr_u is never consumed unwritten.
    static const int  uf_tt_env   = mach1_env_int("GGML_MACH1_UFUSE_TT", 0);
    static const int  walk_tt00   = mach1_env_int("GGML_MACH1_WALK_TT", 16);
    static const int  walk_ttw00  = mach1_env_int("GGML_MACH1_WALK_TTW", 0);
    static const bool walk_ttof00 = mach1_env_int("GGML_MACH1_WALK_TT_OF", 0) != 0;
    const bool uf_tt = uf_tt_env != 0 && nt >= 2 && walk_tt00 >= 2 && nt <= walk_tt00 &&
                       ufuse_maxb > 0 && m/16 <= ufuse_maxb &&
                       wvec0 && walk_rg0 == 4 && probe00 == 0 && !lut1k00 &&
                       n % 16 == 0 && n/16 <= 256 &&
                       walk_ttw00 == 0 && !walk_ttof00 && xperm == nullptr &&
                       !mach1_ablate(MACH1_ABLATE_RT_U) &&
                       !mach1_ablate(MACH1_ABLATE_RT_WALK) &&
                       mach1_env_int("GGML_MACH1_COOP", 0) == 0;

    // XPERM callers pre-verified this exact gate: the fold rides only the
    // in-block butterfly u prologue (n/16 > 128 also keeps tcw off below)
    GGML_ASSERT(xperm == nullptr || (ufuse && n/16 > 128));

    // Compressed-domain tensor-core path for selected high-cost p16 RT shapes.
    // Resolve this before the u stage so its final store can write the packed
    // half rows directly into existing scratch. Keep it opt-in until the KLD
    // and serving gates pass; the half activation and MMA association are not
    // bit-exact.
    const auto & rt_mma_dev = ggml_cuda_info().devices[ctx.device];
    // nt32/ntlow: an admitted batch rides the audited 16-token spine as
    // {16,8,4,2}-token chunks (u records and scr_v rows are token-major with
    // n/m strides, so a chunk is a plain pointer offset; the B32 census killed
    // the TT-walk route - its per-chunk re-decode loses to even the dense
    // round trip). The nt == 16 admission and launch bytes are unchanged.
    const bool rt_mma16_nt = mach1_nt_ok(nt);
    const bool rt_mma16 = rt_mma16_on && rt_mma16_nt && rt_mma16_shape &&
                          xperm == nullptr && !uf_tt &&
                          ampere_mma_available(rt_mma_dev.cc) && rt_mma_dev.warp_size == 32 &&
                          !mach1_ablate(MACH1_ABLATE_RT_WALK);

    // GGML_MACH1_TC_FWHT (decode only, default ON): the FWHT stages run as the
    // tensor-core Kronecker two-dot. NOT bit-exact (fp16 operands) - certified
    // by the KL gate, so it composes with the bit-exact folds.
    static const int tc_nt = mach1_env_int("GGML_MACH1_TC_NT", 16);
    const half * tc_htab = nt <= tc_nt ? mach1_tc_fwht_tab(ctx.device, stream) : nullptr;

    // Default-off format-aware rung. It reuses the existing ZDP lattice table
    // and scr_u allocation, but changes the transform/spine handoff to q8 K16
    // records consumed by signed-int8 tensor cores. A non-ladder table or a
    // missing TC transform keeps the already-certified fp16 path wholesale.
    static const bool rt_imma8_on = mach1_env_int("GGML_MACH1_RT_IMMA8", 0) != 0;
    mach1_rt_ztab rt_imma8_zt = {nullptr, 0.0f};
    if (rt_imma8_on && rt_mma16 && tc_htab != nullptr && mach1_tc_fwht_dim(n)) {
        rt_imma8_zt = mach1_rt_z_table(tlut->data, stream);
    }
    const bool rt_imma8 = rt_imma8_zt.ztab != nullptr;
    const bool rt_imma8_split4 =
        split4_storage && rt_imma8;

    const auto launch_u_stage = [&]() {
        if (tc_htab != nullptr && mach1_tc_fwht_dim(n)) {
            mach1_timed(stream, std::string("rt_u_tc") + shp, [&]() {
            mach1_rt_u_tc(tc_htab, (const float *) su->data, (const float *) x->data,
                          scr_u, n, stream, nt, rt_mma16 && !rt_imma8,
                          rt_imma8, rt_imma8_zt.zstep);
            });
        } else
        mach1_timed(stream, std::string("rt_u") + shp, [&]() {
        if (mach1_fwht_wg() == 1024) {
            if (rt_mma16) {
                mach1_launch(mach1_rt_u_kernel<1024, true>,
                    ggml_cuda_kernel_launch_params(dim3(nt, 1, 1), dim3(1024, 1, 1), 0, stream),
                    (const float *) su->data, (const float *) x->data, scr_u, n);
            } else {
                mach1_launch(mach1_rt_u_kernel<1024>,
                    ggml_cuda_kernel_launch_params(dim3(nt, 1, 1), dim3(1024, 1, 1), 0, stream),
                    (const float *) su->data, (const float *) x->data, scr_u, n);
            }
        } else if (mach1_fwht_wg512()) {
            if (rt_mma16) {
                mach1_launch(mach1_rt_u_kernel<512, true>,
                    ggml_cuda_kernel_launch_params(dim3(nt, 1, 1), dim3(512, 1, 1), 0, stream),
                    (const float *) su->data, (const float *) x->data, scr_u, n);
            } else {
                mach1_launch(mach1_rt_u_kernel<512>,
                    ggml_cuda_kernel_launch_params(dim3(nt, 1, 1), dim3(512, 1, 1), 0, stream),
                    (const float *) su->data, (const float *) x->data, scr_u, n);
            }
        } else {
            if (rt_mma16) {
                mach1_launch(mach1_rt_u_kernel<256, true>,
                    ggml_cuda_kernel_launch_params(dim3(nt, 1, 1), dim3(256, 1, 1), 0, stream),
                    (const float *) su->data, (const float *) x->data, scr_u, n);
            } else {
                mach1_launch(mach1_rt_u_kernel<256>,
                    ggml_cuda_kernel_launch_params(dim3(nt, 1, 1), dim3(256, 1, 1), 0, stream),
                    (const float *) su->data, (const float *) x->data, scr_u, n);
            }
        }
        });
    };
    if (!ufuse && !uf_tt && !mach1_ablate(MACH1_ABLATE_RT_U)) {
        launch_u_stage();
        // GGML_MACH1_LAUNCH_PROBE: re-issue the u stage. It is a pure function
        // of x and su into scr_u, so the repeat is idempotent and the wall
        // delta prices ONE extra small launch per spine op - the number that
        // decides whether a launch-fusion rung is worth building.
        if (mach1_env_int("GGML_MACH1_LAUNCH_PROBE", 0) != 0) {
            launch_u_stage();
        }
    }

    bool rt_mma16_served = false;
    if (rt_mma16) {
        const char * rt_key = rt_imma8_split4 ? "rt_imma8_split4"
                            : (rt_imma8 ? "rt_imma8" : "rt_mma16");
        mach1_timed(stream, std::string(rt_key) + shp, [&]() {
            if (rt_imma8) {
                mach1_rt_spine_imma8_p16(
                    (const uint16_t *) trellis->data, rt_imma8_zt.ztab,
                    scr_u, scr_v, m, n, stream, rt_imma8_split4, nt);
            } else {
                mach1_rt_spine_mma_p16(
                    (const uint16_t *) trellis->data, (const half *) tlut->data,
                    (const half *) scr_u, scr_v, m, n, stream, nt);
            }
        });
        rt_mma16_served = true;
        if (mach1_debug_on()) {
            static std::once_flag once;
            std::call_once(once, [&]() {
                fprintf(stderr, "mach1: zero-VRAM compressed %s ENGAGED\n",
                        rt_imma8 ? "rt_imma8" : "rt_mma16");
            });
            if (nt != 16) {
                static std::once_flag once_nt;
                std::call_once(once_nt, [&]() {
                    fprintf(stderr, "mach1: chunked spine ENGAGED (nt=%d split4=%d)\n",
                            nt, rt_imma8_split4 ? 1 : 0);
                });
            }
        }
        if (mach1_debug_on() && nt < 16) {
            static std::once_flag once_lo;
            std::call_once(once_lo, [&]() {
                fprintf(stderr, "mach1: ntlo rt spine nt=%d ENGAGED\n", nt);
            });
        }
    }

    // z-bank: the walk becomes a stock Q8_0 mul_mat_vec_q (dp4a) over the
    // integer lattice. Bank bytes are passed to the (half-typed) cache as
    // bytes/2; sizeof(block_q8_0) is even, so this is exact.
    bool zbank_served = false;
    if (!rt_mma16_served && mach1_zbank_enabled() && nt == 1 && n % QK8_0 == 0) {
        const float unit = mach1_lattice_unit(tlut->data, ggml_nelements(tlut), stream);
        if (unit > 0.0f) {
            const size_t bank_bytes = (size_t) m*(n/QK8_0)*sizeof(block_q8_0);
            const auto zfill = [&](half * bdst) {
                ggml_cuda_pool_alloc<half> tmp(ctx.pool(), (size_t) m*n);
                decode_bank(tmp.get());
                const int64_t nb = (int64_t) m*(n/QK8_0);
                mach1_launch(mach1_zbank_pack_q8_0_kernel,
                    ggml_cuda_kernel_launch_params(dim3((unsigned)((nb + 255)/256), 1, 1),
                                                   dim3(256, 1, 1), 0, stream),
                    (const half *) tmp.get(), (block_q8_0 *) bdst, unit, n, nb);
            };
            half * zb = mach1_bank_get(trellis->data, bank_bytes/2, ctx.device, stream, zfill);
            if (zb != nullptr) {
                // ufuse deferred the u stage, but the mmvq consumes scr_u NOW -
                // the late launch at the bank handoff below is stream-ordered
                // after it, so issue the stage here
                if (ufuse) {
                    launch_u_stage();
                }
                mach1_timed(stream, std::string("rt_zbank_mv") + shp, [&]() {
                mach1_zbank_mmvq(ctx, (const void *) zb, scr_u, scr_v, n, m);
                });
                zbank_served = true;
            }
        }
    }

    static const int dense_min_tok = mach1_env_int("GGML_MACH1_DENSE_MIN_TOK", 4);
    // WALK_TT owns nt in [2, walk_tt]: the TT walk reads the trellis once per
    // chunk with no [m, n] fp16 materialization, so the dense decode-and-apply
    // round trip only engages past its ceiling.
    static const int walk_tt_max = mach1_env_int("GGML_MACH1_WALK_TT", 16);
    const int eff_dense_min = walk_tt_max >= 2 ? std::max(dense_min_tok, walk_tt_max + 1)
                                               : dense_min_tok;
    ggml_cuda_pool_alloc<half> bank_alloc(ctx.pool());

    // The rt bank is keyed purely by the weight tensor ([m, n] - no routing in
    // it), so once decoded it is reusable for every later ubatch. It is a
    // PREFILL lever: it replaces a transient decode this path was going to do
    // anyway (same kernel, same bytes), which is why it costs nothing in
    // numerics. Below eff_dense_min no [m, n] image is materialized at all -
    // the TT walk reads the compressed stream, ~638 MB against ~3 GB of fp16
    // per step - so banking there both loses and would change the certified
    // decode path. Fold mode owns the same cache key with folded contents.
    // GGML_MACH1_BANK_NT lowers the floor for an A/B of that loss; 0 keeps it
    // at eff_dense_min, so a banked process leaves decode textually alone.
    // The lowered floor is a WINDOW [BANK_NT, BANK_NT_MAX] so a single width
    // can be admitted without handing the bank every width below the prefill
    // floor. MEASURED: no width wins - see the BANK_NT certification rung.
    static const int bank_nt_min = mach1_env_int("GGML_MACH1_BANK_NT", 0);
    static const int bank_nt_max = mach1_env_int("GGML_MACH1_BANK_NT_MAX", 0);
    const bool bank_nt_ok = bank_nt_min > 0 && nt >= bank_nt_min &&
                            (bank_nt_max <= 0 || nt <= bank_nt_max);
    half * bank = (rt_mma16_served || mach1_fold_enabled() || zbank_served ||
                   (!bank_nt_ok && nt < eff_dense_min)) ? nullptr
        : mach1_bank_get(trellis->data, (size_t) m*n, ctx.device, stream, decode_bank);

    if (!rt_mma16_served && !zbank_served && bank == nullptr && nt >= eff_dense_min) {
        bank = bank_alloc.alloc((size_t) m*n);
        decode_bank(bank);
    }

    // a u fold predicted a walk but a bank path owns the op - issue the
    // deferred u stage now so scr_u is materialized before the apply. BOTH
    // folds need this: ufuse (nt == 1) was reachable only through BANK_NT,
    // and without it the apply consumed an unwritten scr_u. The zbank path
    // consumes scr_u earlier and issues its own launch above.
    if ((uf_tt || ufuse) && bank != nullptr) {
        launch_u_stage();
    }

    bool ofused = false;   // rt_out folded into the walk's last block
    if (rt_mma16_served) {
        // scr_v was produced directly from the compressed trellis above.
    } else if (zbank_served) {
        // walk already done by the stock mmvq path above
    } else if (bank != nullptr) {
        static const bool rt_tc4 = mach1_env_int("GGML_MACH1_RT_TC", 8) == 4;
        // PP lane: the bank is a plain fp16 [m, n] matrix, so past a token
        // floor the apply is a real GEMM - hand it to the tensor cores
        // (u rounded to fp16, cuBLAS 32F accumulate) instead of the scalar
        // FMA kernel. Prefill only; decode shapes keep the certified path.
        static const bool apply_tc = mach1_env_int("GGML_MACH1_RT_APPLY_TC", 0) != 0;
        if (apply_tc && nt >= mach1_pp_min() && !mach1_ablate(MACH1_ABLATE_RT_WALK)) {
            mach1_pplow_mark("rt apply tc", nt);
            if (mach1_debug_on()) {
                static std::once_flag once;
                std::call_once(once, []() {
                    fprintf(stderr, "mach1: PP rt apply tensor-core GEMM ENGAGED (fp16 u, 32F acc)\n");
                });
            }
            ggml_cuda_pool_alloc<half> u16_alloc(ctx.pool(), (size_t) nt*n);
            half * u16 = u16_alloc.get();
            mach1_timed(stream, std::string("rt_u16cvt") + shp, [&]() {
                const int64_t k = (int64_t) nt*n;
                mach1_launch(mach1_rt_u16_kernel,
                    ggml_cuda_kernel_launch_params(
                        dim3((unsigned) ((k + 255)/256), 1, 1), dim3(256, 1, 1), 0, stream),
                    (const float *) scr_u, u16, k);
            });
            // residency probe (GGML_MACH1_RT_TC_COPY=1): GEMM from a fresh
            // pool copy of the bank - if the m=8192 anomaly is the persistent
            // bank allocation's residency, the copy restores peak TFLOPS
            static const bool tc_copy = mach1_env_int("GGML_MACH1_RT_TC_COPY", 0) != 0;
            ggml_cuda_pool_alloc<half> bcopy_alloc(ctx.pool());
            const half * gbank = bank;
            if (tc_copy) {
                half * bc = bcopy_alloc.alloc((size_t) m*n);
                CUDA_CHECK(cudaMemcpyAsync(bc, bank, (size_t) m*n*sizeof(half),
                                           cudaMemcpyDeviceToDevice, stream));
                gbank = bc;
            }
            mach1_timed(stream, std::string("rt_apply_tc") + shp, [&]() {
                // one handle per (m, n) shape: production alternates GEMM
                // shapes call-to-call and cuBLAS re-runs its heuristic (a
                // ~2 ms cudaGetDeviceProperties each, 1296 calls in the nsys
                // API sum) when the handle's last shape differs - the
                // same-shape chainbench loop runs at 235 TFLOPS because its
                // handle stays cached
                static std::once_flag hb_once;
                static std::map<std::pair<int,int>, cublasHandle_t> hmap;
                static std::mutex hmu;
                cublasHandle_t cbh;
                {
                    std::lock_guard<std::mutex> lk(hmu);
                    auto & hh = hmap[{m, n}];
                    if (hh == nullptr) {
                        CUBLAS_CHECK(cublasCreate(&hh));
                        // explicit workspace so the split-K path never
                        // allocates per call (one 32 MB scratch shared by
                        // all handles - every call runs on the same stream)
                        static void * ws = nullptr;
                        static const size_t wssz = 32u*1024u*1024u;
                        if (ws == nullptr) {
                            CUDA_CHECK(cudaMalloc(&ws, wssz));
                        }
                        CUBLAS_CHECK(cublasSetWorkspace(hh, ws, wssz));
                    }
                    cbh = hh;
                }
                (void) hb_once;
                CUBLAS_CHECK(cublasSetStream(cbh, stream));
                const float onef = 1.0f, zerof = 0.0f;
                // single call: the isolated GemmEx probe (`gemm-probe-s1`)
                // reads 235.6 TFLOPS at m=8192 and m-chunking is WORSE in
                // isolation (94.1 vs 72.9 us) - the census's 730 us window
                // is concurrency absorbed into the interval, not the GEMM
                CUBLAS_CHECK(cublasGemmEx(cbh, CUBLAS_OP_T, CUBLAS_OP_N,
                    m, nt, n,
                    &onef,  gbank, CUDA_R_16F, n,
                            u16,  CUDA_R_16F, n,
                    &zerof, scr_v, CUDA_R_32F, m,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            });
        } else {
        if (!mach1_ablate(MACH1_ABLATE_RT_WALK))
    mach1_timed(stream, std::string("rt_apply") + shp, [&]() {
        if (nt == 1) {
            // Decode. TC>1 exists to amortize bank traffic across a chunk of
            // tokens, but with one token every extra slot clamps to the same
            // token (tt = min(t0+t, nt-1)): TC-fold redundant compute plus
            // TC*16 accumulator registers per thread, which guts occupancy.
            // A single-token instantiation is a plain bandwidth-bound GEMV.
            mach1_launch(mach1_rt_apply_kernel<1>,
                ggml_cuda_kernel_launch_params(
                    dim3(m/16, 1, 1), dim3(128, 1, 1), 0, stream),
                (const half *) bank, (const float *) scr_u, scr_v, m, n, nt);
        } else if (rt_tc4) {
            mach1_launch(mach1_rt_apply_kernel<4>,
                ggml_cuda_kernel_launch_params(
                    dim3(m/16, 1, (nt + 3)/4), dim3(128, 1, 1), 0, stream),
                (const half *) bank, (const float *) scr_u, scr_v, m, n, nt);
        } else {
            mach1_launch(mach1_rt_apply_kernel<8>,
                ggml_cuda_kernel_launch_params(
                    dim3(m/16, 1, (nt + 7)/8), dim3(128, 1, 1), 0, stream),
                (const half *) bank, (const float *) scr_u, scr_v, m, n, nt);
        }
        });
        }
    } else {
        if (!mach1_ablate(MACH1_ABLATE_RT_WALK))
    mach1_timed(stream, std::string("rt_walk") + shp, [&]() {
        // row-split walk: 4x threads, 4x fewer accumulators, bit-identical
        // (GGML_MACH1_WALK_RG=1 restores the one-thread-per-column-tile form)
        static const int walk_rg = mach1_env_int("GGML_MACH1_WALK_RG", 4);
        static const int probe0 = mach1_env_int("GGML_MACH1_WALK_PROBE", 0);
        static const bool lut1k0 = mach1_env_int("GGML_MACH1_LUT1K", 0) != 0;
        const int tiles_y0 = n/16;

        // vector trellis load + u staged in shared: -40% on the isolated
        // shape census (3.845 -> 2.304 ms/token). See the kernel comment.
        static const bool wvec = mach1_env_int("GGML_MACH1_WALK_VEC", 1) != 0;
        if (wvec && walk_rg == 4 && probe0 == 0 && !lut1k0 && n % 16 == 0 &&
            tiles_y0 <= 256) {
            const int wg = tiles_y0 <= 128 ? 512 : 1024;
            // OFUSE: fold the rt_out stage into the walk's last block
            static const bool ofuse_on = mach1_env_int("GGML_MACH1_OFUSE", 0) != 0;
            int * ocnt = nullptr;
            if (ofuse_on && nt == 1 && !mach1_ablate(MACH1_ABLATE_RT_OUT)) {
                ocnt = mach1_stream_counter(ctx.device, stream);
            }
            // TCF covers whichever folded stages are active; a shape outside
            // the census set keeps the butterfly kernel wholesale. wg 512
            // only: with 1024 threads the walk+mma register set exceeds the
            // 64-reg/thread block budget ("too many resources requested for
            // launch"), and the only wg 1024 rt walks at decode are the
            // shexp-shaped n = 4096 ops the shexp fusion normally swallows.
            const bool tcw = tc_htab != nullptr && mach1_tc_in_walk() && wg == 512 &&
                             (ufuse || ocnt != nullptr) &&
                             (!ufuse || mach1_tc_fwht_dim(n)) &&
                             (ocnt == nullptr || mach1_tc_fwht_dim(m));
            size_t smem = ocnt != nullptr ? (size_t) m*sizeof(float) : 0;
            if (tcw) {
                smem = std::max(ufuse ? mach1_tc_ws_bytes(n) : (size_t) 0,
                                ocnt != nullptr ? mach1_tc_ws_bytes(m) : (size_t) 0);
            }
            const ggml_cuda_kernel_launch_params vp =
                ggml_cuda_kernel_launch_params(dim3(m/16, 1, nt), dim3(wg, 1, 1), smem, stream);
            const uint16_t * tre_v = (const uint16_t *) trellis->data;
            const half     * tl_v  = (const half     *) tlut->data;
            const float    * su_v  = (const float    *) su->data;
            const float    * x_v   = x_data_override != nullptr ? (const float *) x_data_override
                                                                : (const float *) x->data;
            const float    * sv_v  = (const float    *) sv->data;
            float          * dst_v = (float *) dst->data;
            // integer-lattice walk: engages on the serving branches (tcw and
            // plain); the ofuse forms keep the fp gather
            const mach1_rt_ztab zt = mach1_zdp_on() ? mach1_rt_z_table(tlut->data, stream)
                                                    : mach1_rt_ztab{nullptr, 0.0f};
            const bool zdp = zt.ztab != nullptr;
            // TT walk (GGML_MACH1_WALK_TT = max admitted nt, 0 off): at
            // nt >= 2 decode each row's windows ONCE and MAC the chunk's
            // tokens against the register cache - the per-token decode
            // redundancy of the grid.z form was 4.08 ms/step at nt=2 (r258).
            // Values are per-token identical to the grid.z walk.
            static const int walk_tt = mach1_env_int("GGML_MACH1_WALK_TT", 16);
            if (walk_tt >= 2 && nt >= 2 && nt <= walk_tt && xperm == nullptr) {
                const int ttc = (wg == 512 && nt >= 3) ? 4 : 2;
                // WALK_TT_OF: fold the rt_out stage into the last block - the
                // out transform is on the data-dependency chain, so this
                // shortens the per-op serial path. Single-chunk grids only
                // (the completion counter spans the whole grid).
                static const bool walk_tt_of = mach1_env_int("GGML_MACH1_WALK_TT_OF", 0) != 0;
                int * tcnt = (walk_tt_of && nt <= ttc && !mach1_ablate(MACH1_ABLATE_RT_OUT))
                             ? mach1_stream_counter(ctx.device, stream) : nullptr;
                const size_t tsmem = tcnt != nullptr ? (size_t) m*sizeof(float) : 0;
                // WALK_TTW: the wide form for latency-bound small-m shapes -
                // WG 1024 / RG 8 puts twice the threads on the same tiles
                // (the r275 theory: these grids are 32-128 blocks on 142 SMs)
                static const int walk_ttw = mach1_env_int("GGML_MACH1_WALK_TTW", 0);
                if (walk_ttw != 0 && m/16 <= 128 && n/16 <= 128 && tcnt == nullptr) {
                    const int ttw = nt >= 3 ? 4 : 2;
                    const ggml_cuda_kernel_launch_params twp =
                        ggml_cuda_kernel_launch_params(dim3(m/16, 1, (nt + ttw - 1)/ttw),
                                                       dim3(1024, 1, 1), 0, stream);
#define M1_TTWIDE(TTV, ZD)                     mach1_launch(mach1_rt_walk_tt_kernel<1024, TTV, ZD, 0, 8>, twp,                         tre_v, tl_v, scr_u, scr_v, m, n, nt,                         ZD ? zt.ztab : (const uint16_t *) nullptr, ZD ? zt.zstep : 0.0f,                         (const float *) nullptr, (float *) nullptr, (int *) nullptr,                         (const float *) nullptr, (const float *) nullptr)
                    if (ttw == 4) {
                        if (zdp) { M1_TTWIDE(4, 1); } else { M1_TTWIDE(4, 0); }
                    } else {
                        if (zdp) { M1_TTWIDE(2, 1); } else { M1_TTWIDE(2, 0); }
                    }
#undef M1_TTWIDE
                    return;
                }
                const ggml_cuda_kernel_launch_params tp =
                    ggml_cuda_kernel_launch_params(dim3(m/16, 1, (nt + ttc - 1)/ttc),
                                                   dim3(wg, 1, 1), tsmem, stream);
#define M1_TTWALK(WGT, TTV, ZD, OF, UFV) \
                do { \
                    if (OF) { \
                        /* opt in at the worst-case m: the static block is 33-41 KB, so   */ \
                        /* even an 8 KB epilogue buffer crosses the 48 KB default budget  */ \
                        /* and the helper's small-size early-return would skip the call   */ \
                        mach1_smem_opt_in((const void *) mach1_rt_walk_tt_kernel<WGT, TTV, ZD, 1>, \
                                          std::max(tsmem, (size_t) 32*1024)); \
                        mach1_launch(mach1_rt_walk_tt_kernel<WGT, TTV, ZD, 1>, tp, \
                            tre_v, tl_v, scr_u, scr_v, m, n, nt, \
                            ZD ? zt.ztab : (const uint16_t *) nullptr, ZD ? zt.zstep : 0.0f, \
                            sv_v, dst_v, tcnt, \
                            (const float *) nullptr, (const float *) nullptr); \
                    } else if (UFV) { \
                        mach1_launch(mach1_rt_walk_tt_kernel<WGT, TTV, ZD, 0, 4, 1>, tp, \
                            tre_v, tl_v, (const float *) nullptr, scr_v, m, n, nt, \
                            ZD ? zt.ztab : (const uint16_t *) nullptr, ZD ? zt.zstep : 0.0f, \
                            (const float *) nullptr, (float *) nullptr, (int *) nullptr, \
                            su_v, x_v); \
                    } else { \
                        mach1_launch(mach1_rt_walk_tt_kernel<WGT, TTV, ZD, 0>, tp, \
                            tre_v, tl_v, scr_u, scr_v, m, n, nt, \
                            ZD ? zt.ztab : (const uint16_t *) nullptr, ZD ? zt.zstep : 0.0f, \
                            (const float *) nullptr, (float *) nullptr, (int *) nullptr, \
                            (const float *) nullptr, (const float *) nullptr); \
                    } \
                } while (0)
                const bool of = tcnt != nullptr;
                if (wg == 512 && ttc == 4) {
                    if (zdp) { M1_TTWALK(512, 4, 1, of, uf_tt); } else { M1_TTWALK(512, 4, 0, of, uf_tt); }
                } else if (wg == 512) {
                    if (zdp) { M1_TTWALK(512, 2, 1, of, uf_tt); } else { M1_TTWALK(512, 2, 0, of, uf_tt); }
                } else {
                    if (zdp) { M1_TTWALK(1024, 2, 1, of, uf_tt); } else { M1_TTWALK(1024, 2, 0, of, uf_tt); }
                }
#undef M1_TTWALK
                if (of) {
                    ofused = true;
                }
                return;
            }
            if (tcw && ocnt != nullptr) {
                ofused = true;
                if (ufuse) {
                    mach1_smem_opt_in((const void *) mach1_rt_walk_rows_v_kernel<512, 4, 1, 1, 1>, smem);
                    mach1_launch(mach1_rt_walk_rows_v_kernel<512, 4, 1, 1, 1>, vp,
                        tre_v, tl_v, scr_u, su_v, x_v, scr_v, m, n, sv_v, dst_v, ocnt, (const int *) nullptr, tc_htab,
                        (const uint16_t *) nullptr, 0.0f, xperm);
                } else {
                    mach1_smem_opt_in((const void *) mach1_rt_walk_rows_v_kernel<512, 4, 0, 1, 1>, smem);
                    mach1_launch(mach1_rt_walk_rows_v_kernel<512, 4, 0, 1, 1>, vp,
                        tre_v, tl_v, scr_u, su_v, x_v, scr_v, m, n, sv_v, dst_v, ocnt, (const int *) nullptr, tc_htab,
                        (const uint16_t *) nullptr, 0.0f, xperm);
                }
            } else if (tcw) {
                if (zdp && ufuse) {
                    mach1_smem_opt_in((const void *) mach1_rt_walk_rows_v_kernel<512, 4, 1, 0, 1, 0, 1>, smem);
                    mach1_launch(mach1_rt_walk_rows_v_kernel<512, 4, 1, 0, 1, 0, 1>, vp,
                        tre_v, tl_v, scr_u, su_v, x_v, scr_v, m, n,
                        (const float *) nullptr, (float *) nullptr, (int *) nullptr, (const int *) nullptr, tc_htab,
                        zt.ztab, zt.zstep, xperm);
                } else if (zdp) {
                    mach1_smem_opt_in((const void *) mach1_rt_walk_rows_v_kernel<512, 4, 0, 0, 1, 0, 1>, smem);
                    mach1_launch(mach1_rt_walk_rows_v_kernel<512, 4, 0, 0, 1, 0, 1>, vp,
                        tre_v, tl_v, scr_u, su_v, x_v, scr_v, m, n,
                        (const float *) nullptr, (float *) nullptr, (int *) nullptr, (const int *) nullptr, tc_htab,
                        zt.ztab, zt.zstep, xperm);
                } else {
                    mach1_smem_opt_in((const void *) mach1_rt_walk_rows_v_kernel<512, 4, 1, 0, 1>, smem);
                    mach1_launch(mach1_rt_walk_rows_v_kernel<512, 4, 1, 0, 1>, vp,
                        tre_v, tl_v, scr_u, su_v, x_v, scr_v, m, n,
                        (const float *) nullptr, (float *) nullptr, (int *) nullptr, (const int *) nullptr, tc_htab,
                        (const uint16_t *) nullptr, 0.0f, xperm);
                }
            } else if (ocnt != nullptr) {
                ofused = true;
                if (wg == 512 && ufuse) {
                    mach1_launch(mach1_rt_walk_rows_v_kernel<512, 4, 1, 1>, vp,
                        tre_v, tl_v, scr_u, su_v, x_v, scr_v, m, n, sv_v, dst_v, ocnt, (const int *) nullptr, (const half *) nullptr,
                        (const uint16_t *) nullptr, 0.0f, xperm);
                } else if (wg == 512) {
                    mach1_launch(mach1_rt_walk_rows_v_kernel<512, 4, 0, 1>, vp,
                        tre_v, tl_v, scr_u, su_v, x_v, scr_v, m, n, sv_v, dst_v, ocnt, (const int *) nullptr, (const half *) nullptr,
                        (const uint16_t *) nullptr, 0.0f, xperm);
                } else if (ufuse) {
                    mach1_launch(mach1_rt_walk_rows_v_kernel<1024, 4, 1, 1>, vp,
                        tre_v, tl_v, scr_u, su_v, x_v, scr_v, m, n, sv_v, dst_v, ocnt, (const int *) nullptr, (const half *) nullptr,
                        (const uint16_t *) nullptr, 0.0f, xperm);
                } else {
                    mach1_launch(mach1_rt_walk_rows_v_kernel<1024, 4, 0, 1>, vp,
                        tre_v, tl_v, scr_u, su_v, x_v, scr_v, m, n, sv_v, dst_v, ocnt, (const int *) nullptr, (const half *) nullptr,
                        (const uint16_t *) nullptr, 0.0f, xperm);
                }
            } else {
                // The plain walk is the one nt == 1 lands on, so it is where
                // the lane-addressed codebook is selectable.
                const int lx = mach1_lutx();
#define M1_VWALK(WGV, UF, LX, ZD) \
                mach1_launch(mach1_rt_walk_rows_v_kernel<WGV, 4, UF, 0, 0, LX, ZD>, vp, \
                    tre_v, tl_v, scr_u, su_v, x_v, scr_v, m, n, \
                    (const float *) nullptr, (float *) nullptr, (int *) nullptr, (const int *) nullptr, (const half *) nullptr, \
                    zt.ztab, zt.zstep, xperm)
#define M1_VWALK_LX(WGV, UF) \
                do { \
                    if      (zdp)      { M1_VWALK(WGV, UF,  0, 1); } \
                    else if (lx ==  8) { M1_VWALK(WGV, UF,  8, 0); } \
                    else if (lx ==  4) { M1_VWALK(WGV, UF,  4, 0); } \
                    else if (lx == 16) { M1_VWALK(WGV, UF, 16, 0); } \
                    else               { M1_VWALK(WGV, UF,  0, 0); } \
                } while (0)
                // wg 1024 caps the replication at 8: at CT = 256 the 16-way
                // table plus shu exceeds sm_89's 48 KB static-shared ptxas
                // limit, and 8 measured identical to 16 (round 209)
#define M1_VWALK_LX1024(UF) \
                do { \
                    if      (zdp)      { M1_VWALK(1024, UF, 0, 1); } \
                    else if (lx == 4)  { M1_VWALK(1024, UF, 4, 0); } \
                    else if (lx != 0)  { M1_VWALK(1024, UF, 8, 0); } \
                    else               { M1_VWALK(1024, UF, 0, 0); } \
                } while (0)
                if      (wg == 512 && ufuse) { M1_VWALK_LX(512,  1); }
                else if (wg == 512)          { M1_VWALK_LX(512,  0); }
                else if (ufuse)              { M1_VWALK_LX1024(1); }
                else                         { M1_VWALK_LX1024(0); }
#undef M1_VWALK_LX1024
#undef M1_VWALK_LX
#undef M1_VWALK
            }
            return;
        }

        // MEASURED (round 22 ablation): the rt walk is 4.1 ms of a 15.4 ms
        // token, and the shapes that miss the staged row-split are the reason.
        // n = 4096 (tiles_y 256) falls all the way through to the original
        // one-thread-per-column-tile kernel: 256 threads per block, 16 live
        // accumulators, no trellis staging - 128 blocks x 256 = 32768 threads
        // where the part wants ~270000, plus the ~12x read amplification the
        // staging exists to remove.
        //
        // CT = 256 (WG 1024, RG 4) covers it, and the reduction tree is
        // UNCHANGED: the old kernel warp-reduces 32 consecutive column tiles
        // and combines 8 warps sequentially, and so does this - CT is a
        // multiple of 32, so a warp never straddles a row group. Bit-exact.
        if (walk_rg == 4 && probe0 == 0 && n % 16 == 0 && tiles_y0 == 256 &&
            mach1_env_int("GGML_MACH1_WALK_WIDE", 0) != 0) {
            const ggml_cuda_kernel_launch_params wp =
                ggml_cuda_kernel_launch_params(dim3(m/16, 1, nt), dim3(1024, 1, 1), 0, stream);
            if (lut1k0) {
                mach1_launch(mach1_rt_walk_rows_s_kernel<1024, 4>, wp,
                    (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, scr_v, m, n);
            } else {
                mach1_launch(mach1_rt_walk_rows_kernel<1024, 4, 0>, wp,
                    (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, scr_v, m, n);
            }
            return;
        }
        // Same idea for the shapes that DO take the row-split: RG 8 halves the
        // per-thread accumulators again and doubles the block, so a small-m op
        // (m 512 -> 32 blocks) puts 32768 threads up instead of 16384. CT is
        // still 128, so the warp butterfly and the 4-warp combine are the same
        // ones RG 4 runs - bit-exact against it.
        if (walk_rg == 4 && probe0 == 0 && n % 16 == 0 && tiles_y0 <= 128 &&
            mach1_env_int("GGML_MACH1_WALK_WG", 512) == 1024) {
            const ggml_cuda_kernel_launch_params wp =
                ggml_cuda_kernel_launch_params(dim3(m/16, 1, nt), dim3(1024, 1, 1), 0, stream);
            if (lut1k0) {
                mach1_launch(mach1_rt_walk_rows_s_kernel<1024, 8>, wp,
                    (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, scr_v, m, n);
            } else {
                mach1_launch(mach1_rt_walk_rows_kernel<1024, 8, 0>, wp,
                    (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, scr_v, m, n);
            }
            return;
        }
        if (walk_rg == 4 && n/16 <= 128 && n % 16 == 0) {
            static const int probe = mach1_env_int("GGML_MACH1_WALK_PROBE", 0);
            static const bool lut1k = mach1_env_int("GGML_MACH1_LUT1K", 0) != 0;
            const ggml_cuda_kernel_launch_params rp =
                ggml_cuda_kernel_launch_params(dim3(m/16, 1, nt), dim3(512, 1, 1), 0, stream);
            if (lut1k && probe == 0) {
                mach1_launch(mach1_rt_walk_rows_s_kernel<512, 4>, rp,
                    (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, scr_v, m, n);
                return;
            }
            switch (probe) {
                case 1:  mach1_launch(mach1_rt_walk_rows_kernel<512, 4, 1>, rp,
                             (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, scr_v, m, n); break;
                case 2:  mach1_launch(mach1_rt_walk_rows_kernel<512, 4, 2>, rp,
                             (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, scr_v, m, n); break;
                case 3:  mach1_launch(mach1_rt_walk_rows_kernel<512, 4, 3>, rp,
                             (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, scr_v, m, n); break;
                default: mach1_launch(mach1_rt_walk_rows_kernel<512, 4, 0>, rp,
                             (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, scr_v, m, n); break;
            }
        } else if (n/16 <= 128) {
            mach1_launch(mach1_rt_walk_kernel<128>,
                ggml_cuda_kernel_launch_params(dim3(m/16, 1, nt), dim3(128, 1, 1), 0, stream),
                (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, scr_v, m, n);
        } else {
            mach1_launch(mach1_rt_walk_kernel<256>,
                ggml_cuda_kernel_launch_params(dim3(m/16, 1, nt), dim3(256, 1, 1), 0, stream),
                (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, scr_v, m, n);
        }
        });
    }

    // the tail matcher admitted this op on exactly the split-K TC out arm; a
    // stage that stopped issuing must abort here, never write an unfolded dst
    GGML_ASSERT(tail == nullptr ||
                (!ofused && !mach1_ablate(MACH1_ABLATE_RT_OUT) &&
                 rt_imma8_split4 && !mach1_rt_stdout_on()));

    if (!ofused && !mach1_ablate(MACH1_ABLATE_RT_OUT)) {
        if (mach1_rt_stdout_on() && rt_imma8_split4) {
            mach1_timed(stream, std::string("rt_out_rs_sum4") + shp, [&]() {
            mach1_rt_out_rs_sum4_p16(
                (const float *) sv->data, scr_v, (float *) dst->data, stream, nt);
            });
        } else if (mach1_rt_stdout_on()) {
            // key mirrors the branch the epilogue replaces so the TIME census
            // stays label-for-label comparable against the control arm
            const bool rs_tc = tc_htab != nullptr && mach1_tc_fwht_dim(m);
            mach1_timed(stream, std::string(rs_tc ? "rt_out_rs" : "rt_out_rs_plain") + shp, [&]() {
            mach1_rt_out_rs((const float *) sv->data, scr_v, (float *) dst->data, m, stream, nt);
            });
        } else if (rt_imma8_split4) {
            mach1_timed(stream, std::string(tail != nullptr ? "rt_out_tc_split4_tail"
                                                            : "rt_out_tc_split4") + shp, [&]() {
            mach1_rt_out_tc_sum4_p16(
                tc_htab, (const float *) sv->data, scr_v,
                (float *) (tail != nullptr ? tail->dst->data : dst->data), stream, nt, tail);
            });
        } else if (tc_htab != nullptr && mach1_tc_fwht_dim(m)) {
            mach1_timed(stream, std::string("rt_out_tc") + shp, [&]() {
            mach1_rt_out_tc(tc_htab, (const float *) sv->data, scr_v, (float *) dst->data, m, stream, nt);
            });
        } else
        mach1_timed(stream, std::string("rt_out") + shp, [&]() {
        if (mach1_fwht_wg() == 1024) {
            mach1_launch(mach1_rt_out_kernel<1024>,
                ggml_cuda_kernel_launch_params(dim3(nt, 1, 1), dim3(1024, 1, 1), 0, stream),
                (const float *) sv->data, scr_v, (float *) dst->data, m);
        } else if (mach1_fwht_wg512()) {
            mach1_launch(mach1_rt_out_kernel<512>,
                ggml_cuda_kernel_launch_params(dim3(nt, 1, 1), dim3(512, 1, 1), 0, stream),
                (const float *) sv->data, scr_v, (float *) dst->data, m);
        } else {
            mach1_launch(mach1_rt_out_kernel<256>,
                ggml_cuda_kernel_launch_params(dim3(nt, 1, 1), dim3(256, 1, 1), 0, stream),
                (const float *) sv->data, scr_v, (float *) dst->data, m);
        }
        });
    }
}

// GGML_MACH1_RT_TAIL: the two standalone rt projections per layer (attention
// wo / GDN ssm_out, and the shared-expert down) are each followed by
// elementwise nodes that read the whole [m, nt] rt output and nothing else -
// the residual ADD, or the sigmoid-gate MUL plus the moe and residual ADDs.
// The out transform already holds every element in a register at store time,
// so those launches become part of its store instead of their own grids. This
// is a LAUNCH-COUNT lever, not an arithmetic one: the B16 spine is grid bound
// and one small decode launch prices at 3.69 us (the LAUNCH_PROBE receipt).
// Values are the same fp32 operations in the same order as the nodes it
// replaces - see the TAIL block in mach1_tc_fwht_apply.
void mach1_diag_note(const char * key);   // defined in ggml-cuda.cu

static bool mach1_rt_tail_disjoint(const ggml_tensor * a, const ggml_tensor * b) {
    if (a == nullptr || b == nullptr || a->data == b->data) {
        return true;   // identical bases are per-element in place, which is safe
    }
    const char * ac = (const char *) a->data;
    const char * bc = (const char *) b->data;
    return !(ac < bc + ggml_nbytes(b) && bc < ac + ggml_nbytes(a));
}

int ggml_cuda_mach1_rt_tail_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    static const bool on = mach1_env_int("GGML_MACH1_RT_TAIL", 0) != 0 &&
                           mach1_env_int("GGML_MACH1_ABLATE", 0) == 0 &&
                           mach1_env_int("GGML_MACH1_FUSE_RT", 0) == 0 &&
                           mach1_env_int("GGML_MACH1_COOP", 0) == 0 &&
                           mach1_env_int("GGML_MACH1_UFUSE_TT", 0) == 0;
    if (!on || mach1_fold_enabled() || mach1_zbank_enabled() || mach1_rt_stdout_on()) {
        return 0;
    }

    ggml_tensor * rt = cgraph->nodes[node_idx];
    const ggml_tensor * su   = rt->src[1];
    const ggml_tensor * sv   = rt->src[2];
    const ggml_tensor * tlut = rt->src[3];
    const ggml_tensor * x    = rt->src[4];
    if (su == nullptr || sv == nullptr || tlut == nullptr || x == nullptr) {
        return 0;
    }

    // Admission is EXACTLY the split-K TC out arm this fold is written for
    // (mirrors split4_storage + rt_mma16 + rt_imma8_split4 in
    // ggml_cuda_op_mach1_rt_mm). Any other arm keeps its ordinary nodes; the
    // GGML_ASSERT there turns a drifted dispatch into a stop, never a silently
    // unfolded output.
    const int n  = (int) su->ne[0];
    const int m  = (int) sv->ne[0];
    const int nt = (int)(x->ne[1]*x->ne[2]*x->ne[3]);
    if (mach1_env_int("GGML_MACH1_RT_IMMA8", 0) == 0 ||
        mach1_env_int("GGML_MACH1_RT_MMA16", 0) == 0 ||
        mach1_env_int("GGML_MACH1_RT_MMA16_M2", 0) == 0 ||
        mach1_env_int("GGML_MACH1_RT_IMMA8_SPLITK", 0) == 0 ||
        !mach1_rt_imma8_cpasync_enabled()) {
        return 0;
    }
    if (m != 2048 ||
        !(n == 4096 || (n == 512 && mach1_env_int("GGML_MACH1_RT_IMMA8_SPLITK_N512", 0) != 0))) {
        return 0;
    }
    if (!(nt == 16 || (mach1_ntlow_on() && mach1_nt_ok(nt))) || !mach1_nt_ok(nt) ||
        nt > mach1_env_int("GGML_MACH1_TC_NT", 16)) {
        return 0;
    }
    const auto & dev = ggml_cuda_info().devices[ctx.device];
    if (!ampere_mma_available(dev.cc) || dev.warp_size != 32) {
        return 0;
    }
    if (!mach1_tc_fwht_dim(n) || !mach1_tc_fwht_dim(m)) {
        return 0;
    }
    if (tlut->ne[0] != 2 || tlut->ne[1] != 512 || rt->src[0]->ne[0] != 64) {
        return 0;
    }
    if (rt->type != GGML_TYPE_F32 || !ggml_is_contiguous(rt) ||
        rt->ne[0] != m || (int) ggml_nrows(rt) != nt) {
        return 0;
    }
    // runtime gates the env mirror above cannot see: the TC FWHT table
    // (GGML_MACH1_TC_FWHT + one-time cudaMalloc) and the z-ladder test on
    // this tlut. Both are cached, so dispatch resolves the same values;
    // failing either must degrade to the plain nodes here instead of
    // tripping the dispatch assert.
    if (mach1_tc_fwht_tab(ctx.device, ctx.stream()) == nullptr ||
        mach1_rt_z_table(tlut->data, ctx.stream()).ztab == nullptr) {
        return 0;
    }

    // a shape-preserving RESHAPE sits between the GDN ssm_out op and its
    // residual ADD (the attention wo op has none)
    ggml_op ops[6];
    int  k   = node_idx + 1;
    int  nop = 1;
    ops[0] = GGML_OP_MACH1_RT_MM;
    const ggml_tensor * val = rt;
    if (k < cgraph->n_nodes && cgraph->nodes[k]->op == GGML_OP_RESHAPE &&
        cgraph->nodes[k]->src[0] == rt) {
        val    = cgraph->nodes[k];
        ops[1] = GGML_OP_RESHAPE;
        nop    = 2;
        ++k;
    }
    if (k >= cgraph->n_nodes || (int) ggml_nelements(val) != m*nt) {
        return 0;
    }

    mach1_rt_tail tail = {};
    const ggml_tensor * mul = nullptr;
    if (cgraph->nodes[k]->op == GGML_OP_MUL) {
        // shared-expert down: gated = down * sigmoid[1, nt], then moe ADD
        mul = cgraph->nodes[k];
        if (mul->src[0] != val || mul->type != GGML_TYPE_F32 || !ggml_is_contiguous(mul)) {
            return 0;
        }
        const ggml_tensor * sig = mul->src[1];
        if (sig->type != GGML_TYPE_F32 || sig->ne[0] != 1 ||
            (int) ggml_nelements(sig) != nt || !ggml_is_contiguous(sig)) {
            return 0;
        }
        tail.gate = (const float *) sig->data;
        ops[nop++] = GGML_OP_MUL;
        ++k;
        if (k >= cgraph->n_nodes) {
            return 0;
        }
        val = mul;
    }
    if (cgraph->nodes[k]->op != GGML_OP_ADD) {
        return 0;
    }

    // up to two chained ADDs: the region output (moe + gated) and the ffn
    // residual. The stock multi-add fusion already runs that pair as one
    // launch, so a fold that took only the first would not remove a launch.
    ggml_tensor * out = nullptr;
    for (int i = 0; i < 2 && k < cgraph->n_nodes && cgraph->nodes[k]->op == GGML_OP_ADD; ++i) {
        ggml_tensor * add = cgraph->nodes[k];
        const ggml_tensor * other = add->src[0] == val ? add->src[1]
                                  : (add->src[1] == val ? add->src[0] : nullptr);
        if (other == nullptr || other->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(other) || (int) ggml_nelements(other) != m*nt ||
            add->type != GGML_TYPE_F32 || !ggml_is_contiguous(add) ||
            (int) ggml_nelements(add) != m*nt) {
            break;
        }
        if (i == 0) {
            tail.add = (const float *) other->data;
        } else {
            tail.add2 = (const float *) other->data;
        }
        ops[nop++] = GGML_OP_ADD;
        out = add;
        val = add;
        ++k;
    }
    if (out == nullptr) {
        return 0;
    }

    const int skip   = nop - 1;
    const int out_ix = node_idx + skip;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, nop, ops, &out_ix, 1)) {
        return 0;
    }

    // the store reads the folded operands at the same index it writes, so an
    // identical base is in place and safe; any partial overlap is not
    for (int j = 0; j < nop; ++j) {
        const ggml_tensor * nd = cgraph->nodes[node_idx + j];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = nd->src[s];
            if (src == nullptr) {
                continue;
            }
            bool inside = false;
            for (int q = 0; q < nop && !inside; ++q) {
                inside = src == cgraph->nodes[node_idx + q];
            }
            if (!inside && !mach1_rt_tail_disjoint(out, src)) {
                return 0;
            }
        }
    }

    tail.dst = out;
    ggml_cuda_op_mach1_rt_mm(ctx, rt, nullptr, nullptr, &tail);
    // per-step fold count (GGML_MACH1_TIME >= 2): the marker only proves that
    // ONE site engaged, and the wall delta only reads if every site did
    mach1_diag_note(mul != nullptr ? "rttail:shexp" : "rttail:residual");
    if (mach1_debug_on()) {
        static std::once_flag once_res;
        static std::once_flag once_glu;
        if (mul != nullptr) {
            std::call_once(once_glu, []() {
                fprintf(stderr, "mach1: rt tail fold ENGAGED (shexp down gate+add)\n");
            });
        } else {
            std::call_once(once_res, []() {
                fprintf(stderr, "mach1: rt tail fold ENGAGED (residual add)\n");
            });
        }
    }
    return skip;
}

//
// EXP_BASIS — one block per (slot, token) pair (mach1_exp_basis.comp)
//

static __global__ void mach1_exp_basis_kernel(
        const half    * __restrict__ a,
        const half    * __restrict__ b,
        const half    * __restrict__ c,
        const int32_t * __restrict__ remap,
        const int32_t * __restrict__ ids,
        const float   * __restrict__ x,
        const float   * __restrict__ acc_in,   // nullptr when has_acc == 0
        float         * __restrict__ dst,
        const int n, const int r, const int m, const int n_used, const int xne1,
        const int ids_s0, const int ids_s1, const int has_acc) {
    constexpr int WG = 256;
    __shared__ float tv[256];

    const int s   = blockIdx.x;
    const int t   = blockIdx.y;
    const int tid = threadIdx.x;

    const uint32_t id = (uint32_t) ids[s*ids_s0 + t*ids_s1];
    const uint32_t rm = (uint32_t) remap[id];

    const int64_t obase = (int64_t) t*n_used*m + (int64_t) s*m;
    if ((rm & GGML_CUDA_MACH1_DEM_FLAG) == 0u) {   // kept slot: block-uniform
        for (int i = tid; i < m; i += WG) {
            dst[obase + i] = has_acc ? acc_in[obase + i] : 0.0f;
        }
        return;
    }
    const int di = (int)(rm & ~GGML_CUDA_MACH1_DEM_FLAG);
    const int64_t xbase = (int64_t)(xne1 == 1 ? 0 : s*n) + (int64_t) t*xne1*n;

    // t_j = c_j * (A_j . x)
    for (int j = tid; j < r; j += WG) {
        float acc = 0.0f;
        for (int i = 0; i < n; ++i) {
            acc += __half2float(a[(int64_t) j*n + i]) * x[xbase + i];
        }
        // one rounding: t_j = fl(c_j * v_j)
        tv[j] = __half2float(c[(int64_t) di*r + j]) * acc;
    }
    __syncthreads();
    for (int i = tid; i < m; i += WG) {
        float acc = 0.0f;
        for (int j = 0; j < r; ++j) {
            // reference rounds B_ij*t_j before the add — pinned, no fma
            const float pr = __fmul_rn(__half2float(b[(int64_t) i*r + j]), tv[j]);
            acc = __fadd_rn(acc, pr);
        }
        dst[obase + i] = has_acc ? acc_in[obase + i] + acc : acc;
    }
}

void ggml_cuda_op_mach1_exp_basis(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a     = dst->src[0];
    const ggml_tensor * b     = dst->src[1];
    const ggml_tensor * c     = dst->src[2];
    const ggml_tensor * remap = dst->src[3];
    const ggml_tensor * ids   = dst->src[4];
    const ggml_tensor * x     = dst->src[5];
    const ggml_tensor * accs  = dst->src[6];

    GGML_ASSERT(a->type     == GGML_TYPE_F16);
    GGML_ASSERT(b->type     == GGML_TYPE_F16);
    GGML_ASSERT(c->type     == GGML_TYPE_F16);
    GGML_ASSERT(remap->type == GGML_TYPE_I32);
    GGML_ASSERT(ids->type   == GGML_TYPE_I32);
    GGML_ASSERT(x->type     == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));
    GGML_ASSERT(ggml_is_contiguous(c));
    GGML_ASSERT(ggml_is_contiguous(remap));
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(dst));
    if (accs != nullptr) {
        GGML_ASSERT(accs->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(accs));
    }

    const int n      = (int) a->ne[0];
    const int r      = (int) a->ne[1];
    const int m      = (int) b->ne[1];
    const int n_used = (int) ids->ne[0];
    const int n_tok  = (int) ids->ne[1];
    const int xne1   = (int) x->ne[1];
    const int ids_s0 = (int)(ids->nb[0]/sizeof(int32_t));
    const int ids_s1 = (int)(ids->nb[1]/sizeof(int32_t));
    GGML_ASSERT(r <= 256);   // tv shared bound

    mach1_launch(mach1_exp_basis_kernel,
        ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(256, 1, 1), 0, ctx.stream()),
        (const half    *) a->data,
        (const half    *) b->data,
        (const half    *) c->data,
        (const int32_t *) remap->data,
        (const int32_t *) ids->data,
        (const float   *) x->data,
        accs != nullptr ? (const float *) accs->data : nullptr,
        (float         *) dst->data,
        n, r, m, n_used, xne1, ids_s0, ids_s1, accs != nullptr ? 1 : 0);
}

//
// EXP_MM — 4 kernels: group pairs by storage expert -> u = H(su_e ⊙ x) per
// pair -> trellis walk/dot per group -> y = sv_e ⊙ H(v) per pair
// (mach1_exp_group/exp_u/exp_walk/exp_out.comp). Scratch: int32
// [4*n_groups + P] (cnt | off | orig | cursor(unused) | pairs) and float
// [P*n + P*m] (u then v).
//

static __global__ void mach1_exp_group_kernel(
        const int32_t * __restrict__ remap,
        const int32_t * __restrict__ ids,
        int32_t       * __restrict__ scr,
        const int n_used, const int n_tok, const int n_kept, const int n_groups,
        const int ids_s0, const int ids_s1) {
    constexpr int WG = 256;
    __shared__ int sh_cnt[512];    // MAX_GROUPS = 512, enforced by supports_op
    __shared__ int sh_cur[512];
    __shared__ int sh_orig[512];

    const int tid = threadIdx.x;
    const int P   = n_used*n_tok;

    for (int g = tid; g < n_groups; g += WG) {
        sh_cnt[g]  = 0;
        sh_orig[g] = 0;
    }
    __syncthreads();
    for (int p = tid; p < P; p += WG) {
        const int s = p % n_used;
        const int t = p / n_used;
        const int g = (int) mach1_group_of(remap, ids, p, n_used, n_kept, ids_s0, ids_s1);
        atomicAdd(&sh_cnt[g], 1);
        atomicExch(&sh_orig[g], ids[s*ids_s0 + t*ids_s1]);   // orig id (same value per group)
    }
    __syncthreads();
    if (tid == 0) {
        int off = 0;
        for (int g = 0; g < n_groups; ++g) {
            const int c = sh_cnt[g];
            sh_cur[g] = off;
            scr[g]              = c;            // cnt
            scr[n_groups + g]   = off;          // off
            scr[2*n_groups + g] = sh_orig[g];   // orig id
            off += c;
        }
    }
    __syncthreads();
    for (int p = tid; p < P; p += WG) {
        const int g   = (int) mach1_group_of(remap, ids, p, n_used, n_kept, ids_s0, ids_s1);
        const int pos = atomicAdd(&sh_cur[g], 1);
        scr[4*n_groups + pos] = p;              // flat pair rank p = t*n_used + s
    }
}

template <int WG>
__global__ void mach1_exp_u_kernel(
        const half    * __restrict__ su,
        const int32_t * __restrict__ ids,
        const float   * __restrict__ x,
        float         * __restrict__ scr_u,
        const int n, const int n_used, const int xne1,
        const int ids_s0, const int ids_s1) {
    __shared__ float sh[2048];

    const int s   = blockIdx.x;
    const int t   = blockIdx.y;
    const int p   = t*n_used + s;
    const int tid = threadIdx.x;

    const int64_t e = ids[s*ids_s0 + t*ids_s1];

    const int64_t xbase = (int64_t)(xne1 == 1 ? 0 : s*n) + (int64_t) t*xne1*n;
    for (int i = tid; i < n; i += WG) {
        sh[i] = __half2float(su[e*n + i]) * x[xbase + i];
    }
    __syncthreads();
    mach1_fwht_block(sh, n, tid, WG);
    const float sc = __fsqrt_rn((float) n);
    for (int i = tid; i < n; i += WG) {
        scr_u[(int64_t) p*n + i] = __fdiv_rn(sh[i], sc);
    }
}

// One-time host-side repack of the V8 tlut (F16 [8, 32768], 512 KB) into the
// mach engine's p4 form (MACH1_V8_LUT=p4): the V8 alphabet is <= 16 distinct
// fp16 levels, so a codeword row becomes one nibble-packed u32, resolved
// through a 16-entry level table.
//
// Only the EVEN rows are kept.  The runtime row is (reg*(reg+1)) & 0x7FFF and
// reg*(reg+1) is a product of consecutive integers, so it is always even: half
// the table has never been readable.  Dropping it is exact and takes the table
// from 128 KB to 64 KB, which on Ada is the difference between a codebook the
// size of the whole L1 and one that leaves half of it for everything else. The levels are stored as the exact
// fp32 values of the distinct F16 entries, so the resolved value is
// bit-identical to __half2float of the original entry by construction (no
// step multiply at runtime; +0/-0 collapse to one level like np.unique in
// the reference). Falls back to the fp16-gather path (packed == nullptr)
// when the alphabet has more than 16 distinct values.
//
// The cache is keyed by the tlut's device pointer and is NEVER freed: one
// entry is 64 KB + 64 B per distinct tlut (one per model) and the key's
// lifetime matches the weight allocation — intentionally leaked.
struct mach1_p4_table {
    const uint32_t * packed;   // device [32768], 8 x 4-bit level indices per row
    const float    * levels;   // device [16], exact fp32 of the fp16 alphabet
    const uint32_t * zpacked;  // device [16384], 8 x 4-bit (z+8) per row; the
                               // alphabet as an integer lattice (levels = s0*z),
                               // nullptr when the alphabet is not a ladder
    float            zstep;    // s0
};

static mach1_p4_table mach1_exp_p4_table(const void * tlut_data, cudaStream_t stream) {
    struct entry { const uint32_t * packed; const float * levels; const uint32_t * zpacked; float zstep; };
    static std::mutex mach1_p4_mutex;
    static std::unordered_map<const void *, entry> mach1_p4_cache;

    std::lock_guard<std::mutex> lock(mach1_p4_mutex);
    const auto it = mach1_p4_cache.find(tlut_data);
    if (it != mach1_p4_cache.end()) {
        return {it->second.packed, it->second.levels, it->second.zpacked, it->second.zstep};
    }

    // same capture guard as mach1_stream_counter: the D2H sync and cudaMalloc
    // below are illegal mid-capture, so return the fp16-gather fallback
    // uncached and let a later uncaptured call build the table
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) {
        cudaGetLastError();
        return {nullptr, nullptr, nullptr, 0.0f};
    }

    constexpr int ROWS  = 32768;      // rows in the persisted tlut
    constexpr int PROWS = ROWS / 2;   // even rows: the only ones addressable
    std::vector<uint16_t> host(8*ROWS);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), tlut_data, host.size()*sizeof(uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float levels[16];
    int   n_levels = 0;
    bool  ok = true;
    // alphabet over the EVEN rows only: an odd row is never gathered, so a
    // 17th level hiding in one must not disable the repack
    for (size_t r = 0; r < (size_t) ROWS; r += 2)
    for (size_t i = 8*r; i < 8*r + 8; ++i) {
        const float v = GGML_FP16_TO_FP32((ggml_fp16_t) host[i]);
        if (v != v) {   // NaN can never match a level slot
            ok = false;
            break;
        }
        int j = 0;
        while (j < n_levels && levels[j] != v) {
            j++;
        }
        if (j == n_levels) {
            if (n_levels == 16) {
                ok = false;
                break;
            }
            levels[n_levels++] = v;
        }
    }

    entry e = {nullptr, nullptr, nullptr, 0.0f};
    if (ok) {
        std::sort(levels, levels + n_levels);   // deterministic nibble order
        std::vector<uint32_t> packed(PROWS);
        for (int r = 0; r < PROWS; ++r) {
            uint32_t pk = 0;
            for (int c = 0; c < 8; ++c) {
                const float v = GGML_FP16_TO_FP32((ggml_fp16_t) host[16*r + c]);
                uint32_t idx = 0;
                while (levels[idx] != v) {
                    idx++;
                }
                pk |= idx << (4*c);
            }
            packed[r] = pk;
        }
        float lv16[16] = {0.0f};
        for (int i = 0; i < n_levels; ++i) {
            lv16[i] = levels[i];
        }
        // integer-lattice form: the z4 extraction documents the alphabet as
        // s0*z with z in {-4..4}, so levels/s0 must land on integers. The fp16
        // rounding of s0*z is absorbed by the 2 percent tolerance; the walk
        // then reconstructs s0*z instead of the fp16 value, which is the same
        // tolerance class as the TC-FWHT default.
        float s0 = 0.0f;
        for (int i = 0; i < n_levels; ++i) {
            const float a = fabsf(levels[i]);
            if (a > 0.0f && (s0 == 0.0f || a < s0)) {
                s0 = a;
            }
        }
        int  zmap[16] = {0};
        bool ladder   = s0 > 0.0f;
        for (int i = 0; ladder && i < n_levels; ++i) {
            const float r = levels[i]/s0;
            const int   z = (int) lroundf(r);
            ladder = fabsf(r - (float) z) <= 0.02f && z >= -8 && z <= 7;
            zmap[i] = z + 8;
        }
        std::vector<uint32_t> zpacked;
        if (ladder) {
            zpacked.resize(PROWS);
            for (int r = 0; r < PROWS; ++r) {
                const uint32_t pk = packed[r];
                uint32_t zk = 0;
                for (int c = 0; c < 8; ++c) {
                    zk |= (uint32_t) zmap[(pk >> (4*c)) & 0xFu] << (4*c);
                }
                zpacked[r] = zk;
            }
        }
        void * buf = nullptr;
        const size_t zoff = PROWS*sizeof(uint32_t) + 16*sizeof(float);
        CUDA_CHECK(cudaMalloc(&buf, zoff + (ladder ? PROWS*sizeof(uint32_t) : 0)));
        CUDA_CHECK(cudaMemcpyAsync(buf, packed.data(), PROWS*sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync((char *) buf + PROWS*sizeof(uint32_t), lv16,
                                   16*sizeof(float), cudaMemcpyHostToDevice, stream));
        if (ladder) {
            CUDA_CHECK(cudaMemcpyAsync((char *) buf + zoff, zpacked.data(),
                                       PROWS*sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        e.packed  = (const uint32_t *) buf;
        e.levels  = (const float *)((const char *) buf + PROWS*sizeof(uint32_t));
        e.zpacked = ladder ? (const uint32_t *)((const char *) buf + zoff) : nullptr;
        e.zstep   = ladder ? s0 : 0.0f;
    }
    mach1_p4_cache.emplace(tlut_data, e);
    if (mach1_debug_on() || mach1_time_on()) {
        fprintf(stderr, "mach1: p4 tlut repack %s (n_levels=%d, lattice=%s s0=%g, tlut=%p)\n",
                ok ? "ENGAGED" : "FALLBACK (fp16 gather)", n_levels,
                e.zpacked != nullptr ? "yes" : "no", e.zstep, tlut_data);
    }
    return {e.packed, e.levels, e.zpacked, e.zstep};
}

// expert integer walk: 0 off, 1 q8 activations, 2 the 15-bit two-plane form.
// EXP_DP4A implies the walk and wins over ZDP, which additionally converts the
// rt spine (mach1_zdp_on).
static int mach1_exp_zdp_mode() {
    static const int mode = mach1_env_int("GGML_MACH1_EXP_DP4A", 1) != 0 ? 2
                          : (mach1_env_int("GGML_MACH1_ZDP", 1) != 0 ? 1 : 0);
    return mode;
}

// V2 serial walk over register-held trellis words. STEPB/WORDS are template
// constants so the fully-unrolled window indices are compile-time and wl
// stays in registers (kept: <4,32>, demoted: <2,16>). Numerics identical to
// the previous in-loop decode.
template <int STEPB, int WORDS>
static __device__ __forceinline__ void mach1_exp_walk_v2_steps(
        const uint16_t * __restrict__ wl,      // [WORDS] registers
        const float    * __restrict__ tlut,    // F32 [512, 2]
        const float    * __restrict__ ub,
        float          * __restrict__ partial) {
#pragma unroll
    for (int i = 0; i < 128; ++i) {
        // direct-window state: 16 bits at stream bit STEPB*i (wrapping)
        const int      bb  = STEPB*i;
        const int      wi  = bb >> 4;
        const int      o   = bb & 15;
        const int      wn  = wi + 1 < WORDS ? wi + 1 : 0;
        const uint32_t w2  = ((uint32_t) wl[wi] << 16) | (uint32_t) wl[wn];
        const uint32_t reg = (w2 >> (16 - o)) & 0xFFFFu;
        const uint32_t ph  = reg*(reg + 1u);
        const uint32_t row = (ph >> 6) & 511u;
        float v0 = tlut[2*row + 0];
        const float v1raw = tlut[2*row + 1];
        if (ph & 0x8000u) {
            v0 = -v0;                         // exact either side of the round
        }
        // hatWr is defined at fp16 precision — pinned round-trip
        const float w0 = __half2float(__float2half_rn(v0));
        const float w1 = __half2float(__float2half_rn(v1raw));
        const int ri = (2*i) >> 4;
        const int ci = (2*i) & 15;
        partial[ri] += w0*ub[ci] + w1*ub[ci + 1];
    }
}

// V8 = false: payload v2, V=2, K=2 kept (32 words) / K=1 demoted (16 words),
//             tlut F32 [512,2], row = (p>>6)&511, per-value fp16 round here.
// V8 = true:  payload v3, V=8, K=1.5 (24 words, 12 fresh bits per step), tlut
//             pre-rounded F16 [32768,8], row = p&0x7FFF, per-tile wave gamma
//             applied after the round, no demoted tier.
// P4 (V8 only): gather one nibble-packed u32 per state from the repacked
//             table and resolve levels from shared — bit-identical values.
// The tile's trellis words and (for V8) the whole per-step state stream are
// pair-invariant, so they are hoisted out of the pair loop (for V2 the full
// 128-step state precompute would cost ~64+ registers, so only the words are
// hoisted there).
template <bool V8, bool P4>
__global__ void mach1_exp_walk_kernel(
        const uint16_t * __restrict__ kept,
        const uint16_t * __restrict__ dem,     // placeholder (= kept) when absent
        const void     * __restrict__ tlut,    // float (V2) or half (V8)
        const uint32_t * __restrict__ p4,      // P4 only, nullptr otherwise
        const float    * __restrict__ p4lv,    // P4 only, nullptr otherwise
        const int32_t  * __restrict__ scr_i,
        const float    * __restrict__ scr_u,
        float          * __restrict__ scr_v,
        const half     * __restrict__ gamma,   // V8 only, nullptr otherwise
        const int m, const int n, const int n_kept, const int n_groups) {
    static_assert(V8 || !P4, "P4 is a V8-tlut repack");
    constexpr int WG        = 128;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = WG / warp_size;
    __shared__ float red[16][4];   // >= n_warps for warp_size 32 (and 64 on HIP)
    __shared__ float lv[16];       // P4 level table

    (void) dem;
    (void) gamma;
    (void) p4;
    (void) p4lv;

    const int g   = blockIdx.z;
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;

    const int cnt = scr_i[g];
    if (cnt == 0) {                // block-uniform: whole block exits together
        return;
    }
    if constexpr (P4) {
        if (tid < 16) {
            lv[tid] = p4lv[tid];
        }
        __syncthreads();
    }
    const int  off     = scr_i[n_groups + g];
    const int  tiles_y = n / 16;
    const int  ntiles  = (m / 16)*tiles_y;
    const bool have    = tid < tiles_y;

    int              words;
    int              stepb;        // fresh bits per step
    const uint16_t * trd;
    int64_t          tbase;
    float            gsc = 0.0f;
    if constexpr (V8) {
        words = 24;
        stepb = 12;
        trd   = kept;
        tbase = ((int64_t) g*ntiles + (int64_t) tr*tiles_y)*words;
        // per-tile wave gamma: last anti-diagonal wavefront writing tile (tr, tid)
        const int e_orig = scr_i[2*n_groups + g];
        const int Mb     = m / 16;
        if (have) {
            const int wv = (tr + tid <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + tid)
                                                     : Mb + tiles_y - 2 - (tr + tid);
            gsc = __half2float(gamma[(int64_t) e_orig*(Mb + tiles_y) + wv]);
        }
    } else {
        const bool is_dem = g >= n_kept;
        const int  ei     = is_dem ? g - n_kept : g;
        words = is_dem ? 16 : 32;
        stepb = is_dem ? 2 : 4;
        trd   = is_dem ? dem : kept;
        tbase = ((int64_t) ei*ntiles + (int64_t) tr*tiles_y)*words;
    }
    (void) stepb;

    // pair-invariant hoist: V8 precomputes all 32 step states once (the pair
    // loop then only gathers + MACs); V2 stages the tile words in registers.
    uint32_t phv[V8 ? 32 : 1];
    uint16_t wl[V8 ? 1 : 32];
    if (have) {
        const int64_t tw = tbase + (int64_t) tid*words;
        if constexpr (V8) {
            uint16_t w8[24];
#pragma unroll
            for (int q = 0; q < 24; ++q) {
                w8[q] = trd[tw + q];
            }
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                // direct-window state: 16 bits at stream bit 12*i (wrapping)
                const int      bb  = 12*i;
                const int      wi  = bb >> 4;
                const int      o   = bb & 15;
                const int      wn  = wi + 1 < 24 ? wi + 1 : 0;
                const uint32_t w2  = ((uint32_t) w8[wi] << 16) | (uint32_t) w8[wn];
                const uint32_t reg = (w2 >> (16 - o)) & 0xFFFFu;
                phv[i] = reg*(reg + 1u);
            }
        } else {
#pragma unroll
            for (int q = 0; q < 32; ++q) {   // predicated: no read past 16-word dem tiles
                wl[q] = q < words ? trd[tw + q] : (uint16_t) 0;
            }
        }
    }

    const int lane = tid % warp_size;
    const int wid  = tid / warp_size;

    // one pair per iteration on purpose — see the MEASURED note in
    // mach1_exp_walk.comp: blocking pairs 2-at-a-time regressed 3x.
    for (int qq = 0; qq < cnt; ++qq) {
        const int p = scr_i[4*n_groups + off + qq];

        float partial[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            partial[i] = 0.0f;
        }
        if (have) {
            const float * ub = scr_u + (int64_t) p*n + tid*16;
            if constexpr (V8) {
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    const uint32_t ph  = phv[i];
                    const uint32_t row = ph & 0x7FFFu;
                    const int      ri  = (8*i) >> 4;
                    const int      ci  = (8*i) & 15;
                    float dotp = 0.0f;
                    if constexpr (P4) {
                        const uint32_t pk = p4[row >> 1];
#pragma unroll
                        for (int c = 0; c < 8; ++c) {
                            // exact fp32 of the fp16 entry — bit-identical to
                            // the __half2float gather; gamma order unchanged
                            float vv = lv[(pk >> (4*c)) & 0xFu];
                            if (c == 0 && (ph & 0x8000u)) {
                                vv = -vv;                 // exact (sign bit)
                            }
                            dotp += (vv * gsc) * ub[ci + c];
                        }
                    } else {
                        const half * tl = (const half *) tlut + 8*(int64_t) row;
#pragma unroll
                        for (int c = 0; c < 8; ++c) {
                            // tlut is pre-rounded fp16; wave gamma then
                            // multiplies in fp32 (reference op order)
                            float vv = __half2float(tl[c]);
                            if (c == 0 && (ph & 0x8000u)) {
                                vv = -vv;                 // exact in fp16 (sign bit)
                            }
                            dotp += (vv * gsc) * ub[ci + c];
                        }
                    }
                    partial[ri] += dotp;
                }
            } else {
                // stepb is block-uniform (kept vs demoted group); branch so
                // the unrolled walk sees compile-time window indices
                if (words == 32) {
                    mach1_exp_walk_v2_steps<4, 32>(wl, (const float *) tlut, ub, partial);
                } else {
                    mach1_exp_walk_v2_steps<2, 16>(wl, (const float *) tlut, ub, partial);
                }
            }
        }
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
            scr_v[(int64_t) p*m + tr*16 + tid] = sum;
        }
        __syncthreads();   // red is reused by the next pair
    }
}

// Row-split V8 walk: the exp_walk twin of mach1_rt_walk_rows_kernel.
//
// In the V8 layout state i feeds tile-row i/2 (columns (8i)&15 .. +8), so a
// thread that owns tile-rows [4rg, 4rg+4) owns exactly states [8rg, 8rg+8) -
// 8 of 32 - and carries 4 accumulators instead of 16. Same 4x thread count,
// same register relief. Per-state gathers, wave gamma and the per-pair
// accumulation order are unchanged, and the cross-tile reduction keeps the
// same warp-butterfly shape, so this is bit-exact against
// mach1_exp_walk_kernel<true, P4>.
template <bool P4, int WG, int RG>
__global__ void mach1_exp_walk_rows_kernel(
        const uint16_t * __restrict__ kept,
        const void     * __restrict__ tlut,    // half [32768, 8]
        const uint32_t * __restrict__ p4,
        const float    * __restrict__ p4lv,
        const int32_t  * __restrict__ scr_i,
        const float    * __restrict__ scr_u,
        float          * __restrict__ scr_v,
        const half     * __restrict__ gamma,
        const int m, const int n, const int n_groups) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int CT        = WG / RG;
    constexpr int n_warps   = CT / warp_size;
    constexpr int ROWS      = 16/RG;
    constexpr int STATES    = 32/RG;
    __shared__ float red[RG][ROWS][4];
    __shared__ float lv[16];

    (void) p4;
    (void) p4lv;

    const int g   = blockIdx.z;
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;
    const int ct  = tid % CT;
    const int rg  = tid / CT;

    const int cnt = scr_i[g];
    if (cnt == 0) {
        return;
    }
    if constexpr (P4) {
        if (tid < 16) {
            lv[tid] = p4lv[tid];
        }
        __syncthreads();
    }
    const int  off     = scr_i[n_groups + g];
    const int  tiles_y = n / 16;
    const int  ntiles  = (m / 16)*tiles_y;
    const bool have    = ct < tiles_y;

    constexpr int words = 24;
    const int64_t tbase = ((int64_t) g*ntiles + (int64_t) tr*tiles_y)*words;

    const int e_orig = scr_i[2*n_groups + g];
    const int Mb     = m / 16;
    float gsc = 0.0f;
    if (have) {
        const int wv = (tr + ct <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + ct)
                                                 : Mb + tiles_y - 2 - (tr + ct);
        gsc = __half2float(gamma[(int64_t) e_orig*(Mb + tiles_y) + wv]);
    }

    uint32_t phv[STATES];
    if (have) {
        const int64_t tw = tbase + (int64_t) ct*words;
        // 3 x 128-bit instead of 24 LDG.U16 (see the fused twin): same words,
        // aligned because tw*2 is a multiple of 48.
        uint4 kv[3];
        {
            const uint4 * kb = (const uint4 *) (kept + tw);
#pragma unroll
            for (int q = 0; q < 3; ++q) {
                kv[q] = kb[q];
            }
        }
        const uint16_t * w8 = (const uint16_t *) kv;
#pragma unroll
        for (int s = 0; s < STATES; ++s) {
            const int      i   = rg*STATES + s;
            const int      bb  = 12*i;
            const int      wi  = bb >> 4;
            const int      o   = bb & 15;
            const int      wn  = wi + 1 < 24 ? wi + 1 : 0;
            const uint32_t w2  = ((uint32_t) w8[wi] << 16) | (uint32_t) w8[wn];
            // no & 0xFFFF: only bits 0..15 of ph are read (row = ph & 0x7FFF,
            // sign = bit 15) and bit i of a product depends only on bits 0..i
            // of its operands, so a dirty high half cannot reach them.
            const uint32_t reg = w2 >> (16 - o);
            phv[s] = reg*(reg + 1u);
        }
    }

    const int lane = ct % warp_size;
    const int wid  = ct / warp_size;

    for (int qq = 0; qq < cnt; ++qq) {
        const int p = scr_i[4*n_groups + off + qq];

        float partial[ROWS];
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            partial[i] = 0.0f;
        }
        if (have) {
            const float * ub = scr_u + (int64_t) p*n + ct*16;
#pragma unroll
            for (int s = 0; s < STATES; ++s) {
                const int      i   = rg*STATES + s;
                const uint32_t ph  = phv[s];
                const uint32_t row = ph & 0x7FFFu;
                const int      ri  = ((8*i) >> 4) - rg*ROWS;
                const int      ci  = (8*i) & 15;
                float dotp = 0.0f;
                if constexpr (P4) {
                    const uint32_t pk = p4[row >> 1];
#pragma unroll
                    for (int c = 0; c < 8; ++c) {
                        float vv = lv[(pk >> (4*c)) & 0xFu];
                        if (c == 0 && (ph & 0x8000u)) {
                            vv = -vv;
                        }
                        dotp += (vv * gsc) * ub[ci + c];
                    }
                } else {
                    const half * tl = (const half *) tlut + 8*(int64_t) row;
#pragma unroll
                    for (int c = 0; c < 8; ++c) {
                        float vv = __half2float(tl[c]);
                        if (c == 0 && (ph & 0x8000u)) {
                            vv = -vv;
                        }
                        dotp += (vv * gsc) * ub[ci + c];
                    }
                }
                partial[ri] += dotp;
            }
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const float s = warp_reduce_sum<warp_size>(partial[r]);
            if (lane == 0) {
                red[rg][r][wid] = s;
            }
        }
        __syncthreads();
        if (tid < 16) {
            const int gg = tid / ROWS;
            const int rr = tid % ROWS;
            float sum = 0.0f;
#pragma unroll
            for (int wj = 0; wj < n_warps; ++wj) {
                sum += red[gg][rr][wj];
            }
            scr_v[(int64_t) p*m + tr*16 + gg*ROWS + rr] = sum;
        }
        __syncthreads();
    }
}

// V8 fused walk for narrow inputs (tiles_y <= warp_size): one WARP per row
// tile, lanes = column tiles, 4 row tiles per block — the block-per-row-tile
// kernel above leaves (WG - tiles_y) of its threads idle in the walk, which
// wastes 3/4 of the block for the real model's down projection (n = 512).
// Per-(row tile, column tile) state math, gathers, wave gamma and per-pair
// accumulation order are identical to mach1_exp_walk_kernel<true, P4>; the
// warp reduction is the same shuffle tree, and the cross-warp shared step it
// replaces only ever added exact zeros there.
template <bool P4>
__global__ void mach1_exp_walk_v8_warp_kernel(
        const uint16_t * __restrict__ kept,
        const void     * __restrict__ tlut,    // half [32768, 8]
        const uint32_t * __restrict__ p4,      // P4 only, nullptr otherwise
        const float    * __restrict__ p4lv,    // P4 only, nullptr otherwise
        const int32_t  * __restrict__ scr_i,
        const float    * __restrict__ scr_u,
        float          * __restrict__ scr_v,
        const half     * __restrict__ gamma,
        const int m, const int n, const int n_groups) {
    constexpr int warp_size = 32;   // caller guards tiles_y <= 32
    __shared__ float lv[16];

    (void) p4;
    (void) p4lv;

    const int g   = blockIdx.z;
    const int tid = threadIdx.x;

    const int cnt = scr_i[g];
    if (cnt == 0) {                // block-uniform: whole block exits together
        return;
    }
    if constexpr (P4) {
        if (tid < 16) {
            lv[tid] = p4lv[tid];
        }
        __syncthreads();
    }
    const int  off     = scr_i[n_groups + g];
    const int  tiles_y = n / 16;
    const int  ntiles  = (m / 16)*tiles_y;
    const int  lane    = tid % warp_size;
    const int  tr      = blockIdx.x*4 + tid / warp_size;   // 4 warps = 4 row tiles
    const bool have    = lane < tiles_y;                    // tr < m/16 by grid

    constexpr int words = 24;   // V = 8, K = 1.5
    const int64_t tbase = ((int64_t) g*ntiles + (int64_t) tr*tiles_y)*words;

    // per-tile wave gamma: same closed form as the block kernel
    const int e_orig = scr_i[2*n_groups + g];
    const int Mb     = m / 16;
    float gsc = 0.0f;
    if (have) {
        const int wv = (tr + lane <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + lane)
                                                  : Mb + tiles_y - 2 - (tr + lane);
        gsc = __half2float(gamma[(int64_t) e_orig*(Mb + tiles_y) + wv]);
    }

    // pair-invariant hoist: all 32 step states once per (row tile, column tile)
    uint32_t phv[32];
    if (have) {
        const int64_t tw = tbase + (int64_t) lane*words;
        uint16_t w8[24];
#pragma unroll
        for (int q = 0; q < 24; ++q) {
            w8[q] = kept[tw + q];
        }
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int      bb  = 12*i;
            const int      wi  = bb >> 4;
            const int      o   = bb & 15;
            const int      wn  = wi + 1 < 24 ? wi + 1 : 0;
            const uint32_t w2  = ((uint32_t) w8[wi] << 16) | (uint32_t) w8[wn];
            // no & 0xFFFF: only bits 0..15 of ph are read (row = ph & 0x7FFF,
            // sign = bit 15) and bit i of a product depends only on bits 0..i
            // of its operands, so a dirty high half cannot reach them.
            const uint32_t reg = w2 >> (16 - o);
            phv[i] = reg*(reg + 1u);
        }
    }

    for (int qq = 0; qq < cnt; ++qq) {
        const int p = scr_i[4*n_groups + off + qq];

        float partial[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            partial[i] = 0.0f;
        }
        if (have) {
            const float * ub = scr_u + (int64_t) p*n + lane*16;
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                const uint32_t ph  = phv[i];
                const uint32_t row = ph & 0x7FFFu;
                const int      ri  = (8*i) >> 4;
                const int      ci  = (8*i) & 15;
                float dotp = 0.0f;
                if constexpr (P4) {
                    const uint32_t pk = p4[row >> 1];
#pragma unroll
                    for (int c = 0; c < 8; ++c) {
                        float vv = lv[(pk >> (4*c)) & 0xFu];
                        if (c == 0 && (ph & 0x8000u)) {
                            vv = -vv;                 // exact (sign bit)
                        }
                        dotp += (vv * gsc) * ub[ci + c];
                    }
                } else {
                    const half * tl = (const half *) tlut + 8*(int64_t) row;
#pragma unroll
                    for (int c = 0; c < 8; ++c) {
                        float vv = __half2float(tl[c]);
                        if (c == 0 && (ph & 0x8000u)) {
                            vv = -vv;                 // exact in fp16 (sign bit)
                        }
                        dotp += (vv * gsc) * ub[ci + c];
                    }
                }
                partial[ri] += dotp;
            }
        }
        float sums[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            sums[i] = warp_reduce_sum<warp_size>(partial[i]);
        }
        if (lane == 0) {
            float * vo = scr_v + (int64_t) p*m + tr*16;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                vo[i] = sums[i];
            }
        }
        // no __syncthreads(): warps share nothing across pairs
    }
}

// Fused decode-path EXP walk (V8, single token): grid (m/16, 1, n_used), one
// z-plane per used slot. At n_tok == 1 the router's top-k slots are distinct
// experts, so the exp_group pass buys nothing: each block resolves its slot's
// storage index directly (remap[ids[s]], same mapping mach1_group_of applies)
// and computes u = H(su_e .* x)/sqrt(n) in shared memory itself — bit-identical
// to mach1_exp_u_kernel (same fwht_block, same pinned round, same inputs) —
// then runs the exp_walk body for its 16 rows. Removes two of the four chain
// kernels; exp_out is unchanged. Per-pair walk order is exactly the split
// kernel's, so results are bit-identical banks-off.
// FAST (GGML_MACH1_EXPFAST=1) keeps the same math and removes shared traffic
// from the inner loop, which is where this kernel spends its instructions:
//   - ub[0..15] is fixed per thread (ub = shu + tid*16, ci is 0 or 8) yet the
//     unrolled body re-reads it 8x per state, 256 shared loads per thread.
//     Hoisting it into 16 registers costs nothing and removes all of them.
//   - the P4 level gather lv[nibble] is one LDS.32 per weight. lv2[byte]
//     holds the two levels of a nibble PAIR, so it is one LDS.64 per two
//     weights instead - same values, half the gather instructions.
// Both are value-identical and keep the c = 0..7 accumulation order, so FAST
// is bit-exact against the default body.
// ZDP runs the same walk as integers (mach1_exp_zdp_planes/_state): 1 is q8
// activations, 2 the 15-bit two-plane form. This is the split path, so the
// nibble table stays in GLOBAL - worth less than the megakernel's resident
// copy (round 46: -30% on the walk against -52%) and not bit-exact.
template <bool P4, bool FAST, int ZDP = 0>
__global__ void mach1_exp_fused_kernel(
        const uint16_t * __restrict__ kept,
        const void     * __restrict__ tlut,    // half [32768, 8]
        const uint32_t * __restrict__ p4,      // P4 only, nullptr otherwise
        const float    * __restrict__ p4lv,    // P4 only, nullptr otherwise
        const half     * __restrict__ su,
        const int32_t  * __restrict__ remap,
        const int32_t  * __restrict__ ids,
        const float    * __restrict__ x,
        float          * __restrict__ scr_v,
        const half     * __restrict__ gamma,
        const int m, const int n, const int xne1,
        const int n_used, const int n_kept, const int ids_s0, const int ids_s1,
        const uint32_t * __restrict__ zp4,   // ZDP only, nullptr otherwise
        const float zs0) {
    constexpr int WG        = 128;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = WG / warp_size;
    __shared__ float shu[2048];    // n <= 2048 (supports_op)
    __shared__ float red[16][4];   // >= n_warps
    __shared__ float lv[16];
    __shared__ float2 lv2[(P4 && FAST && !ZDP) ? 256 : 1];

    (void) p4;
    (void) p4lv;

    const int s   = blockIdx.z;
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;

    const int64_t  e = ids[s*ids_s0];
    const uint32_t g = mach1_group_of(remap, ids, s, n_used, n_kept, ids_s0, ids_s1);

    if constexpr (P4 && !ZDP) {
        if (tid < 16) {
            lv[tid] = p4lv[tid];
        }
        if constexpr (FAST) {
            __syncthreads();
            for (int b = tid; b < 256; b += WG) {
                lv2[b] = make_float2(lv[b & 0xF], lv[(b >> 4) & 0xF]);
            }
        }
    }
    const int64_t xbase = (int64_t)(xne1 == 1 ? 0 : s*n);
    for (int i = tid; i < n; i += WG) {
        shu[i] = __half2float(su[e*n + i]) * x[xbase + i];
    }
    __syncthreads();
    mach1_fwht_block(shu, n, tid, WG);
    const float scn = __fsqrt_rn((float) n);
    for (int i = tid; i < n; i += WG) {
        shu[i] = __fdiv_rn(shu[i], scn);
    }
    __syncthreads();

    const int  tiles_y = n / 16;
    const int  ntiles  = (m / 16)*tiles_y;
    const bool have    = tid < tiles_y;

    constexpr int words = 24;   // V = 8, K = 1.5
    const int64_t tbase = ((int64_t) g*ntiles + (int64_t) tr*tiles_y)*words;

    const int Mb = m / 16;
    float gsc = 0.0f;
    if (have) {
        const int wv = (tr + tid <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + tid)
                                                 : Mb + tiles_y - 2 - (tr + tid);
        gsc = __half2float(gamma[e*(Mb + tiles_y) + wv]);
    }

    float partial[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        partial[i] = 0.0f;
    }
    if (have) {
        const int64_t tw = tbase + (int64_t) tid*words;
        // 24 uint16 is exactly 3 x 128-bit, and tw*2 is a multiple of 48, so
        // the vector form is aligned. The scalar loop issues 24 LDG.U16 whose
        // lanes are 48 bytes apart - 32 sectors touched per instruction to
        // deliver 64 bytes. Same values; measured -40% on the rt twin.
        uint4 kv[3];
        {
            const uint4 * kb = (const uint4 *) (kept + tw);
#pragma unroll
            for (int q = 0; q < 3; ++q) {
                kv[q] = kb[q];
            }
        }
        const uint16_t * w8 = (const uint16_t *) kv;
        const float * ub = shu + tid*16;
        float uu[(P4 && (FAST || ZDP)) ? 16 : 1];
        (void) uu;
        if constexpr (P4 && (FAST || ZDP)) {
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                uu[c] = ub[c];
            }
        }
        if constexpr (ZDP) {
            constexpr bool HI2  = ZDP == 2;
            constexpr int  QMAX = HI2 ? 16319 : 127;
            uint32_t qe[2], qo[2];
            uint32_t qle[HI2 ? 2 : 1], qlo[HI2 ? 2 : 1];
            int      qc0[2];
            int      sumq;
            const float du = mach1_exp_zdp_planes<HI2>(uu, qe, qo, qle, qlo, qc0, sumq);
            int ipart[16];
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                ipart[r] = 0;
            }
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                const int      bb  = 12*i;
                const int      wi  = bb >> 4;
                const int      o   = bb & 15;
                const int      wn  = wi + 1 < 24 ? wi + 1 : 0;
                const uint32_t w2  = ((uint32_t) w8[wi] << 16) | (uint32_t) w8[wn];
                const uint32_t reg = w2 >> (16 - o);
                const uint32_t ph  = reg*(reg + 1u);
                ipart[i >> 1] += mach1_exp_zdp_state<HI2>(zp4[(ph & 0x7FFFu) >> 1], ph,
                                     qe, qo, qle, qlo, qc0, i & 1);
            }
            // row ri takes states 2ri and 2ri+1, one per column half, so the
            // z + 8 bias is the same 8*sumq for every row
            const float fsc = (zs0*gsc)*(du*(1.0f/(float) QMAX));
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                partial[r] = (float)(ipart[r] - 8*sumq)*fsc;
            }
        } else {
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int      bb  = 12*i;
            const int      wi  = bb >> 4;
            const int      o   = bb & 15;
            const int      wn  = wi + 1 < 24 ? wi + 1 : 0;
            const uint32_t w2  = ((uint32_t) w8[wi] << 16) | (uint32_t) w8[wn];
            // no & 0xFFFF: only bits 0..15 of ph are read (row = ph & 0x7FFF,
            // sign = bit 15) and bit i of a product depends only on bits 0..i
            // of its operands, so a dirty high half cannot reach them.
            const uint32_t reg = w2 >> (16 - o);
            const uint32_t ph  = reg*(reg + 1u);
            const uint32_t row = ph & 0x7FFFu;
            const int      ri  = (8*i) >> 4;
            const int      ci  = (8*i) & 15;
            float dotp = 0.0f;
            if constexpr (P4 && FAST) {
                const uint32_t pk = p4[row >> 1];
#pragma unroll
                for (int c = 0; c < 8; c += 2) {
                    const float2 pr = lv2[(pk >> (4*c)) & 0xFFu];
                    float vv = pr.x;
                    if (c == 0 && (ph & 0x8000u)) {
                        vv = -vv;                 // exact (sign bit)
                    }
                    dotp += (vv * gsc) * uu[ci + c];
                    dotp += (pr.y * gsc) * uu[ci + c + 1];
                }
            } else if constexpr (P4) {
                const uint32_t pk = p4[row >> 1];
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    float vv = lv[(pk >> (4*c)) & 0xFu];
                    if (c == 0 && (ph & 0x8000u)) {
                        vv = -vv;                 // exact (sign bit)
                    }
                    dotp += (vv * gsc) * ub[ci + c];
                }
            } else {
                const half * tl = (const half *) tlut + 8*(int64_t) row;
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    float vv = __half2float(tl[c]);
                    if (c == 0 && (ph & 0x8000u)) {
                        vv = -vv;                 // exact in fp16 (sign bit)
                    }
                    dotp += (vv * gsc) * ub[ci + c];
                }
            }
            partial[ri] += dotp;
        }
        }
    }
    const int lane = tid % warp_size;
    const int wid  = tid / warp_size;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float s2 = warp_reduce_sum<warp_size>(partial[i]);
        if (lane == 0) {
            red[i][wid] = s2;
        }
    }
    __syncthreads();
    if (tid < 16) {
        float sum = 0.0f;
#pragma unroll
        for (int wj = 0; wj < n_warps; ++wj) {
            sum += red[tid][wj];
        }
        scr_v[(int64_t) s*m + tr*16 + tid] = sum;
    }
}

// Row-split fused EXP walk (GGML_MACH1_EXP_ROWS=1).
//
// MEASURED (round 22 ablation): the exp walk is 3.4 ms of a 15.4 ms token
// while the whole non-weight graph is 2.3 ms, and the fused kernel above puts
// far too few threads on the part to hide any of it. Its block is 128 threads
// and its grid is (m/16, 1, n_used): for the gate/up projections that is
// 32 x 8 = 256 blocks x 128 = 32768 threads, 12% of what an H200 holds, with
// every thread carrying 16 accumulators and a whole 16x16 tile (256 weights).
// The down projection is worse - n = 512 means tiles_y = 32, so 96 of every
// 128 threads sit out the walk entirely.
//
// Here a block is RG row groups x CT column tiles. Thread (rg, ct) owns
// tile-rows [rg*ROWS, rg*ROWS+ROWS) of column tile ct, which in the V8 layout
// is exactly states [rg*STATES, rg*STATES+STATES) - the same split the
// row-split split-path walk uses. CT is sized to cover tiles_y so no thread
// is idle in the walk, and the trellis tile row is staged cooperatively so
// the global reads are contiguous instead of 24 strided uint16 per thread.
//
// BIT-EXACT against mach1_exp_fused_kernel<P4, false>: the u stage is the
// same fwht_block (bit-identical for any block width by construction), each
// (tile-row, column-tile) partial is the same two states in the same order,
// and the reduction tree is unchanged - CT is a multiple of the warp size so
// a warp never straddles a row group, its butterfly covers the same 32
// column tiles, and the sequential warp combine that follows only ever
// dropped exact-zero terms from threads the old block left idle.
template <bool P4, int WG, int RG>
__global__ void mach1_exp_fused_rows_kernel(
        const uint16_t * __restrict__ kept,
        const void     * __restrict__ tlut,    // half [32768, 8]
        const uint32_t * __restrict__ p4,
        const float    * __restrict__ p4lv,
        const half     * __restrict__ su,
        const int32_t  * __restrict__ remap,
        const int32_t  * __restrict__ ids,
        const float    * __restrict__ x,
        float          * __restrict__ scr_v,
        const half     * __restrict__ gamma,
        const int m, const int n, const int xne1,
        const int n_used, const int n_kept, const int ids_s0, const int ids_s1) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int CT        = WG / RG;
    constexpr int n_warps   = CT / warp_size;
    constexpr int ROWS      = 16 / RG;
    constexpr int STATES    = 32 / RG;
    constexpr int words     = 24;              // V = 8, K = 1.5
    __shared__ float    shu[2048];             // n <= 2048 (supports_op)
    __shared__ float    red[RG][ROWS][CT/32];
    __shared__ float    lv[16];
    __shared__ uint16_t stre[CT*words];

    const int s   = blockIdx.z;
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;
    const int ct  = tid % CT;
    const int rg  = tid / CT;

    const int64_t  e = ids[s*ids_s0];
    const uint32_t g = mach1_group_of(remap, ids, s, n_used, n_kept, ids_s0, ids_s1);

    if constexpr (P4) {
        if (tid < 16) {
            lv[tid] = p4lv[tid];
        }
    }
    const int64_t xbase = (int64_t)(xne1 == 1 ? 0 : s*n);
    for (int i = tid; i < n; i += WG) {
        shu[i] = __half2float(su[e*n + i]) * x[xbase + i];
    }
    __syncthreads();
    mach1_fwht_block(shu, n, tid, WG);
    const float scn = __fsqrt_rn((float) n);
    for (int i = tid; i < n; i += WG) {
        shu[i] = __fdiv_rn(shu[i], scn);
    }

    const int     tiles_y = n / 16;
    const int     ntiles  = (m / 16)*tiles_y;
    const int64_t tbase   = ((int64_t) g*ntiles + (int64_t) tr*tiles_y)*words;
    for (int i = tid; i < tiles_y*words; i += WG) {
        stre[i] = kept[tbase + i];
    }
    __syncthreads();

    const bool have = ct < tiles_y;
    const int  Mb   = m / 16;
    float gsc = 0.0f;
    if (have) {
        const int wv = (tr + ct <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + ct)
                                                 : Mb + tiles_y - 2 - (tr + ct);
        gsc = __half2float(gamma[e*(Mb + tiles_y) + wv]);
    }

    float partial[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        partial[r] = 0.0f;
    }
    if (have) {
        const int     tw = ct*words;
        const float * ub = shu + ct*16;
#pragma unroll
        for (int sx = 0; sx < STATES; ++sx) {
            const int      i   = rg*STATES + sx;
            const int      bb  = 12*i;
            const int      wi  = bb >> 4;
            const int      o   = bb & 15;
            const int      wn  = wi + 1 < words ? wi + 1 : 0;
            const uint32_t w2  = ((uint32_t) stre[tw + wi] << 16) | (uint32_t) stre[tw + wn];
            const uint32_t reg = w2 >> (16 - o);
            const uint32_t ph  = reg*(reg + 1u);
            const uint32_t row = ph & 0x7FFFu;
            const int      ri  = sx >> 1;
            const int      ci  = (8*i) & 15;
            float dotp = 0.0f;
            if constexpr (P4) {
                const uint32_t pk = p4[row >> 1];
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    float vv = lv[(pk >> (4*c)) & 0xFu];
                    if (c == 0 && (ph & 0x8000u)) {
                        vv = -vv;                 // exact (sign bit)
                    }
                    dotp += (vv * gsc) * ub[ci + c];
                }
            } else {
                const half * tl = (const half *) tlut + 8*(int64_t) row;
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    float vv = __half2float(tl[c]);
                    if (c == 0 && (ph & 0x8000u)) {
                        vv = -vv;                 // exact in fp16 (sign bit)
                    }
                    dotp += (vv * gsc) * ub[ci + c];
                }
            }
            partial[ri] += dotp;
        }
    }

    const int lane = ct % warp_size;
    const int wid  = ct / warp_size;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const float sm = warp_reduce_sum<warp_size>(partial[r]);
        if (lane == 0) {
            red[rg][r][wid] = sm;
        }
    }
    __syncthreads();
    if (tid < 16) {
        const int gg = tid / ROWS;
        const int rr = tid % ROWS;
        float sum = 0.0f;
#pragma unroll
        for (int wj = 0; wj < n_warps; ++wj) {
            sum += red[gg][rr][wj];
        }
        scr_v[(int64_t) s*m + tr*16 + gg*ROWS + rr] = sum;
    }
}

// dense prefill: when P = n_used*n_tok is large, most pairs share a storage
// group and the fused walk re-decodes each group's trellis once per pair.
// Instead decode every ACTIVE group (cnt > 0) once into an fp16 bank
// [n_groups, m, n] and apply it with plain fp32 dots. The bank holds the
// exact hatWr halves of the fused path (V2 pins the fp16 round; V8 stores
// the pre-rounded tlut entries UNSCALED — wave gamma is applied in the
// apply kernel, folded into u per column tile, so no extra fp16 rounding
// enters the chain). exp_group/exp_u/exp_out are untouched.

template <bool V8, bool P4>
__global__ void mach1_exp_dense_decode_kernel(
        const uint16_t * __restrict__ kept,
        const uint16_t * __restrict__ dem,     // placeholder (= kept) when absent
        const void     * __restrict__ tlut,    // float (V2) or half (V8)
        const uint32_t * __restrict__ p4,      // P4 only, nullptr otherwise
        const float    * __restrict__ p4lv,    // P4 only, nullptr otherwise
        const int32_t  * __restrict__ scr_i,
        half           * __restrict__ bank,    // [n_groups, m, n]
        const int m, const int n, const int n_kept, const int n_groups) {
    static_assert(V8 || !P4, "P4 is a V8-tlut repack");
    constexpr int WG = 256;
    __shared__ float lv[16];

    (void) dem;
    (void) p4;
    (void) p4lv;
    (void) n_groups;

    const int g   = blockIdx.z;
    const int tid = threadIdx.x;

    if (scr_i[g] == 0) {           // block-uniform: skip inactive groups
        return;
    }
    if constexpr (P4) {
        if (tid < 16) {
            lv[tid] = p4lv[tid];
        }
        __syncthreads();
    }

    const int tiles_y = n / 16;
    const int gx = blockIdx.x*WG + tid;
    if (gx >= (m/16)*16*tiles_y) {   // one thread per (tile, tile-row)
        return;
    }
    // column tile fastest: adjacent threads write adjacent 16-half chunks
    const int tc     = gx % tiles_y;
    const int rowall = gx / tiles_y;
    const int tr     = rowall >> 4;
    const int rr     = rowall & 15;
    const int tile   = tr*tiles_y + tc;
    const int ntiles = (m/16)*tiles_y;

    int              words;
    int              stepb;
    const uint16_t * trd;
    int64_t          tbase;
    if constexpr (V8) {
        words = 24;
        stepb = 12;
        trd   = kept;
        tbase = ((int64_t) g*ntiles + tile)*words;
    } else {
        const bool is_dem = g >= n_kept;
        const int  ei     = is_dem ? g - n_kept : g;
        words = is_dem ? 16 : 32;
        stepb = is_dem ? 2 : 4;
        trd   = is_dem ? dem : kept;
        tbase = ((int64_t) ei*ntiles + tile)*words;
    }

    half * dst = bank + ((int64_t) g*m + tr*16 + rr)*n + tc*16;
    half tmp[16];

    if constexpr (V8) {
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {     // tile row rr = states 2*rr, 2*rr+1
            const int      i   = 2*rr + jj;
            const int      bb  = 12*i;
            const int      wi  = bb >> 4;
            const int      o   = bb & 15;
            const int      wn  = wi + 1 < 24 ? wi + 1 : 0;
            const uint32_t w2  = ((uint32_t) trd[tbase + wi] << 16) | (uint32_t) trd[tbase + wn];
            const uint32_t reg = (w2 >> (16 - o)) & 0xFFFFu;
            const uint32_t ph  = reg*(reg + 1u);
            const uint32_t row = ph & 0x7FFFu;
            if constexpr (P4) {
                const uint32_t pk = p4[row >> 1];
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    float vv = lv[(pk >> (4*c)) & 0xFu];
                    if (c == 0 && (ph & 0x8000u)) {
                        vv = -vv;
                    }
                    // vv is an exact fp16 value: the round-trip stores its bits
                    tmp[8*jj + c] = __float2half_rn(vv);
                }
            } else {
                const half * tl = (const half *) tlut + 8*(int64_t) row;
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    float vv = __half2float(tl[c]);
                    if (c == 0 && (ph & 0x8000u)) {
                        vv = -vv;                         // exact in fp16 (sign bit)
                    }
                    tmp[8*jj + c] = __float2half_rn(vv);
                }
            }
        }
    } else {
#pragma unroll
        for (int s = 0; s < 8; ++s) {        // tile row rr = steps 8*rr .. 8*rr+8
            const int      i   = 8*rr + s;
            const int      bb  = stepb*i;
            const int      wi  = bb >> 4;
            const int      o   = bb & 15;
            const int      wn  = wi + 1 < words ? wi + 1 : 0;
            const uint32_t w2  = ((uint32_t) trd[tbase + wi] << 16) | (uint32_t) trd[tbase + wn];
            const uint32_t reg = (w2 >> (16 - o)) & 0xFFFFu;
            const uint32_t ph  = reg*(reg + 1u);
            const uint32_t row = (ph >> 6) & 511u;
            const float * tl = (const float *) tlut;
            float v0 = tl[2*row + 0];
            const float v1 = tl[2*row + 1];
            if (ph & 0x8000u) {
                v0 = -v0;                                 // exact either side of the round
            }
            // hatWr is defined at fp16 precision — the fused path's pinned
            // round IS this store
            tmp[2*s + 0] = __float2half_rn(v0);
            tmp[2*s + 1] = __float2half_rn(v1);
        }
    }
    mach1_store_half16(dst, tmp);   // same bits, 2x16B instead of 16x2B
}

// grid (m/16, n_groups): each block owns 16 output rows of one group's bank
// tile and loops the group's pair list (group-ordered, so the 16 x n slice
// stays hot in L2 across pairs). V8 folds wave gamma into u per column tile
// (gamma is constant within a tile and W*(g*u) == (g*W)*u up to contractible
// fp32 association).
// Pairs are processed PC at a time: the 16-row W column slice is loaded into
// registers once per column iteration and reused across the chunk's pairs,
// amortizing the dominant bank traffic PC-fold (same trick as
// mach1_rt_apply_kernel's token chunk). Per-pair accumulation order over the
// columns is unchanged (same WG stride, same warp reduction), so numerics are
// identical to the unchunked kernel; the tail chunk clamps to the last pair
// and write-guards (duplicate compute, no duplicate store). PC=1 reproduces
// the unchunked kernel exactly; GGML_MACH1_EXP_PC=1 selects it (A/B knob).
template <bool V8, int PC>
__global__ void mach1_exp_apply_kernel(
        const half    * __restrict__ bank,     // [n_groups, m, n]
        const int32_t * __restrict__ scr_i,
        const float   * __restrict__ scr_u,    // [P, n]
        float         * __restrict__ scr_v,    // [P, m]
        const half    * __restrict__ gamma,    // V8 only, nullptr otherwise
        const int m, const int n, const int n_groups) {
    constexpr int WG        = 128;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = WG / warp_size;
    __shared__ float red[PC][16][4];   // >= n_warps for warp_size 32 (and 64 on HIP)
    __shared__ float gsh[128];         // V8: per-column-tile wave gamma (tiles_y <= 128)

    (void) gamma;

    const int g   = blockIdx.y;
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;

    const int cnt = scr_i[g];
    if (cnt == 0) {                // block-uniform: whole block exits together
        return;
    }
    const int off     = scr_i[n_groups + g];
    const int tiles_y = n / 16;

    if constexpr (V8) {
        // same closed-form wave index as the fused walk, per column tile
        const int e_orig = scr_i[2*n_groups + g];
        const int Mb     = m / 16;
        for (int i = tid; i < tiles_y; i += WG) {
            const int wv = (tr + i <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + i)
                                                   : Mb + tiles_y - 2 - (tr + i);
            gsh[i] = __half2float(gamma[(int64_t) e_orig*(Mb + tiles_y) + wv]);
        }
        __syncthreads();
    }

    const half * W = bank + ((int64_t) g*m + tr*16)*n;

    const int lane = tid % warp_size;
    const int wid  = tid / warp_size;

    for (int q0 = 0; q0 < cnt; q0 += PC) {
        int pidx[PC];
#pragma unroll
        for (int t = 0; t < PC; ++t) {
            // tail chunk: clamp (duplicate compute, write-guarded below)
            pidx[t] = scr_i[4*n_groups + off + min(q0 + t, cnt - 1)];
        }

        float acc[PC][16];
#pragma unroll
        for (int t = 0; t < PC; ++t) {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                acc[t][i] = 0.0f;
            }
        }
        // half2/float2 loads over column pairs (c0 even, so both columns are
        // in the same 16-wide gamma tile). Merging a pair's two terms into
        // one partial is a contractible fp32 reassociation, like the fused
        // walk's own ordering.
        for (int c0 = 2*tid; c0 < n; c0 += 2*WG) {
            float2 wc[16];
#pragma unroll
            for (int ri = 0; ri < 16; ++ri) {
                wc[ri] = __half22float2(*(const half2 *)(W + (int64_t) ri*n + c0));
            }
#pragma unroll
            for (int t = 0; t < PC; ++t) {
                float2 uc = *(const float2 *)(scr_u + (int64_t) pidx[t]*n + c0);
                if constexpr (V8) {
                    const float gv = gsh[c0 >> 4];
                    uc.x *= gv;
                    uc.y *= gv;
                }
#pragma unroll
                for (int ri = 0; ri < 16; ++ri) {
                    acc[t][ri] += wc[ri].x*uc.x + wc[ri].y*uc.y;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < PC; ++t) {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const float s = warp_reduce_sum<warp_size>(acc[t][i]);
                if (lane == 0) {
                    red[t][i][wid] = s;
                }
            }
        }
        __syncthreads();
        if (tid < PC*16) {
            const int t = tid >> 4;
            const int i = tid & 15;
            if (q0 + t < cnt) {
                float sum = 0.0f;
#pragma unroll
                for (int wj = 0; wj < n_warps; ++wj) {
                    sum += red[t][i][wj];
                }
                scr_v[(int64_t) pidx[t]*m + tr*16 + i] = sum;
            }
        }
        __syncthreads();   // red is reused by the next chunk
    }
}

// fused dense prefill (GGML_MACH1_EXP_FUSED_DENSE, V8 only): decode a
// [16, 512] slice of the group's weights into SHARED once and dot it against
// every pair of the group. The bank form above round-trips an fp16
// [n_groups, m, n] image through global - ~0.5 GB written and ~1 GB read per
// op at nt=256, the two fattest lines of the r258 prefill census - and this
// form never materializes it: the trellis is read once per (tile, slice),
// the decode amortizes over the group's pairs exactly like the bank did,
// and the apply reads weights from shared.
//
// The dispatch admits n == 512 only, so kb_n is always 1 and the partials
// buffer IS scr_v; the kb_n/blockIdx.y plumbing in the kernel is kept for a
// future k-sliced form.
//
// Decode values are the bank's bit for bit (same fp16 round, same sign
// path); the per-pair chunk loop, gamma fold, column stride and reduction
// tree are mach1_exp_apply_kernel's own.
template <bool P4, int PC>
__global__ void mach1_exp_fused_apply_kernel(
        const uint16_t * __restrict__ kept,
        const void     * __restrict__ tlut,    // half (V8)
        const uint32_t * __restrict__ p4,      // P4 only
        const float    * __restrict__ p4lv,    // P4 only
        const int32_t  * __restrict__ scr_i,
        const float    * __restrict__ scr_u,   // [P, n]
        float          * __restrict__ scr_p,   // [P, kb_n, m] (== scr_v at kb_n 1)
        const half     * __restrict__ gamma,
        const int m, const int n, const int n_groups, const int kb_n) {
    constexpr int WG        = 128;
    constexpr int BK        = 512;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = WG / warp_size;
    __shared__ half  swt[16*BK];
    __shared__ float red[PC][16][4];
    __shared__ float gsh[BK/16];
    __shared__ float lv[16];

    const int g   = blockIdx.z;
    const int kb  = blockIdx.y;
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;

    const int cnt = scr_i[g];
    if (cnt == 0) {                // block-uniform: whole block exits together
        return;
    }
    if constexpr (P4) {
        if (tid < 16) {
            lv[tid] = p4lv[tid];
        }
        __syncthreads();
    }

    const int tiles_y = n / 16;
    const int c0      = kb*BK;     // dispatch guarantees n % BK == 0
    const int ntiles  = (m/16)*tiles_y;

    // decode phase: 16 rows x BK columns = 16*(BK/8) states, once per block
    for (int sidx = tid; sidx < 16*(BK/8); sidx += WG) {
        const int rr   = sidx / (BK/8);
        const int col  = (sidx % (BK/8))*8;
        const int tc   = (c0 + col) >> 4;
        const int jj   = ((c0 + col) >> 3) & 1;
        const int tile = tr*tiles_y + tc;
        const int i    = 2*rr + jj;

        const int64_t  tbase = ((int64_t) g*ntiles + tile)*24;
        const int      bb    = 12*i;
        const int      wi    = bb >> 4;
        const int      o     = bb & 15;
        const int      wn    = wi + 1 < 24 ? wi + 1 : 0;
        const uint32_t w2    = ((uint32_t) kept[tbase + wi] << 16) | (uint32_t) kept[tbase + wn];
        const uint32_t reg   = (w2 >> (16 - o)) & 0xFFFFu;
        const uint32_t ph    = reg*(reg + 1u);
        const uint32_t row   = ph & 0x7FFFu;
        if constexpr (P4) {
            const uint32_t pk = p4[row >> 1];
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                float vv = lv[(pk >> (4*c)) & 0xFu];
                if (c == 0 && (ph & 0x8000u)) {
                    vv = -vv;
                }
                swt[rr*BK + col + c] = __float2half_rn(vv);
            }
        } else {
            const half * tl = (const half *) tlut + 8*(int64_t) row;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                float vv = __half2float(tl[c]);
                if (c == 0 && (ph & 0x8000u)) {
                    vv = -vv;
                }
                swt[rr*BK + col + c] = __float2half_rn(vv);
            }
        }
    }

    // wave gamma per column tile of the slice (the apply kernel's closed form)
    {
        const int e_orig = scr_i[2*n_groups + g];
        const int Mb     = m / 16;
        for (int i = tid; i < BK/16; i += WG) {
            const int ti = (c0 >> 4) + i;
            const int wv = (tr + ti <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + ti)
                                                    : Mb + tiles_y - 2 - (tr + ti);
            gsh[i] = __half2float(gamma[(int64_t) e_orig*(Mb + tiles_y) + wv]);
        }
    }
    __syncthreads();

    const int off  = scr_i[n_groups + g];
    const int lane = tid % warp_size;
    const int wid  = tid / warp_size;

    for (int q0 = 0; q0 < cnt; q0 += PC) {
        int pidx[PC];
#pragma unroll
        for (int t = 0; t < PC; ++t) {
            pidx[t] = scr_i[4*n_groups + off + min(q0 + t, cnt - 1)];
        }

        float acc[PC][16];
#pragma unroll
        for (int t = 0; t < PC; ++t) {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                acc[t][i] = 0.0f;
            }
        }
        for (int c = 2*tid; c < BK; c += 2*WG) {
            float2 wc[16];
#pragma unroll
            for (int ri = 0; ri < 16; ++ri) {
                wc[ri] = __half22float2(*(const half2 *)(swt + ri*BK + c));
            }
            const float gv = gsh[c >> 4];
#pragma unroll
            for (int t = 0; t < PC; ++t) {
                float2 uc = *(const float2 *)(scr_u + (int64_t) pidx[t]*n + c0 + c);
                uc.x *= gv;
                uc.y *= gv;
#pragma unroll
                for (int ri = 0; ri < 16; ++ri) {
                    acc[t][ri] += wc[ri].x*uc.x + wc[ri].y*uc.y;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < PC; ++t) {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const float s = warp_reduce_sum<warp_size>(acc[t][i]);
                if (lane == 0) {
                    red[t][i][wid] = s;
                }
            }
        }
        __syncthreads();
        if (tid < PC*16) {
            const int t = tid >> 4;
            const int i = tid & 15;
            if (q0 + t < cnt) {
                float sum = 0.0f;
#pragma unroll
                for (int wj = 0; wj < n_warps; ++wj) {
                    sum += red[t][i][wj];
                }
                scr_p[((int64_t) pidx[t]*kb_n + kb)*m + tr*16 + i] = sum;
            }
        }
        __syncthreads();   // red and swt are reused by the next chunk
    }
}

// integer dense prefill (GGML_MACH1_EXP_ZDP_APPLY, V8+P4 with the ladder
// repack): the group's weights decode to Z-NIBBLE words in shared - 4 bits
// per weight, so the n = 2048 gate/up tile is 16 KB and needs NO k-slicing -
// and the apply runs the mega's integer-lattice arithmetic: u quantized to
// q8 per 16-column tile, two dp4a per state, the z+8 bias folded as -8*sumq,
// scales (zs0 * du/127 * gamma) associated outside the integer sum. Each
// thread owns whole column tiles, so the quantization is per (pair, tile)
// exactly like the mega's walk - the same certified numeric class.
//
// Against the fp16 bank apply this cuts the weight-side traffic 4x and packs
// 4 MACs per dp4a - the r262 census says the nt=512 apply is compute and L2
// bound, which is precisely the pair of axes the integer form shrinks.
// QONCE: the per (pair, tile) quantization hoisted to mach1_exp_quant_u_kernel
// - every row block re-quantized the same tile m/16 times (32x for gate/up),
// and the packed record is 24 B against the 64 B of fp32 u it replaces.
// Values identical: the same math, computed once.
// TPT (threads per tile): n = 512 has only 32 column tiles against 128
// threads, so TPT = 4 gives each thread a (tile, row-group) pair - and warp
// w then holds ALL 32 tiles of row-group w, so warp_reduce_sum yields the
// complete cross-tile row sums with no red array or block sync at all.
// LB: __launch_bounds__ min-blocks floor. The r266 ptxas print shows the
// kernel register-saturated at 255 with 2 blocks/SM; LB > 1 trades acc
// spills (L1-backed, MMQ-style) for residency - the cheap test of whether
// the 12x gap to the instruction floor is latency or bandwidth.
template <int PC, int QONCE = 0, int TPT = 1, int LB = 1>
__global__ void __launch_bounds__(128, LB) mach1_exp_zdp_apply_kernel(
        const uint16_t * __restrict__ kept,
        const uint32_t * __restrict__ zp4,     // [16384] nibble-packed z+8
        const float      zs0,
        const int32_t  * __restrict__ scr_i,
        const float    * __restrict__ scr_u,   // [P, n]
        const uint32_t * __restrict__ scr_q8,  // QONCE: [P*(n/16)][6]
        float          * __restrict__ scr_v,   // [P, m]
        const half     * __restrict__ gamma,
        const int m, const int n, const int n_groups) {
    constexpr int WG        = 128;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = WG / warp_size;
    constexpr int CTT       = WG / TPT;   // tile-owning threads per row-group
    constexpr int RPT       = 16 / TPT;   // rows per thread
    static_assert(TPT == 1 || WG/TPT == 32, "the TPT reduce rides warp = row-group");
    // [2 state-halves][16 rows][n/16 tiles]: splitting the even/odd states
    // makes the MAC's z-word load lane-stride 1 - with si = 2*ti + h the
    // joint layout was a 2-way bank conflict on every load. Same values,
    // same order, different addresses only.
    __shared__ uint32_t swz[2*16*128];    // n <= 2048
    __shared__ uint32_t ssgn[16*8];       // sign bit per state (n/8 <= 256), 32/word
    __shared__ float    red[PC][16][4];
    __shared__ float    gsh[128];         // per-column-tile wave gamma

    const int g   = blockIdx.y;
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;

    const int cnt = scr_i[g];
    if (cnt == 0) {
        return;
    }

    const int tiles_y = n / 16;
    const int ntiles  = (m/16)*tiles_y;

    for (int i = tid; i < 16*8; i += WG) {
        ssgn[i] = 0;
    }
    __syncthreads();

    // decode phase: 16 rows x n/8 states -> z-words + sign bits, once
    for (int sidx = tid; sidx < 16*(n/8); sidx += WG) {
        const int rr   = sidx / (n/8);
        const int si   = sidx % (n/8);     // state within the row: tile si/2, half si&1
        const int tc   = si >> 1;
        const int jj   = si & 1;
        const int tile = tr*tiles_y + tc;
        const int i    = 2*rr + jj;

        const int64_t  tbase = ((int64_t) g*ntiles + tile)*24;
        const int      bb    = 12*i;
        const int      wi    = bb >> 4;
        const int      o     = bb & 15;
        const int      wn    = wi + 1 < 24 ? wi + 1 : 0;
        const uint32_t w2    = ((uint32_t) kept[tbase + wi] << 16) | (uint32_t) kept[tbase + wn];
        const uint32_t reg   = (w2 >> (16 - o)) & 0xFFFFu;
        const uint32_t ph    = reg*(reg + 1u);
        const uint32_t row   = ph & 0x7FFFu;
        swz[((si & 1)*16 + rr)*128 + (si >> 1)] = zp4[row >> 1];
        if (ph & 0x8000u) {
            atomicOr(&ssgn[rr*8 + (si >> 5)], 1u << (si & 31));
        }
    }

    // wave gamma per column tile (the apply kernel's closed form)
    {
        const int e_orig = scr_i[2*n_groups + g];
        const int Mb     = m / 16;
        for (int i = tid; i < tiles_y; i += WG) {
            const int wv = (tr + i <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + i)
                                                   : Mb + tiles_y - 2 - (tr + i);
            gsh[i] = __half2float(gamma[(int64_t) e_orig*(Mb + tiles_y) + wv]);
        }
    }
    __syncthreads();

    const int off  = scr_i[n_groups + g];
    const int lane = tid % warp_size;
    const int wid  = tid / warp_size;

    for (int q0 = 0; q0 < cnt; q0 += PC) {
        int pidx[PC];
#pragma unroll
        for (int t = 0; t < PC; ++t) {
            pidx[t] = scr_i[4*n_groups + off + min(q0 + t, cnt - 1)];
        }

        float acc[PC][RPT];
#pragma unroll
        for (int t = 0; t < PC; ++t) {
#pragma unroll
            for (int i = 0; i < RPT; ++i) {
                acc[t][i] = 0.0f;
            }
        }
        // thread owns whole 16-column tiles (TPT > 1: of its row-group); per
        // (pair, tile) the mega's quantize-and-dp4a body runs on the z-words
        const int rg0 = TPT > 1 ? tid/CTT : 0;
        for (int ti = tid % CTT; ti < tiles_y; ti += CTT) {
            const float gv = gsh[ti];
#pragma unroll
            for (int t = 0; t < PC; ++t) {
                uint32_t qe[2], qo[2];
                int      qc0[2];
                float    du;
                int      sumq;
                if constexpr (QONCE) {
                    const uint32_t * qw = scr_q8 + ((int64_t) pidx[t]*tiles_y + ti)*6;
                    qe[0] = qw[0]; qo[0] = qw[1];
                    qe[1] = qw[2]; qo[1] = qw[3];
                    du    = __uint_as_float(qw[4]);
                    sumq  = (int) qw[5];
                    qc0[0] = (int)(signed char)(qe[0] & 0xFFu);
                    qc0[1] = (int)(signed char)(qe[1] & 0xFFu);
                } else {
                float uu[16];
#pragma unroll
                for (int c = 0; c < 16; ++c) {
                    uu[c] = scr_u[(int64_t) pidx[t]*n + ti*16 + c];
                }
                du = 0.0f;
#pragma unroll
                for (int c = 0; c < 16; ++c) {
                    du = fmaxf(du, fabsf(uu[c]));
                }
                const float qsf = du > 0.0f ? 127.0f/du : 0.0f;
                int q[16];
#pragma unroll
                for (int c = 0; c < 16; ++c) {
                    q[c] = __float2int_rn(uu[c]*qsf);
                }
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    qe[h] = (uint32_t)(q[8*h + 0] & 0xFF) | ((uint32_t)(q[8*h + 2] & 0xFF) << 8)
                          | ((uint32_t)(q[8*h + 4] & 0xFF) << 16) | ((uint32_t)(q[8*h + 6] & 0xFF) << 24);
                    qo[h] = (uint32_t)(q[8*h + 1] & 0xFF) | ((uint32_t)(q[8*h + 3] & 0xFF) << 8)
                          | ((uint32_t)(q[8*h + 5] & 0xFF) << 16) | ((uint32_t)(q[8*h + 7] & 0xFF) << 24);
                    qc0[h] = q[8*h];
                }
                sumq = 0;
#pragma unroll
                for (int c = 0; c < 16; ++c) {
                    sumq += q[c];
                }
                }
                const float fsc = (zs0*gv)*(du*(1.0f/127.0f));
#pragma unroll
                for (int r = 0; r < RPT; ++r) {
                    const int ri = RPT*rg0 + r;
                    int ipart = 0;
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int      si = 2*ti + h;
                        const uint32_t zk = swz[(h*16 + ri)*128 + ti];
                        int d = ggml_cuda_dp4a((int)(zk & 0x0F0F0F0Fu), (int) qe[h],
                                ggml_cuda_dp4a((int)((zk >> 4) & 0x0F0F0F0Fu), (int) qo[h], 0));
                        if (ssgn[ri*8 + (si >> 5)] & (1u << (si & 31))) {
                            d -= 2*((int)(zk & 0xFu) - 8)*qc0[h];
                        }
                        ipart += d;
                    }
                    acc[t][r] += (float)(ipart - 8*sumq)*fsc;
                }
            }
        }
        if constexpr (TPT > 1) {
            // warp == row-group: the shuffle sum IS the full cross-tile sum
#pragma unroll
            for (int t = 0; t < PC; ++t) {
#pragma unroll
                for (int r = 0; r < RPT; ++r) {
                    const float s = warp_reduce_sum<warp_size>(acc[t][r]);
                    if (lane == 0 && q0 + t < cnt) {
                        scr_v[(int64_t) pidx[t]*m + tr*16 + RPT*rg0 + r] = s;
                    }
                }
            }
        } else {
#pragma unroll
        for (int t = 0; t < PC; ++t) {
#pragma unroll
            for (int i = 0; i < RPT; ++i) {
                const float s = warp_reduce_sum<warp_size>(acc[t][i]);
                if (lane == 0) {
                    red[t][i][wid] = s;
                }
            }
        }
        __syncthreads();
        if (tid < PC*16) {
            const int t = tid >> 4;
            const int i = tid & 15;
            if (q0 + t < cnt) {
                float sum = 0.0f;
#pragma unroll
                for (int wj = 0; wj < n_warps; ++wj) {
                    sum += red[t][i][wj];
                }
                scr_v[(int64_t) pidx[t]*m + tr*16 + i] = sum;
            }
        }
        __syncthreads();   // red is reused by the next chunk
        }
    }
}

// PP lane (GGML_MACH1_EXP_APPLY_MMA, requires the QONCE records): the zdp
// apply's MAC phase as m16n8k16 s8 MMA. The decode phase expands the z-nibbles
// to TRUE int8 weights (w = z - 8, state sign folded into w0), so the MMA's
// integer result equals the dp4a form's (nibble dot - 8*sumq + sign fix)
// exactly; the per-(pair, tile) scale fsc = zs0*gv*du/127 is applied on the
// i32 fragment per tile, so only the cross-tile float association differs
// from the dp4a form (token-exact/KLD lane, like the spine TC apply).
// NIB selects the shared-weight form: 1 = z-nibble words + sign plane (21 KB,
// 4 blocks/SM - wins on the n = 512 down shape), 0 = expanded int8 tile
// (33.4 KB, 2 blocks/SM but no per-tile expansion - wins at tiles_y = 128).
template <int NIB>
__global__ void __launch_bounds__(128) mach1_exp_zdp_mma_kernel(
        const uint16_t * __restrict__ kept,
        const uint32_t * __restrict__ zp4,
        const float      zs0,
        const int32_t  * __restrict__ scr_i,
        const uint32_t * __restrict__ scr_q8,   // [P*(n/16)][6]
        float          * __restrict__ scr_v,    // [P, m]
        const half     * __restrict__ gamma,
        const int m, const int n, const int n_groups,
        const int8_t   * __restrict__ hotmask,  // nullptr: all groups; else skip hot ones (GG path)
        const int probe = 0) {   // 1: stop after decode (stopwatch only)
    constexpr int WG = 128;
    // z-nibble words + a sign-folded first-byte plane instead of the expanded
    // int8 tile: 21 KB of shared against 33.4, so 4 blocks/SM instead of 2 -
    // the loop is latency-bound and lives on warps in flight, and the decode
    // phase's smem traffic halves. Expansion to s8 happens at A-fragment
    // build (XOR-8 plus s4 sign extension, ~11 instructions per register).
    __shared__ uint32_t    sZ[16][NIB ? 257 : 1];    // nibble word per (row, state);
    __shared__ signed char sC0[16][NIB ? 257 : 1];   // +1 pad breaks the row-bank tie
    __shared__ signed char sA[NIB ? 1 : 16][NIB ? 1 : 2048];  // expanded int8 form
    __shared__ float gsh[128];
    __shared__ int   spidx[32];

    const int g   = blockIdx.y;
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;

    const int cnt = scr_i[g];
    if (cnt == 0 || (hotmask != nullptr && hotmask[g] != 0)) {
        return;
    }
    const int tiles_y = n / 16;
    const int ntiles  = (m/16)*tiles_y;

    // decode + expand: identical trellis indexing to the zdp decode phase,
    // but each state's 8 nibbles land as true int8 (sign folded into w0)
    for (int sidx = tid; sidx < 16*(n/8); sidx += WG) {
        const int rr = sidx / (n/8);
        const int si = sidx % (n/8);
        const int tc = si >> 1;
        const int jj = si & 1;
        const int tile = tr*tiles_y + tc;
        const int i    = 2*rr + jj;
        const int64_t  tbase = ((int64_t) g*ntiles + tile)*24;
        const int      bb    = 12*i;
        const int      wi    = bb >> 4;
        const int      o     = bb & 15;
        const int      wn    = wi + 1 < 24 ? wi + 1 : 0;
        const uint32_t w2    = ((uint32_t) kept[tbase + wi] << 16) | (uint32_t) kept[tbase + wn];
        const uint32_t reg   = (w2 >> (16 - o)) & 0xFFFFu;
        const uint32_t ph    = reg*(reg + 1u);
        const uint32_t zk    = zp4[(ph & 0x7FFFu) >> 1];
        if (NIB) {
            sZ[rr][si] = zk;
            int v0 = (int)(zk & 0xFu) - 8;
            if (ph & 0x8000u) {
                v0 = -v0;
            }
            sC0[rr][si] = (signed char) v0;
        } else {
            // expand to true int8 here, packed into two u32 stores
            uint32_t lo = 0, hi = 0;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                int v = (int)((zk >> (8*(c >> 1) + 4*(c & 1))) & 0xFu) - 8;
                if (c == 0 && (ph & 0x8000u)) {
                    v = -v;
                }
                const uint32_t bv = (uint32_t)(v & 0xFF);
                if (c < 4) {
                    lo |= bv << (8*c);
                } else {
                    hi |= bv << (8*(c - 4));
                }
            }
            uint32_t * dst = (uint32_t *) &sA[rr][tc*16 + 8*jj];
            dst[0] = lo;
            dst[1] = hi;
        }
    }
    {
        const int e_orig = scr_i[2*n_groups + g];
        const int Mb     = m / 16;
        for (int i = tid; i < tiles_y; i += WG) {
            const int wv = (tr + i <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + i)
                                                   : Mb + tiles_y - 2 - (tr + i);
            gsh[i] = __half2float(gamma[(int64_t) e_orig*(Mb + tiles_y) + wv]);
        }
    }
    const int off  = scr_i[n_groups + g];
    const int lane = tid % 32;
    const int wid  = tid / 32;

    if (probe == 1) {   // decode-floor stopwatch: publish a byte so the
        __syncthreads(); // expansion cannot be dead-code-eliminated
        if (tid == 0) {
            scr_v[(int64_t) scr_i[4*n_groups + off]*m + tr*16] = NIB
                ? (float)(int)(sZ[0][0] & 0xFu) + (float) sC0[0][0]
                : (float) sA[0][0];
        }
        return;
    }

    for (int q0 = 0; q0 < cnt; q0 += 32) {
        __syncthreads();
        if (tid < 32) {
            spidx[tid] = scr_i[4*n_groups + off + min(q0 + tid, cnt - 1)];
        }
        __syncthreads();

        // warp w owns pair columns qb..qb+7; a warp whose whole slot is past
        // cnt would compute clamped duplicates only - guard the work (NOT a
        // continue: every warp must still reach the loop-top barriers).
        // probe == 2 disables the skip for same-container A/B.
        const int qb = 8*wid;
        if (q0 + qb < cnt || probe == 2) {
        const int bcol = qb + lane/4;             // this lane's B pair slot
        const int64_t brec = ((int64_t) spidx[bcol]*tiles_y)*6;
        const int c0 = 4*(lane & 3);              // this lane's B k-bytes
        const int h  = c0 >> 3;                   // record half
        float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int ti = 0; ti < tiles_y; ++ti) {
            const uint32_t * qw = scr_q8 + brec + (int64_t) ti*6;
            const uint32_t qe = qw[2*h + 0];
            const uint32_t qo = qw[2*h + 1];
            // bytes c0..c0+3 in column order: even cols from qe, odd from qo
            const int bsel = (c0 & 7) >> 1;       // 0 or 2: starting byte pair
            mach1_tc_frag_bi8 b;
            b.r = __byte_perm(qe >> (8*bsel), qo >> (8*bsel), 0x5140);
            const float duv = __uint_as_float(qw[4]);

            // expand this lane's 4 columns from the nibble word: cols 0-3 of
            // a state sit in the low 16 bits (4-bit stride), 4-7 in the high;
            // XOR 8 turns nibble n into n-8 as s4, then sign-extend to s8 and
            // patch the state's true first byte (sign-folded) when present
            const int si = 2*ti + h;
            const int hh = (lane & 1);            // c0 = 4*(lane&3): half select
            mach1_tc_frag_ai8 a;
            if (NIB) {
#pragma unroll
            for (int rp = 0; rp < 2; ++rp) {
                const int r = (lane >> 2) + 8*rp;
                const uint32_t x = (sZ[r][si] >> (16*hh)) & 0xFFFFu;
                uint32_t y = (x & 0xFu) | ((x & 0xF0u) << 4)
                           | ((x & 0xF00u) << 8) | ((x & 0xF000u) << 12);
                y ^= 0x08080808u;
                y |= ((y >> 3) & 0x01010101u)*0xF0u;
                if (hh == 0) {
                    y = (y & 0xFFFFFF00u) | (uint32_t)(uint8_t) sC0[r][si];
                }
                a.r[rp] = y;
            }
            } else {
                a.r[0] = *(const unsigned *)&sA[lane >> 2][ti*16 + c0];
                a.r[1] = *(const unsigned *)&sA[(lane >> 2) + 8][ti*16 + c0];
            }

            mach1_tc_frag_ci8 c = {{0, 0, 0, 0}};
            mach1_tc_mma_i8(c, a, b);

            // per-element scale: du of the C element's own pair column
            const float base = zs0*gsh[ti]*(1.0f/127.0f);
            const float du0  = __shfl_sync(0xFFFFFFFFu, duv, 8*(lane & 3), 32);
            const float du1  = __shfl_sync(0xFFFFFFFFu, duv, 8*(lane & 3) + 4, 32);
            facc[0] += (float) c.r[0] * (base*du0);
            facc[1] += (float) c.r[1] * (base*du1);
            facc[2] += (float) c.r[2] * (base*du0);
            facc[3] += (float) c.r[3] * (base*du1);
        }

        const int r0 = lane >> 2;
        const int cc = 2*(lane & 3);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int col = qb + cc + (e & 1);
            const int row = r0 + (e >> 1)*8;
            if (q0 + col < cnt) {
                scr_v[(int64_t) spidx[col]*m + tr*16 + row] = facc[e];
            }
        }
        }
    }
}

// fp16-A form of the zdp apply (GGML_MACH1_EXP_APPLY_FP16): A expands from
// the nibble words through a 256-entry half2 LUT (one lookup per fragment
// register - a byte's two nibbles ARE the register's two columns), B reads
// straight from a fp16 copy of scr_u (two aligned u32 loads per lane), and
// the m16n8k16 f16 MMA accumulates fp32. The q8 records, byte_perm and du
// shuffles of the s8 form vanish; the epilogue scale is warp-uniform
// zs0*gv[ti]. ~26.5 KB shared -> 3 blocks/SM. Numeric class: fp16 weight x
// fp16 activation (ulp vs the integer lattice) - token receipt + KLD lane.
__global__ void __launch_bounds__(128) mach1_exp_zdp_fp16_kernel(
        const uint16_t * __restrict__ kept,
        const uint32_t * __restrict__ zp4,
        const float      zs0,
        const int32_t  * __restrict__ scr_i,
        const half     * __restrict__ u16,      // [P, n] fp16 copy of scr_u
        float          * __restrict__ scr_v,    // [P, m]
        const half     * __restrict__ gamma,
        const int m, const int n, const int n_groups,
        const int8_t   * __restrict__ hotmask,  // nullptr: all groups; else skip hot ones (GG path)
        const int probe = 0) {                  // 1: stop after decode (stopwatch only)
    constexpr int WG = 128;
    __shared__ uint32_t sZ[16][257];
    __shared__ half     sC0[16][257];   // sign-folded true first weight, half
    __shared__ half2    slut[256];      // byte -> {lo nibble - 8, hi nibble - 8}
    __shared__ float    gsh[128];
    __shared__ int      spidx[32];

    const int g   = blockIdx.y;
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;

    const int cnt = scr_i[g];
    if (cnt == 0 || (hotmask != nullptr && hotmask[g] != 0)) {
        return;
    }
    const int tiles_y = n / 16;
    const int ntiles  = (m/16)*tiles_y;

    for (int i = tid; i < 256; i += WG) {
        slut[i] = __floats2half2_rn((float)(i & 0xF) - 8.0f, (float)(i >> 4) - 8.0f);
    }

    for (int sidx = tid; sidx < 16*(n/8); sidx += WG) {
        const int rr = sidx / (n/8);
        const int si = sidx % (n/8);
        const int tc = si >> 1;
        const int jj = si & 1;
        const int tile = tr*tiles_y + tc;
        const int i    = 2*rr + jj;
        const int64_t  tbase = ((int64_t) g*ntiles + tile)*24;
        const int      bb    = 12*i;
        const int      wi    = bb >> 4;
        const int      o     = bb & 15;
        const int      wn    = wi + 1 < 24 ? wi + 1 : 0;
        const uint32_t w2    = ((uint32_t) kept[tbase + wi] << 16) | (uint32_t) kept[tbase + wn];
        const uint32_t reg   = (w2 >> (16 - o)) & 0xFFFFu;
        const uint32_t ph    = reg*(reg + 1u);
        const uint32_t zk    = zp4[(ph & 0x7FFFu) >> 1];
        sZ[rr][si] = zk;
        int v0 = (int)(zk & 0xFu) - 8;
        if (ph & 0x8000u) {
            v0 = -v0;
        }
        sC0[rr][si] = __int2half_rn(v0);
    }
    {
        const int e_orig = scr_i[2*n_groups + g];
        const int Mb     = m / 16;
        for (int i = tid; i < tiles_y; i += WG) {
            const int wv = (tr + i <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + i)
                                                   : Mb + tiles_y - 2 - (tr + i);
            gsh[i] = __half2float(gamma[(int64_t) e_orig*(Mb + tiles_y) + wv]);
        }
    }
    const int off  = scr_i[n_groups + g];
    const int lane = tid % 32;
    const int wid  = tid / 32;

    if (probe == 1) {   // decode-floor stopwatch: publish a byte so the
        __syncthreads(); // expansion cannot be dead-code-eliminated
        if (tid == 0) {
            scr_v[(int64_t) scr_i[4*n_groups + off]*m + tr*16] =
                (float)(int)(sZ[0][0] & 0xFu) + __half2float(sC0[0][0]);
        }
        return;
    }

    for (int q0 = 0; q0 < cnt; q0 += 32) {
        __syncthreads();
        if (tid < 32) {
            spidx[tid] = scr_i[4*n_groups + off + min(q0 + tid, cnt - 1)];
        }
        __syncthreads();

        const int qb = 8*wid;
        if (q0 + qb < cnt) {
        const int64_t bu = (int64_t) spidx[qb + lane/4]*n;   // this lane's B pair row
        const int jb = lane & 3;
        float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        // B rides one tile ahead: two clean global loads with no dependent
        // chain (unlike the s8 record pipeline, which lost - see the ledger)
        unsigned bn0 = *(const unsigned *)&u16[bu + 2*jb];
        unsigned bn1 = *(const unsigned *)&u16[bu + 8 + 2*jb];
        for (int ti = 0; ti < tiles_y; ++ti) {
            const int si0 = 2*ti;
            const int r0r = lane >> 2;
            mach1_tc_frag_a a;
            const uint32_t w0lo = sZ[r0r][si0],     w0hi = sZ[r0r][si0 + 1];
            const uint32_t w8lo = sZ[r0r + 8][si0], w8hi = sZ[r0r + 8][si0 + 1];
            half2 h0 = slut[(w0lo >> (8*jb)) & 0xFFu];
            half2 h1 = slut[(w8lo >> (8*jb)) & 0xFFu];
            half2 h2 = slut[(w0hi >> (8*jb)) & 0xFFu];
            half2 h3 = slut[(w8hi >> (8*jb)) & 0xFFu];
            if (jb == 0) {   // the state's first column carries the sign fold
                h0 = __halves2half2(sC0[r0r][si0],         __high2half(h0));
                h1 = __halves2half2(sC0[r0r + 8][si0],     __high2half(h1));
                h2 = __halves2half2(sC0[r0r][si0 + 1],     __high2half(h2));
                h3 = __halves2half2(sC0[r0r + 8][si0 + 1], __high2half(h3));
            }
            a.r[0] = *(const unsigned *)&h0;
            a.r[1] = *(const unsigned *)&h1;
            a.r[2] = *(const unsigned *)&h2;
            a.r[3] = *(const unsigned *)&h3;

            mach1_tc_frag_b b;
            b.r[0] = bn0;
            b.r[1] = bn1;
            if (ti + 1 < tiles_y) {
                const half * un = u16 + bu + (int64_t)(ti + 1)*16;
                bn0 = *(const unsigned *)&un[2*jb];
                bn1 = *(const unsigned *)&un[8 + 2*jb];
            }

            mach1_tc_frag_c c = {{0.0f, 0.0f, 0.0f, 0.0f}};
            mach1_tc_mma(c, a, b);

            const float base = zs0*gsh[ti];
            facc[0] += c.f[0]*base;
            facc[1] += c.f[1]*base;
            facc[2] += c.f[2]*base;
            facc[3] += c.f[3]*base;
        }

        const int r0 = lane >> 2;
        const int cc = 2*(lane & 3);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int col = qb + cc + (e & 1);
            const int row = r0 + (e >> 1)*8;
            if (q0 + col < cnt) {
                scr_v[(int64_t) spidx[col]*m + tr*16 + row] = facc[e];
            }
        }
        }
    }
}

// grouped-GEMM form of the zdp apply (GGML_MACH1_EXP_APPLY_GG): each HOT
// group's weights decode ONCE into a transient [m, n] fp16 bank with zs0*gv
// folded in (gv varies per 16-column tile, so it must live in the bank, not
// an epilogue alpha), and the apply becomes a plain cuBLAS GEMM over the
// group's gathered activation rows. Same trellis indexing as the fused
// kernels above; same fp16 numeric class (ulp vs the fused epilogue scale).
__global__ void mach1_exp_gg_decode_kernel(
        const uint16_t * __restrict__ kept,
        const uint32_t * __restrict__ zp4,
        const float      zs0,
        const int32_t  * __restrict__ scr_i,
        const int32_t  * __restrict__ meta,     // [0..n_hot): group of each hot slot
        half           * __restrict__ bank,     // [n_hot, m, n] row-major
        const half     * __restrict__ gamma,
        const int m, const int n, const int n_groups) {
    constexpr int WG = 128;
    const int hs = blockIdx.y;
    const int tr = blockIdx.x;
    const int g  = meta[hs];
    const int tiles_y = n / 16;
    const int ntiles  = (m/16)*tiles_y;
    const int Mb      = m / 16;
    const int e_orig  = scr_i[2*n_groups + g];
    half * bg = bank + (int64_t) hs*m*n;
    for (int sidx = threadIdx.x; sidx < 16*(n/8); sidx += WG) {
        const int rr = sidx / (n/8);
        const int si = sidx % (n/8);
        const int tc = si >> 1;
        const int jj = si & 1;
        const int tile = tr*tiles_y + tc;
        const int i    = 2*rr + jj;
        const int64_t  tbase = ((int64_t) g*ntiles + tile)*24;
        const int      bb    = 12*i;
        const int      wi    = bb >> 4;
        const int      o     = bb & 15;
        const int      wn    = wi + 1 < 24 ? wi + 1 : 0;
        const uint32_t w2    = ((uint32_t) kept[tbase + wi] << 16) | (uint32_t) kept[tbase + wn];
        const uint32_t reg   = (w2 >> (16 - o)) & 0xFFFFu;
        const uint32_t ph    = reg*(reg + 1u);
        const uint32_t zk    = zp4[(ph & 0x7FFFu) >> 1];
        const int wv = (tr + tc <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + tc)
                                                : Mb + tiles_y - 2 - (tr + tc);
        const float sc = zs0*__half2float(gamma[(int64_t) e_orig*(Mb + tiles_y) + wv]);
        union { half h[8]; int4 i4; } hv;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            int v = (int)((zk >> (8*(c >> 1) + 4*(c & 1))) & 0xFu) - 8;
            if (c == 0 && (ph & 0x8000u)) {
                v = -v;
            }
            hv.h[c] = __float2half_rn(sc*(float) v);
        }
        *(int4 *)(bg + (int64_t)(tr*16 + rr)*n + tc*16 + jj*8) = hv.i4;
    }
}

// meta layout (host-built): [0..n_hot) group, [n_hot..2n_hot) pair-list off,
// [2n_hot..3n_hot) padded row offset, [3n_hot..4n_hot) cnt,
// [4n_hot..4n_hot+rows_pad) row -> hot slot
__global__ void mach1_exp_gg_gather_kernel(
        const half    * __restrict__ u16,       // [P, n] fp16 copy of scr_u
        const int32_t * __restrict__ scr_i,
        const int32_t * __restrict__ meta,
        half          * __restrict__ ub,        // [rows_pad, n]
        const int n, const int n_groups, const int n_hot) {
    const int r  = blockIdx.x;
    const int hs = meta[4*n_hot + r];
    const int q  = r - meta[2*n_hot + hs];
    half2 * dst = (half2 *)(ub + (int64_t) r*n);
    if (q >= meta[3*n_hot + hs]) {              // bucket pad row: zeros
        for (int i = threadIdx.x; i < n/2; i += blockDim.x) {
            dst[i] = __float2half2_rn(0.0f);
        }
        return;
    }
    const int64_t p = scr_i[4*n_groups + meta[n_hot + hs] + q];
    const half2 * src = (const half2 *)(u16 + p*n);
    for (int i = threadIdx.x; i < n/2; i += blockDim.x) {
        dst[i] = src[i];
    }
}

__global__ void mach1_exp_gg_scatter_kernel(
        const float   * __restrict__ cb,        // [rows_pad, m] GEMM outputs
        const int32_t * __restrict__ scr_i,
        const int32_t * __restrict__ meta,
        float         * __restrict__ scr_v,     // [P, m]
        const int m, const int n_groups, const int n_hot) {
    const int r  = blockIdx.x;
    const int hs = meta[4*n_hot + r];
    const int q  = r - meta[2*n_hot + hs];
    if (q >= meta[3*n_hot + hs]) {
        return;
    }
    const int64_t p = scr_i[4*n_groups + meta[n_hot + hs] + q];
    const float * src = cb + (int64_t) r*m;
    float * dst = scr_v + p*m;
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// int8 form of the GG bank decode (GGML_MACH1_GG_I8): gamma varies per
// k-tile so it cannot be a GEMM epilogue alpha - it folds into the int8
// lattice against the row tile's gamma bound instead (|w| <= 8 keeps every
// code within +-127); wsc keeps zs0*8*gmax/127 for the scatter epilogue.
// KLD class: the gamma fold rounds weights, the B rows quantize per row.
__global__ void mach1_exp_gg_decode_i8_kernel(
        const uint16_t * __restrict__ kept,
        const uint32_t * __restrict__ zp4,
        const float      zs0,
        const int32_t  * __restrict__ scr_i,
        const int32_t  * __restrict__ meta,     // [0..n_hot): group of each hot slot
        int8_t         * __restrict__ bank,     // [n_hot, m, n] row-major
        float          * __restrict__ wsc,      // [n_hot, m/16]
        const half     * __restrict__ gamma,
        const int m, const int n, const int n_groups) {
    constexpr int WG = 128;
    __shared__ float gsh[128];
    __shared__ float gred[WG/32];
    const int hs  = blockIdx.y;
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;
    const int g   = meta[hs];
    const int tiles_y = n / 16;
    const int ntiles  = (m/16)*tiles_y;
    const int Mb      = m / 16;
    const int e_orig  = scr_i[2*n_groups + g];
    float lm = 0.0f;
    for (int i = tid; i < tiles_y; i += WG) {
        const int wv = (tr + i <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + i)
                                               : Mb + tiles_y - 2 - (tr + i);
        const float gv = __half2float(gamma[(int64_t) e_orig*(Mb + tiles_y) + wv]);
        gsh[i] = gv;
        lm = fmaxf(lm, fabsf(gv));
    }
    lm = warp_reduce_max(lm);
    if (tid % 32 == 0) {
        gred[tid/32] = lm;
    }
    __syncthreads();
    float gmax = gred[0];
#pragma unroll
    for (int w = 1; w < WG/32; ++w) {
        gmax = fmaxf(gmax, gred[w]);
    }
    const float qs0 = gmax > 0.0f ? (127.0f/8.0f)/gmax : 0.0f;
    if (tid == 0) {
        wsc[(int64_t) hs*(m/16) + tr] = zs0*8.0f*gmax/127.0f;
    }
    int8_t * bg = bank + (int64_t) hs*m*n;
    for (int sidx = tid; sidx < 16*(n/8); sidx += WG) {
        const int rr = sidx / (n/8);
        const int si = sidx % (n/8);
        const int tc = si >> 1;
        const int jj = si & 1;
        const int tile = tr*tiles_y + tc;
        const int i    = 2*rr + jj;
        const int64_t  tbase = ((int64_t) g*ntiles + tile)*24;
        const int      bb    = 12*i;
        const int      wi    = bb >> 4;
        const int      o     = bb & 15;
        const int      wn    = wi + 1 < 24 ? wi + 1 : 0;
        const uint32_t w2    = ((uint32_t) kept[tbase + wi] << 16) | (uint32_t) kept[tbase + wn];
        const uint32_t reg   = (w2 >> (16 - o)) & 0xFFFFu;
        const uint32_t ph    = reg*(reg + 1u);
        const uint32_t zk    = zp4[(ph & 0x7FFFu) >> 1];
        const float qs = gsh[tc]*qs0;
        union { int8_t b[8]; uint2 u2; } qv;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            int v = (int)((zk >> (8*(c >> 1) + 4*(c & 1))) & 0xFu) - 8;
            if (c == 0 && (ph & 0x8000u)) {
                v = -v;
            }
            qv.b[c] = (int8_t) __float2int_rn(qs*(float) v);
        }
        *(uint2 *)(bg + (int64_t)(tr*16 + rr)*n + tc*16 + jj*8) = qv.u2;
    }
}

// int8 B rows (GGML_MACH1_GG_I8): gather + symmetric per-row quantization in
// one pass (absmax then codes); bsc keeps du/127 for the scatter epilogue
__global__ void mach1_exp_gg_gather_i8_kernel(
        const float   * __restrict__ scr_u,     // [P, n]
        const int32_t * __restrict__ scr_i,
        const int32_t * __restrict__ meta,
        int8_t        * __restrict__ ub,        // [rows_pad, n]
        float         * __restrict__ bsc,      // [rows_pad]
        const int n, const int n_groups, const int n_hot) {
    constexpr int WG = 256;
    const int r  = blockIdx.x;
    const int hs = meta[4*n_hot + r];
    const int q  = r - meta[2*n_hot + hs];
    char4 * dst = (char4 *)(ub + (int64_t) r*n);
    if (q >= meta[3*n_hot + hs]) {              // bucket pad row: zeros
        for (int i = threadIdx.x; i < n/4; i += WG) {
            dst[i] = make_char4(0, 0, 0, 0);
        }
        if (threadIdx.x == 0) {
            bsc[r] = 0.0f;
        }
        return;
    }
    __shared__ float red[WG/32];
    __shared__ float du_sh;
    const int64_t p = scr_i[4*n_groups + meta[n_hot + hs] + q];
    const float4 * src = (const float4 *)(scr_u + p*n);
    float lm = 0.0f;
    for (int i = threadIdx.x; i < n/4; i += WG) {
        const float4 v = src[i];
        lm = fmaxf(lm, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    lm = warp_reduce_max(lm);
    if (threadIdx.x % 32 == 0) {
        red[threadIdx.x/32] = lm;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float du = red[0];
#pragma unroll
        for (int w = 1; w < WG/32; ++w) {
            du = fmaxf(du, red[w]);
        }
        du_sh = du;
        bsc[r] = du/127.0f;
    }
    __syncthreads();
    const float qsf = du_sh > 0.0f ? 127.0f/du_sh : 0.0f;
    for (int i = threadIdx.x; i < n/4; i += WG) {
        const float4 v = src[i];
        dst[i] = make_char4((int8_t) __float2int_rn(v.x*qsf), (int8_t) __float2int_rn(v.y*qsf),
                            (int8_t) __float2int_rn(v.z*qsf), (int8_t) __float2int_rn(v.w*qsf));
    }
}

__global__ void mach1_exp_gg_scatter_i8_kernel(
        const int32_t * __restrict__ cb,        // [rows_pad, m] i32 GEMM outputs
        const int32_t * __restrict__ scr_i,
        const int32_t * __restrict__ meta,
        const float   * __restrict__ wsc,       // [n_hot, m/16]
        const float   * __restrict__ bsc,       // [rows_pad]
        float         * __restrict__ scr_v,     // [P, m]
        const int m, const int n_groups, const int n_hot) {
    const int r  = blockIdx.x;
    const int hs = meta[4*n_hot + r];
    const int q  = r - meta[2*n_hot + hs];
    if (q >= meta[3*n_hot + hs]) {
        return;
    }
    const int64_t p = scr_i[4*n_groups + meta[n_hot + hs] + q];
    const int32_t * src = cb + (int64_t) r*m;
    const float   * ws  = wsc + (int64_t) hs*(m/16);
    const float     bs  = bsc[r];
    float * dst = scr_v + p*m;
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        dst[i] = (float) src[i]*(ws[i >> 4]*bs);
    }
}

// one q8 record per (pair, 16-column tile): qe0 qo0 qe1 qo1 du sumq - the
// mega's packing, computed once instead of once per row block
__global__ void mach1_exp_quant_u_kernel(
        const float * __restrict__ scr_u,    // [P, n]
        uint32_t    * __restrict__ scr_q8,   // [P*(n/16)][6]
        const int n, const int64_t total) {  // total = P*(n/16)
    const int64_t gid = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (gid >= total) {
        return;
    }
    const int     T  = n/16;
    const int64_t p  = gid / T;
    const int     ti = (int)(gid - p*T);

    float uu[16];
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        uu[c] = scr_u[p*n + ti*16 + c];
    }
    float du = 0.0f;
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        du = fmaxf(du, fabsf(uu[c]));
    }
    const float qsf = du > 0.0f ? 127.0f/du : 0.0f;
    int q[16];
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        q[c] = __float2int_rn(uu[c]*qsf);
    }
    int sumq = 0;
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        sumq += q[c];
    }
    uint32_t * o = scr_q8 + gid*6;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        o[2*h + 0] = (uint32_t)(q[8*h + 0] & 0xFF) | ((uint32_t)(q[8*h + 2] & 0xFF) << 8)
                   | ((uint32_t)(q[8*h + 4] & 0xFF) << 16) | ((uint32_t)(q[8*h + 6] & 0xFF) << 24);
        o[2*h + 1] = (uint32_t)(q[8*h + 1] & 0xFF) | ((uint32_t)(q[8*h + 3] & 0xFF) << 8)
                   | ((uint32_t)(q[8*h + 5] & 0xFF) << 16) | ((uint32_t)(q[8*h + 7] & 0xFF) << 24);
    }
    o[4] = __float_as_uint(du);
    o[5] = (uint32_t) sumq;
}

template <int WG>
__global__ void mach1_exp_out_kernel(
        const half    * __restrict__ sv,
        const int32_t * __restrict__ ids,
        const float   * __restrict__ scr_v,
        float         * __restrict__ dst,
        const int m, const int n_used, const int ids_s0, const int ids_s1) {
    __shared__ float sh[2048];

    const int s   = blockIdx.x;
    const int t   = blockIdx.y;
    const int p   = t*n_used + s;
    const int tid = threadIdx.x;

    const int64_t e = ids[s*ids_s0 + t*ids_s1];

    for (int i = tid; i < m; i += WG) {
        sh[i] = scr_v[(int64_t) p*m + i];
    }
    __syncthreads();
    mach1_fwht_block(sh, m, tid, WG);
    const float sc = __fsqrt_rn((float) m);
    const int64_t obase = (int64_t) t*n_used*m + (int64_t) s*m;
    for (int i = tid; i < m; i += WG) {
        dst[obase + i] = __fdiv_rn(sh[i], sc) * __half2float(sv[e*m + i]);
    }
}

// Persistent from-codes expert op (GGML_MACH1_EXP_PERSIST=1; V8 + P4, decode
// only). ONE launch replaces the fused-walk + exp_out pair, and the codebook
// gather moves off the LSU path:
//  * the p4 nibble codebook (128 KB; lossless - the fp16 tlut is a 10-level
//    integer lattice) is staged into dynamic shared once per block, so every
//    state gather is an LDS.32 on shared ports instead of the measured
//    LSU-address-bound L2 gather (0.61 ms/token). Round 26 falsified shared
//    residency for TRANSIENT blocks - 4096 gathers against 32768 staged
//    words; a block that serves the whole op amortizes the staging, and the
//    table streams from L2, not HBM.
//  * the grid is one 1024-thread block per SM, a container of 8 virtual
//    blocks of 128 threads whose walk body, accumulation order and reduction
//    tree are EXACTLY mach1_exp_fused_kernel's (FAST body, which is
//    bit-exact against the default body) - so the output is bit-identical.
//  * x' = H(su_e .* x)/sqrt(n) for all used slots is computed once per block
//    (FWHT values do not depend on thread count), not once per (slot, tile
//    row) block.
//  * the exp_out stage folds in: the last vblock to finish a slot (device
//    scope counter, threadfence pattern) runs sv .* H_m(y_s)/sqrt(m) via the
//    named-barrier FWHT - same butterflies, one launch fewer per op.
__launch_bounds__(1024, 1)
static __global__ void mach1_exp_persist_kernel(
        const uint16_t * __restrict__ kept,
        const uint32_t * __restrict__ p4g,
        const float    * __restrict__ p4lv,
        const half     * __restrict__ su,
        const half     * __restrict__ sv,
        const int32_t  * __restrict__ remap,
        const int32_t  * __restrict__ ids,
        const float    * __restrict__ x,
        float          * __restrict__ scr_v,
        float          * __restrict__ dst,
        const half     * __restrict__ gamma,
        int32_t        * __restrict__ slot_done,
        const int m, const int n, const int xne1,
        const int n_used, const int n_kept, const int ids_s0, const int ids_s1) {
    constexpr int WGP       = 1024;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    extern __shared__ char smem[];
    uint32_t * p4s  = (uint32_t *) smem;                          // 64 KB
    float2   * lv2s = (float2 *)  (smem + 16384*sizeof(uint32_t));
    float    * xbuf = (float *)   (smem + 16384*sizeof(uint32_t) + 256*sizeof(float2));
    float    * obuf = xbuf + (size_t) n_used*n;

    __shared__ float red[8][16][4];
    __shared__ float lv[16];
    __shared__ int   olock;
    __shared__ int   vdone[8];

    const int tid = threadIdx.x;

    if (tid < 16) {
        lv[tid] = p4lv[tid];
    }
    if (tid == 0) {
        olock = 0;
    }
    for (int i = tid; i < 16384; i += WGP) {
        p4s[i] = p4g[i];
    }
    __syncthreads();
    for (int b = tid; b < 256; b += WGP) {
        lv2s[b] = make_float2(lv[b & 0xF], lv[(b >> 4) & 0xF]);
    }

    const float scn = __fsqrt_rn((float) n);
    for (int s = 0; s < n_used; ++s) {
        const int64_t e     = ids[s*ids_s0];
        const int64_t xbase = (int64_t)(xne1 == 1 ? 0 : s*n);
        float * xs = xbuf + (size_t) s*n;
        for (int i = tid; i < n; i += WGP) {
            xs[i] = __half2float(su[e*n + i]) * x[xbase + i];
        }
        __syncthreads();
        mach1_fwht_block(xs, n, tid, WGP);
        for (int i = tid; i < n; i += WGP) {
            xs[i] = __fdiv_rn(xs[i], scn);
        }
        __syncthreads();
    }

    const int vb   = tid >> 7;
    const int vtid = tid & 127;
    const int lane = vtid % warp_size;
    const int wid  = vtid / warp_size;
    const int bar  = vb + 1;

    const int  tiles_y = n / 16;
    const int  ntr     = m / 16;
    const int  ntiles  = ntr*tiles_y;
    const bool have    = vtid < tiles_y;
    const int  Mb      = ntr;
    constexpr int words = 24;   // V = 8, K = 1.5

    const int nvb  = gridDim.x*8;
    const int vgid = blockIdx.x*8 + vb;

    for (int it = vgid; it < n_used*ntr; it += nvb) {
        const int s  = it / ntr;
        const int tr = it % ntr;

        const int64_t  e = ids[s*ids_s0];
        const uint32_t g = mach1_group_of(remap, ids, s, n_used, n_kept, ids_s0, ids_s1);
        const int64_t tbase = ((int64_t) g*ntiles + (int64_t) tr*tiles_y)*words;

        float gsc = 0.0f;
        if (have) {
            const int wv = (tr + vtid <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + vtid)
                                                      : Mb + tiles_y - 2 - (tr + vtid);
            gsc = __half2float(gamma[e*(Mb + tiles_y) + wv]);
        }

        float partial[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            partial[i] = 0.0f;
        }
        if (have) {
            const int64_t tw = tbase + (int64_t) vtid*words;
            uint4 kv[3];
            {
                const uint4 * kb = (const uint4 *) (kept + tw);
#pragma unroll
                for (int q = 0; q < 3; ++q) {
                    kv[q] = kb[q];
                }
            }
            const uint16_t * w8 = (const uint16_t *) kv;
            const float * ub = xbuf + (size_t) s*n + vtid*16;
            float uu[16];
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                uu[c] = ub[c];
            }
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                const int      bb  = 12*i;
                const int      wi  = bb >> 4;
                const int      o   = bb & 15;
                const int      wn  = wi + 1 < 24 ? wi + 1 : 0;
                const uint32_t w2  = ((uint32_t) w8[wi] << 16) | (uint32_t) w8[wn];
                const uint32_t reg = w2 >> (16 - o);
                const uint32_t ph  = reg*(reg + 1u);
                const uint32_t row = ph & 0x7FFFu;
                const int      ri  = (8*i) >> 4;
                const int      ci  = (8*i) & 15;
                float dotp = 0.0f;
                const uint32_t pk = p4s[row >> 1];
#pragma unroll
                for (int c = 0; c < 8; c += 2) {
                    const float2 pr = lv2s[(pk >> (4*c)) & 0xFFu];
                    float vv = pr.x;
                    if (c == 0 && (ph & 0x8000u)) {
                        vv = -vv;                 // exact (sign bit)
                    }
                    dotp += (vv * gsc) * uu[ci + c];
                    dotp += (pr.y * gsc) * uu[ci + c + 1];
                }
                partial[ri] += dotp;
            }
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float s2 = warp_reduce_sum<warp_size>(partial[i]);
            if (lane == 0) {
                red[vb][i][wid] = s2;
            }
        }
        mach1_tile_sync(bar);
        if (vtid < 16) {
            float sum = 0.0f;
#pragma unroll
            for (int wj = 0; wj < 4; ++wj) {
                sum += red[vb][vtid][wj];
            }
            scr_v[(int64_t) s*m + tr*16 + vtid] = sum;
        }
        // publish this tile row, then the last vblock to finish the slot
        // runs the out stage (threadFenceReduction pattern, device scope)
        __threadfence();
        mach1_tile_sync(bar);
        if (vtid == 0) {
            vdone[vb] = atomicAdd(&slot_done[s], 1);
        }
        mach1_tile_sync(bar);
        if (vdone[vb] == ntr - 1) {
            if (vtid == 0) {
                while (atomicCAS(&olock, 0, 1) != 0) {
                    __nanosleep(64);
                }
            }
            mach1_tile_sync(bar);
            __threadfence();
            for (int i = vtid; i < m; i += 128) {
                obuf[i] = __ldcg(&scr_v[(int64_t) s*m + i]);
            }
            mach1_tile_sync(bar);
            mach1_fwht_tile(obuf, m, vtid, bar);
            const float scm = __fsqrt_rn((float) m);
            for (int i = vtid; i < m; i += 128) {
                dst[(int64_t) s*m + i] = __fdiv_rn(obuf[i], scm) * __half2float(sv[e*m + i]);
            }
            mach1_tile_sync(bar);
            if (vtid == 0) {
                atomicExch(&olock, 0);
            }
        }
    }
}

// Whole-layer expert FFN fusion (GGML_MACH1_EXP_FFN=1; V8 + P4, decode only):
// the per-layer expert subgraph [EXP_MM gate, EXP_MM up, swiglu, EXP_MM down,
// router-weight mul, 8 views + 7 adds] runs as TWO launches - the matcher is
// ggml_cuda_mach1_exp_ffn_fuse below. ~10 launches/nodes x ~2-4 us deleted
// per MoE layer, 40 layers/token.
//
// Launch A "gate|up": grid (m/16, 2, n_used), blockIdx.y picks the
// projection. The block body is EXACTLY mach1_exp_fused_kernel<true, true>
// with per-projection pointers, so the walk accumulation is bit-identical.
// The last of a slot's 2*(m/16) blocks (completion counter, the OFUSE
// threadfence pattern) runs BOTH out stages (sv .* H_m(v)/sqrt(m),
// mach1_exp_out_kernel's math) and the GLU node's swiglu_split
// (silu(gate)*up via ggml_cuda_op_silu_single, the inline the CUDA GLU
// kernel applies) and writes h - launch B's x.
__global__ void mach1_exp_ffn_gu_kernel(
        const uint16_t * __restrict__ kept_g,
        const uint16_t * __restrict__ kept_u,
        const uint32_t * __restrict__ p4,
        const float    * __restrict__ p4lv,
        const half     * __restrict__ su_g,
        const half     * __restrict__ su_u,
        const half     * __restrict__ sv_g,
        const half     * __restrict__ sv_u,
        const half     * __restrict__ gamma_g,
        const half     * __restrict__ gamma_u,
        const int32_t  * __restrict__ remap,
        const int32_t  * __restrict__ ids,
        const float    * __restrict__ x,
        float          * __restrict__ scr_v,   // [2, n_used, m]
        float          * __restrict__ h,       // [n_used, m]
        int            * __restrict__ slot_cnt,
        const int m, const int n,
        const int n_used, const int n_kept_g, const int n_kept_u,
        const int ids_s0, const int ids_s1) {
    constexpr int WG        = 128;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = WG / warp_size;
    __shared__ float shu[2048];    // x' (n <= 2048), then H(v_gate) (m <= 2048)
    __shared__ float sho[2048];    // H(v_up)
    __shared__ float red[16][4];   // >= n_warps
    __shared__ float lv[16];
    __shared__ float2 lv2[256];
    __shared__ int   is_last;

    const int s    = blockIdx.z;
    const int proj = blockIdx.y;
    const int tr   = blockIdx.x;
    const int tid  = threadIdx.x;

    const uint16_t * kept   = proj ? kept_u   : kept_g;
    const half     * su     = proj ? su_u     : su_g;
    const half     * gamma  = proj ? gamma_u  : gamma_g;
    const int        n_kept = proj ? n_kept_u : n_kept_g;
    float * vout = scr_v + ((int64_t) proj*n_used + s)*m;

    const int64_t  e = ids[s*ids_s0];
    const uint32_t g = mach1_group_of(remap, ids, s, n_used, n_kept, ids_s0, ids_s1);

    if (tid < 16) {
        lv[tid] = p4lv[tid];
    }
    __syncthreads();
    for (int b = tid; b < 256; b += WG) {
        lv2[b] = make_float2(lv[b & 0xF], lv[(b >> 4) & 0xF]);
    }
    for (int i = tid; i < n; i += WG) {
        shu[i] = __half2float(su[e*n + i]) * x[i];   // xne1 == 1 (matcher)
    }
    __syncthreads();
    mach1_fwht_block(shu, n, tid, WG);
    const float scn = __fsqrt_rn((float) n);
    for (int i = tid; i < n; i += WG) {
        shu[i] = __fdiv_rn(shu[i], scn);
    }
    __syncthreads();

    const int  tiles_y = n / 16;
    const int  ntiles  = (m / 16)*tiles_y;
    const bool have    = tid < tiles_y;

    constexpr int words = 24;   // V = 8, K = 1.5
    const int64_t tbase = ((int64_t) g*ntiles + (int64_t) tr*tiles_y)*words;

    const int Mb = m / 16;
    float gsc = 0.0f;
    if (have) {
        const int wv = (tr + tid <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + tid)
                                                 : Mb + tiles_y - 2 - (tr + tid);
        gsc = __half2float(gamma[e*(Mb + tiles_y) + wv]);
    }

    float partial[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        partial[i] = 0.0f;
    }
    if (have) {
        const int64_t tw = tbase + (int64_t) tid*words;
        uint4 kv[3];
        {
            const uint4 * kb = (const uint4 *) (kept + tw);
#pragma unroll
            for (int q = 0; q < 3; ++q) {
                kv[q] = kb[q];
            }
        }
        const uint16_t * w8 = (const uint16_t *) kv;
        const float * ub = shu + tid*16;
        float uu[16];
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            uu[c] = ub[c];
        }
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int      bb  = 12*i;
            const int      wi  = bb >> 4;
            const int      o   = bb & 15;
            const int      wn  = wi + 1 < 24 ? wi + 1 : 0;
            const uint32_t w2  = ((uint32_t) w8[wi] << 16) | (uint32_t) w8[wn];
            const uint32_t reg = w2 >> (16 - o);
            const uint32_t ph  = reg*(reg + 1u);
            const uint32_t row = ph & 0x7FFFu;
            const int      ri  = (8*i) >> 4;
            const int      ci  = (8*i) & 15;
            float dotp = 0.0f;
            const uint32_t pk = p4[row >> 1];
#pragma unroll
            for (int c = 0; c < 8; c += 2) {
                const float2 pr = lv2[(pk >> (4*c)) & 0xFFu];
                float vv = pr.x;
                if (c == 0 && (ph & 0x8000u)) {
                    vv = -vv;                 // exact (sign bit)
                }
                dotp += (vv * gsc) * uu[ci + c];
                dotp += (pr.y * gsc) * uu[ci + c + 1];
            }
            partial[ri] += dotp;
        }
    }
    const int lane = tid % warp_size;
    const int wid  = tid / warp_size;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float s2 = warp_reduce_sum<warp_size>(partial[i]);
        if (lane == 0) {
            red[i][wid] = s2;
        }
    }
    __syncthreads();
    if (tid < 16) {
        float sum = 0.0f;
#pragma unroll
        for (int wj = 0; wj < n_warps; ++wj) {
            sum += red[tid][wj];
        }
        vout[tr*16 + tid] = sum;
    }

    // last of the slot's 2*(m/16) blocks (both projections) folds the two
    // out stages and the swiglu
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        is_last = atomicAdd(&slot_cnt[s], 1) == 2*(m/16) - 1;
    }
    __syncthreads();
    if (!is_last) {
        return;
    }
    if (tid == 0) {
        slot_cnt[s] = 0;   // self-reset for the down launch on this stream
    }
    __threadfence();
    for (int i = tid; i < m; i += WG) {
        shu[i] = __ldcg(&scr_v[(int64_t) s*m + i]);
        sho[i] = __ldcg(&scr_v[((int64_t) n_used + s)*m + i]);
    }
    __syncthreads();
    mach1_fwht_block(shu, m, tid, WG);
    mach1_fwht_block(sho, m, tid, WG);
    const float scm = __fsqrt_rn((float) m);
    for (int i = tid; i < m; i += WG) {
        const float og = __fdiv_rn(shu[i], scm) * __half2float(sv_g[e*m + i]);
        const float ou = __fdiv_rn(sho[i], scm) * __half2float(sv_u[e*m + i]);
        h[(int64_t) s*m + i] = ggml_cuda_op_silu_single(og) * ou;
    }
}

// Launch B "down": grid (m/16, 1, n_used) on x = h (per-slot columns, the
// xne1 != 1 indexing of mach1_exp_fused_kernel; walk body bit-identical).
// The last block of a slot folds the out stage scaled by the router weight
// (the MUL node's two fp32 multiplies in the same association), and the last
// slot (second counter) folds the view+add slot sum as an ORDERED
// s = 0..n_used-1 fp32 add chain - the ADD kernels' exact association -
// into the region's final dst.
__global__ void mach1_exp_ffn_down_kernel(
        const uint16_t * __restrict__ kept,
        const uint32_t * __restrict__ p4,
        const float    * __restrict__ p4lv,
        const half     * __restrict__ su,
        const half     * __restrict__ sv,
        const half     * __restrict__ gamma,
        const int32_t  * __restrict__ remap,
        const int32_t  * __restrict__ ids,
        const float    * __restrict__ x,        // h from launch A, [n_used, n]
        const float    * __restrict__ weights,  // router weights, stride ws
        float          * __restrict__ scr_v,    // [n_used, m]
        float          * __restrict__ wbuf,     // [n_used, m] weighted slot outs
        float          * __restrict__ dst,      // [m]
        int            * __restrict__ slot_cnt,
        int            * __restrict__ sum_cnt,
        const int m, const int n,
        const int n_used, const int n_kept,
        const int ids_s0, const int ids_s1, const int ws) {
    constexpr int WG        = 128;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = WG / warp_size;
    __shared__ float shu[2048];    // x' (n <= 2048), then H(v) (m <= 2048)
    __shared__ float red[16][4];   // >= n_warps
    __shared__ float lv[16];
    __shared__ float2 lv2[256];
    __shared__ int   is_last;
    __shared__ int   is_final;

    const int s   = blockIdx.z;
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;

    const int64_t  e = ids[s*ids_s0];
    const uint32_t g = mach1_group_of(remap, ids, s, n_used, n_kept, ids_s0, ids_s1);

    if (tid < 16) {
        lv[tid] = p4lv[tid];
    }
    __syncthreads();
    for (int b = tid; b < 256; b += WG) {
        lv2[b] = make_float2(lv[b & 0xF], lv[(b >> 4) & 0xF]);
    }
    const int64_t xbase = (int64_t) s*n;
    for (int i = tid; i < n; i += WG) {
        shu[i] = __half2float(su[e*n + i]) * x[xbase + i];
    }
    __syncthreads();
    mach1_fwht_block(shu, n, tid, WG);
    const float scn = __fsqrt_rn((float) n);
    for (int i = tid; i < n; i += WG) {
        shu[i] = __fdiv_rn(shu[i], scn);
    }
    __syncthreads();

    const int  tiles_y = n / 16;
    const int  ntiles  = (m / 16)*tiles_y;
    const bool have    = tid < tiles_y;

    constexpr int words = 24;   // V = 8, K = 1.5
    const int64_t tbase = ((int64_t) g*ntiles + (int64_t) tr*tiles_y)*words;

    const int Mb = m / 16;
    float gsc = 0.0f;
    if (have) {
        const int wv = (tr + tid <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + tid)
                                                 : Mb + tiles_y - 2 - (tr + tid);
        gsc = __half2float(gamma[e*(Mb + tiles_y) + wv]);
    }

    float partial[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        partial[i] = 0.0f;
    }
    if (have) {
        const int64_t tw = tbase + (int64_t) tid*words;
        uint4 kv[3];
        {
            const uint4 * kb = (const uint4 *) (kept + tw);
#pragma unroll
            for (int q = 0; q < 3; ++q) {
                kv[q] = kb[q];
            }
        }
        const uint16_t * w8 = (const uint16_t *) kv;
        const float * ub = shu + tid*16;
        float uu[16];
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            uu[c] = ub[c];
        }
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int      bb  = 12*i;
            const int      wi  = bb >> 4;
            const int      o   = bb & 15;
            const int      wn  = wi + 1 < 24 ? wi + 1 : 0;
            const uint32_t w2  = ((uint32_t) w8[wi] << 16) | (uint32_t) w8[wn];
            const uint32_t reg = w2 >> (16 - o);
            const uint32_t ph  = reg*(reg + 1u);
            const uint32_t row = ph & 0x7FFFu;
            const int      ri  = (8*i) >> 4;
            const int      ci  = (8*i) & 15;
            float dotp = 0.0f;
            const uint32_t pk = p4[row >> 1];
#pragma unroll
            for (int c = 0; c < 8; c += 2) {
                const float2 pr = lv2[(pk >> (4*c)) & 0xFFu];
                float vv = pr.x;
                if (c == 0 && (ph & 0x8000u)) {
                    vv = -vv;                 // exact (sign bit)
                }
                dotp += (vv * gsc) * uu[ci + c];
                dotp += (pr.y * gsc) * uu[ci + c + 1];
            }
            partial[ri] += dotp;
        }
    }
    const int lane = tid % warp_size;
    const int wid  = tid / warp_size;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float s2 = warp_reduce_sum<warp_size>(partial[i]);
        if (lane == 0) {
            red[i][wid] = s2;
        }
    }
    __syncthreads();
    if (tid < 16) {
        float sum = 0.0f;
#pragma unroll
        for (int wj = 0; wj < n_warps; ++wj) {
            sum += red[tid][wj];
        }
        scr_v[(int64_t) s*m + tr*16 + tid] = sum;
    }

    // last of the slot's m/16 blocks folds out stage * router weight
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        is_last = atomicAdd(&slot_cnt[s], 1) == (int) gridDim.x - 1;
    }
    __syncthreads();
    if (!is_last) {
        return;
    }
    if (tid == 0) {
        slot_cnt[s] = 0;   // self-reset for the next fused op on this stream
    }
    __threadfence();
    for (int i = tid; i < m; i += WG) {
        shu[i] = __ldcg(&scr_v[(int64_t) s*m + i]);
    }
    __syncthreads();
    mach1_fwht_block(shu, m, tid, WG);
    const float scm = __fsqrt_rn((float) m);
    const float w   = weights[s*ws];
    for (int i = tid; i < m; i += WG) {
        wbuf[(int64_t) s*m + i] = (__fdiv_rn(shu[i], scm) * __half2float(sv[e*m + i])) * w;
    }

    // last slot folds the ordered cross-slot sum into the final dst
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        is_final = atomicAdd(sum_cnt, 1) == n_used - 1;
    }
    __syncthreads();
    if (!is_final) {
        return;
    }
    if (tid == 0) {
        *sum_cnt = 0;
    }
    __threadfence();
    for (int i = tid; i < m; i += WG) {
        float acc = __ldcg(&wbuf[i]);
        for (int s2 = 1; s2 < n_used; ++s2) {
            acc = acc + __ldcg(&wbuf[(int64_t) s2*m + i]);
        }
        dst[i] = acc;
    }
}

// Expert-tier megakernel (GGML_MACH1_EXP_MEGA=1): the SAME region the two
// launches above cover - [gate|up walk, out + swiglu, down walk, out * router
// weight, ordered slot sum] - as ONE persistent launch, one 1024-thread block
// per SM. Three things the two-launch form structurally cannot do (round 38,
// pocs/mach1-chainbench --layer):
//  * the 128 KB p4 nibble codebook is staged into dynamic shared ONCE at
//    kernel entry and every state gather is an LDS.32 instead of the
//    LSU-address-bound global gather (arm D vs D': -19.2% / -11.4% on the
//    batched expert steps). Registers already cap the kernel at one block/SM,
//    so the residency costs no occupancy - which is exactly why rounds 26 and
//    EXP_PERSIST, both of which staged it for a single op, failed.
//  * work is batched over ALL (projection, slot, row-tile) items at once and
//    the grid is sized to the DEVICE, not to one op. A block is a container of
//    8 virtual blocks of 128 threads covering vbpb consecutive row tiles of
//    one item; their walk body, accumulation order and reduction tree are
//    EXACTLY the gu/down kernels' above, so the result is bit-identical. The
//    x' = H(su_e .* x)/sqrt(n) prologue is then paid once per (block, item)
//    instead of once per row tile - 4x fewer for gate|up, 8x for down.
//  * the gate|up -> down transition happens IN the kernel: the block that
//    completes a slot's 2*(mff/16) tiles runs both out stages and the swiglu
//    (the certified completion-counter + __threadfence + __ldcg epilogue), and
//    the down phase waits on the h counter reaching n_used - round 38's arm E
//    released transition, not a full grid barrier.
// That wait is what makes this a COOPERATIVE launch: blocks spin on producers
// in other blocks, so the grid must be co-resident. It is sized to
// cudaOccupancyMaxActiveBlocksPerMultiprocessor (see mach1_exp_mega_grid).
#define MACH1_EXP_MEGA_WG   1024
#define MACH1_EXP_MEGA_VB   (MACH1_EXP_MEGA_WG/128)
#define MACH1_EXP_MEGA_MAXD 2048
#define MACH1_EXP_MEGA_SMEM (16384*sizeof(uint32_t) + 256*sizeof(float2) + 2*MACH1_EXP_MEGA_MAXD*sizeof(float))
// MEGA_HALF: 8192 staged rows + sho trimmed to mff (<= 512) - 44 KB, so two
// 512-thread blocks share an SM and every phase runs twice as wide
#define MACH1_EXP_MEGA_SMEM_HALF (8192*sizeof(uint32_t) + 256*sizeof(float2) + (MACH1_EXP_MEGA_MAXD + 512)*sizeof(float))
// MEGA_NOSTAGE: no staged table - lv2s + shu/sho only (~12 KB). The walk
// reads the 64 KB z-table through the read-only path; with the carveout
// preferring L1 the whole table can sit in L1 instead of occupying smem
#define MACH1_EXP_MEGA_SMEM_NOSTAGE (256*sizeof(float2) + (MACH1_EXP_MEGA_MAXD + 512)*sizeof(float))

// counter bank slots (mach1_ffn_counters) for the megakernel, laid out over
// (token, slot) pairs q = t*n_used + s with P = ntok*n_used <= 128 pairs:
//   [0..P-1]           gate|up per-pair tile counters (both projections)
//   [128..128+P-1]     down per-pair tile counters
//   [256]              h-ready release counter (target P)
//   [257]/[258]        MEGA_U1 gate|up / down u-vector ready counters
//   [260..260+P-1]     MEGA_SLOTREL per-pair h-ready flags
//   [388]              pair-completion counter for the single-writer final sum
// nt == 1 is the P = n_used special case of this layout. The two-launch form
// keeps its own [0..7]+[8] layout in the same bank (self-reset coexistence).
// nt32 (GGML_MACH1_NT32): P up to 256 pairs uses the SAME structure at
// doubled ranges (PCAP=256 in the kernel template: dn 256.., h 512, u 513/514,
// hs 516.., sum 772). Wide instantiations exist for plain QONCE only.
//
// The final sum has exactly ONE writer block, and only after every pair's
// down phase completes. This is load-bearing at nt > 1: the graph allocator
// overlays the region's out tensor onto the router's ids/weights buffers
// (dead at the adds in the unfused order), so a per-token sum racing another
// token's down-phase ids reads feeds remap a garbage expert id - the r255
// 5 GiB OOB. All reads precede the one dst write; any aliasing is safe.
#define MACH1_EXP_MEGA_CNT_DN  128
#define MACH1_EXP_MEGA_CNT_H   256
#define MACH1_EXP_MEGA_CNT_UGU 257
#define MACH1_EXP_MEGA_CNT_UDN 258
#define MACH1_EXP_MEGA_CNT_HS  260
#define MACH1_EXP_MEGA_CNT_SUM 388

struct mach1_exp_mega_args {
    const uint16_t * kept[3];    // 0 gate, 1 up, 2 down
    const half     * su[3];
    const half     * sv[3];
    const half     * gamma[3];
    const uint32_t * p4;
    const float    * p4lv;
    const uint32_t * zp4;        // ZDP: nibble-packed z+8, replaces p4 in shared
    float            zs0;        // ZDP: lattice step s0
    const int32_t  * remap;
    const int32_t  * ids;
    const float    * x;
    const float    * weights;
    float          * scr_gu;     // [2, P, mff]   (P = ntok*n_used pairs)
    float          * hbuf;       // [P, mff]
    float          * scr_d;      // [P, md]
    float          * wbuf;       // [P, md]
    float          * ubuf;       // MEGA_U1: [2, P, n]
    float          * dbuf;       // MEGA_U1: [P, mff]
    float          * dst;        // [md] per token, token stride ds1
    int            * cnt;
    int              u1;         // MEGA_U1: compute each u once, not per task
    int              slotrel;    // MEGA_SLOTREL: release down per slot, not globally
    int              probe;      // MEGA_PROBE (stopwatch only): 1 skip walks, 2 skip u FWHTs
    int              gdf;        // MEGA_GDF: GLU block also emits the down u (dbuf)
    int n_kept[3];
    int mff, n, md, n_used, ids_s0, ids_s1, ws;
    int vb_gu, tpo_gu, vb_dn, tpo_dn;
    int ntok, wst, ds1;          // token count; weights and dst token strides
};

// Exact six-word record consumed by the ZDP walk. MEGA_QONCE writes it into
// storage already reserved for later expert phases, then transposes it into
// the existing shu shared slot before each tile task. Keeping this helper
// apply-only leaves the default mega's register/code shape unchanged.
static __device__ __forceinline__ void mach1_exp_mega_pack_q8_16(
        const float * __restrict__ u, uint32_t * __restrict__ out) {
    float uu[16];
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        uu[c] = u[c];
    }
    float du = 0.0f;
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        du = fmaxf(du, fabsf(uu[c]));
    }
    const float qsf = du > 0.0f ? 127.0f/du : 0.0f;
    int q[16];
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        q[c] = __float2int_rn(uu[c]*qsf);
    }
    int sumq = 0;
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        sumq += q[c];
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        out[2*h + 0] = (uint32_t)(q[8*h + 0] & 0xFF) | ((uint32_t)(q[8*h + 2] & 0xFF) << 8)
                     | ((uint32_t)(q[8*h + 4] & 0xFF) << 16) | ((uint32_t)(q[8*h + 6] & 0xFF) << 24);
        out[2*h + 1] = (uint32_t)(q[8*h + 1] & 0xFF) | ((uint32_t)(q[8*h + 3] & 0xFF) << 8)
                     | ((uint32_t)(q[8*h + 5] & 0xFF) << 16) | ((uint32_t)(q[8*h + 7] & 0xFF) << 24);
    }
    out[4] = __float_as_uint(du);
    out[5] = (uint32_t) sumq;
}

// one row tile of the V8/K1.5 walk for a 128-thread virtual block - the body
// of mach1_exp_ffn_gu_kernel/mach1_exp_ffn_down_kernel with p4/lv2 read from
// the resident shared copies and __syncthreads replaced by the vblock's named
// barrier. Same states, same gathers, same accumulation order.
//
// CT is how many of the vblock's 128 threads carry column tiles; the other
// 128/CT groups carry the tile's other rows. At CT = 128 (the gate|up phase,
// n = md) that is the original one-group form. The DOWN phase has n = mff, so
// tiles_y = mff/16 = 32 at the census shape and 96 of the 128 threads sat out
// the walk entirely; CT = 32 gives them a row group each instead. Measured
// -29% on that phase in pocs/mach1-chainbench --walkopt (arm W8).
//
// BIT-EXACT against CT = 128, the same argument mach1_exp_fused_rows_kernel
// makes: rows 2r and 2r+1 come from states 2r, 2r+1 and 32/RG is even, so a
// row group owns whole rows and each (row, column tile) partial is the same
// states in the same order; CT is a multiple of the warp size so no warp
// straddles a group; and the cross-warp combine only ever dropped the exact
// zeros the idle threads wrote.
template <int CT, int ZDP, bool HALF = false, bool QONCE = false, bool NOSTAGE = false>
static __device__ __forceinline__ void mach1_exp_mega_tile(
        const uint16_t * __restrict__ kept,
        const uint32_t *              p4,
        const float2   *              lv2,
        const float    *              shu,
        const half     * __restrict__ gamma,
        float          *              red,
        float          * __restrict__ scr,
        const int64_t e, const uint32_t g,
        const int vt, const int tr, const int m, const int n,
        const float zs0, const int bar, const bool act,
        const uint32_t * __restrict__ p4g = nullptr) {   // HALF: rows >= 8192 read L2
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int RG        = 128/CT;
    constexpr int ROWS      = 16/RG;
    constexpr int STATES    = 32/RG;
    constexpr int n_warps   = CT/warp_size;
    constexpr int words     = 24;   // V = 8, K = 1.5
    static_assert(!QONCE || ZDP == 1, "MEGA_QONCE is a q8-ZDP-only representation");
    static_assert(!NOSTAGE || ZDP == 1, "NOSTAGE has a global path only for the z-table walk");

    const int  vtid    = vt % CT;
    const int  rg      = vt / CT;
    const int  tiles_y = n/16;
    const int  Mb      = m/16;
    const int  ntiles  = Mb*tiles_y;
    const bool have    = act && vtid < tiles_y;
    const int  lane    = vtid % warp_size;
    const int  wid     = vtid / warp_size;

    const int64_t tbase = ((int64_t) g*ntiles + (int64_t) tr*tiles_y)*words;

    float gsc = 0.0f;
    if (have) {
        const int wv = (tr + vtid <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + vtid)
                                                  : Mb + tiles_y - 2 - (tr + vtid);
        gsc = __half2float(gamma[e*(Mb + tiles_y) + wv]);
    }

    float partial[ROWS];
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
        partial[i] = 0.0f;
    }
    if (have) {
        const int64_t tw = tbase + (int64_t) vtid*words;
        uint4 kv[3];
        {
            const uint4 * kb = (const uint4 *) (kept + tw);
#pragma unroll
            for (int q = 0; q < 3; ++q) {
                kv[q] = kb[q];
            }
        }
        const uint16_t * w8 = (const uint16_t *) kv;
        if constexpr (ZDP) {
            // integer-lattice walk (mach1_exp_zdp_planes/_state): each state's
            // 8-value gather-and-MAC becomes two dp4a - four at ZDP == 2, the
            // 15-bit two-plane activation form. Scales associate outside the
            // integer sum.
            constexpr bool HI2  = ZDP == 2;
            constexpr int  QMAX = HI2 ? 16319 : 127;
            float    du;
            uint32_t qe[2], qo[2];
            uint32_t qle[HI2 ? 2 : 1], qlo[HI2 ? 2 : 1];
            int      qc0[2];
            int      sumq;
            if constexpr (QONCE) {
                // shu is an SoA six-plane q8 image in this specialization.
                // The producer record is bit-for-bit the fallback arithmetic.
                const uint32_t * q8s = (const uint32_t *) shu;
                qe[0] = q8s[0*tiles_y + vtid];
                qo[0] = q8s[1*tiles_y + vtid];
                qe[1] = q8s[2*tiles_y + vtid];
                qo[1] = q8s[3*tiles_y + vtid];
                du     = __uint_as_float(q8s[4*tiles_y + vtid]);
                sumq   = (int) q8s[5*tiles_y + vtid];
                qc0[0] = (int)(signed char)(qe[0] & 0xFFu);
                qc0[1] = (int)(signed char)(qe[1] & 0xFFu);
            } else {
                const float * ub = shu + vtid*16;
                float uu[16];
#pragma unroll
                for (int c = 0; c < 16; ++c) {
                    uu[c] = ub[c];
                }
                du = mach1_exp_zdp_planes<HI2>(uu, qe, qo, qle, qlo, qc0, sumq);
            }
            int ipart[ROWS];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                ipart[r] = 0;
            }
#pragma unroll
            for (int sx = 0; sx < STATES; ++sx) {
                const int      i   = rg*STATES + sx;
                const int      bb  = 12*i;
                const int      wi  = bb >> 4;
                const int      o   = bb & 15;
                const int      wn  = wi + 1 < 24 ? wi + 1 : 0;
                const uint32_t w2  = ((uint32_t) w8[wi] << 16) | (uint32_t) w8[wn];
                const uint32_t reg = w2 >> (16 - o);
                const uint32_t ph  = reg*(reg + 1u);
                const uint32_t row = ph & 0x7FFFu;
                const int      h   = i & 1;   // ci = 8*(i&1)
                const uint32_t zr  = row >> 1;
                // HALF: the low 8192 rows are staged, the rest hit L2 - the
                // WORD is identical either way, so the values cannot move.
                // NOSTAGE: no staged rows at all - every lookup reads the
                // 64 KB table from L2 (fully resident there), trading smem
                // instruction pressure (mio_throttle, the top mega stall)
                // for L2 latency. Same word, same values, compile-time path.
                const uint32_t zk  = NOSTAGE ? __ldg(&p4g[zr])
                                   : (HALF && zr >= 8192u) ? __ldg(&p4g[zr]) : p4[zr];
                ipart[sx >> 1] += mach1_exp_zdp_state<HI2>(zk, ph, qe, qo, qle, qlo, qc0, h);
            }
            // each row is one state per half, so the z+8 bias is uniform
            const float fsc = (zs0*gsc)*(du*(1.0f/(float) QMAX));
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                partial[r] = (float)(ipart[r] - 8*sumq)*fsc;
            }
        } else {
        const float * ub = shu + vtid*16;
        float uu[16];
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            uu[c] = ub[c];
        }
#pragma unroll
        for (int sx = 0; sx < STATES; ++sx) {
            const int      i   = rg*STATES + sx;
            const int      bb  = 12*i;
            const int      wi  = bb >> 4;
            const int      o   = bb & 15;
            const int      wn  = wi + 1 < 24 ? wi + 1 : 0;
            const uint32_t w2  = ((uint32_t) w8[wi] << 16) | (uint32_t) w8[wn];
            const uint32_t reg = w2 >> (16 - o);
            const uint32_t ph  = reg*(reg + 1u);
            const uint32_t row = ph & 0x7FFFu;
            const int      ri  = sx >> 1;
            const int      ci  = (8*i) & 15;
            float dotp = 0.0f;
            const uint32_t pk = p4[row >> 1];
#pragma unroll
            for (int c = 0; c < 8; c += 2) {
                const float2 pr = lv2[(pk >> (4*c)) & 0xFFu];
                float vv = pr.x;
                if (c == 0 && (ph & 0x8000u)) {
                    vv = -vv;                 // exact (sign bit)
                }
                dotp += (vv * gsc) * uu[ci + c];
                dotp += (pr.y * gsc) * uu[ci + c + 1];
            }
            partial[ri] += dotp;
        }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const float s2 = warp_reduce_sum<warp_size>(partial[r]);
        if (lane == 0) {
            red[(rg*ROWS + r)*4 + wid] = s2;
        }
    }
    mach1_tile_sync(bar);
    if (act && vt < 16) {
        float sum = 0.0f;
#pragma unroll
        for (int wj = 0; wj < n_warps; ++wj) {
            sum += red[vt*4 + wj];
        }
        scr[tr*16 + vt] = sum;
    }
}

// DNCT is the down phase's column-tile width (see mach1_exp_mega_tile);
// GGML_MACH1_EXP_DNROWS=1 picks 32 when the phase's tiles_y allows it.
// PCAP is the counter-layout pair capacity: 128 is the certified layout (its
// constants below equal the MACH1_EXP_MEGA_CNT_* macros, so existing
// instantiations are codegen-identical), 256 is the nt32 widened layout
// (QONCE only) with the same structure at doubled pair ranges.
// ORDER (GGML_MACH1_MEGA_ORDER): the tile task loops visit pairs through a
// per-launch expert-sorted permutation so same-expert pairs are adjacent and
// their payload re-reads stay cache-resident. Pair identity, per-pair math,
// counters and the token-sum order are all keyed by the remapped q, so the
// values are bitwise identical to the unordered kernel.
template <int DNCT, int ZDP, int WG = MACH1_EXP_MEGA_WG, bool HALF = false,
          int MINB = (HALF && WG == 512) ? 2 : 1, int PHASE = 0, bool QONCE = false, bool QIP = false,
          bool STD = false, int PCAP = 128, bool ORDER = false, bool PFLAGS = false,
          bool NOSTAGE = false>
__launch_bounds__(WG, MINB)
static __global__ void mach1_exp_mega_kernel(const mach1_exp_mega_args a) {
    // NOSTAGE drops the table staging entirely: every walk lookup reads the
    // 64 KB z-table from L2 via the tile's compile-time global path (r230:
    // the pointer must not become a runtime shared-or-global switch). smem
    // falls to ~12 KB so the WG512 split can raise blocks/SM; whether the
    // mio relief beats the L2 latency is the race.
    constexpr int TABW = NOSTAGE ? 0 : (HALF ? 8192 : 16384);
    // NOSTAGE forms: the split pflags pair, and (rung 9) the cooperative
    // QONCE kernel itself - the qs2-wall A/B showed the L2/L1-served table
    // beats the smem-served table by 12-16 percent inside the split family
    // at equal occupancy, so the same trade rides the certified coop launch.
    static_assert(!NOSTAGE || (QONCE && ZDP == 1 && WG == 512 && (PHASE == 1 || PHASE == 2) && PFLAGS) ||
                  (QONCE && ZDP == 1 && WG == MACH1_EXP_MEGA_WG && PHASE == 0 && !PFLAGS && !ORDER && !QIP && !STD && PCAP == 128),
                  "NOSTAGE rides the QONCE forms only (split pflags pair or the plain cooperative kernel)");
    // QONCE lives in the full cooperative kernel and, since the split rung,
    // in the two plain PHASE launches (WG 512, half table, 2 blocks/SM). The
    // kernel boundary is the H join, so the record aliasing proof is the
    // launch order: gate/up records in scr_d/wbuf are dead before phase 2
    // overwrites those slices, and phase 2 reads only the hbuf records.
    // The split phases REQUIRE the PFLAGS producer edge: the CNT_UGU barrier
    // needs all 2*P producer blocks co-resident, which the plain ntask-wide
    // launch does not promise (P=128 deadlocked on it). Per-record flags wait
    // only on record b/tpo <= b, so any scheduled prefix of blocks makes
    // progress. Phase 2 carries PFLAGS for the final flag reset - without it
    // stale flags would release the next region's consumers early.
    static_assert(!QONCE || (ZDP == 1 && WG == MACH1_EXP_MEGA_WG && !HALF && PHASE == 0) ||
                  (ZDP == 1 && WG == 512 && HALF && (PHASE == 1 || PHASE == 2) && PFLAGS),
                  "MEGA_QONCE: full cooperative kernel or the split PHASE pair (PFLAGS) only");
    static_assert(!QIP || QONCE, "QONCE_FWSCALE is a QONCE producer variant");
    static_assert(PCAP == 128 || (PCAP == 256 && QONCE && !QIP && !STD),
                  "pair capacities: 128 (certified layout) or 256 (nt32, plain QONCE only)");
    static_assert(!ORDER || (QONCE && !QIP && !STD && PCAP == 128),
                  "pair ordering is instantiated for the certified plain-QONCE layout only");
    // perm table + ready flag live in the counter bank's idle margin, past
    // every certified layout (PCAP=128 ends 388, wide 772; bank is 1040)
    constexpr int CNT_PERM  = 900;
    constexpr int CNT_PERMF = 1032;
    static_assert(!ORDER || (CNT_PERM + PCAP <= CNT_PERMF && CNT_PERMF < 1040),
                  "perm table must fit the counter bank margin");
    // PFLAGS: per-(proj, pair) record-ready flags replace the global producer
    // barrier - gate/up tiles of pair q start as soon as q's own record is
    // published (the slotrel pattern applied to the producer edge). Values
    // are unchanged; only the wait granularity is. Flags live at [640, 896):
    // inside the wide layout's self-resetting HS/SUM span and clear of the
    // shexp tail (392..396) and the ORDER perm (900+).
    constexpr int CNT_RF = 640;
    static_assert(!PFLAGS || (QONCE && !QIP && !STD && PCAP == 128),
                  "record flags are instantiated for the certified plain-QONCE layout only");
    static_assert(!PFLAGS || (CNT_RF + 2*PCAP <= CNT_PERM),
                  "record flags must fit below the ORDER perm table");
    constexpr int CNT_DN  = PCAP;
    constexpr int CNT_H   = 2*PCAP;
    constexpr int CNT_UGU = 2*PCAP + 1;
    constexpr int CNT_UDN = 2*PCAP + 2;
    constexpr int CNT_HS  = 2*PCAP + 4;
    constexpr int CNT_SUM = 3*PCAP + 4;
    static_assert(PCAP != 128 || (CNT_DN == MACH1_EXP_MEGA_CNT_DN && CNT_H == MACH1_EXP_MEGA_CNT_H &&
                  CNT_UGU == MACH1_EXP_MEGA_CNT_UGU && CNT_UDN == MACH1_EXP_MEGA_CNT_UDN &&
                  CNT_HS == MACH1_EXP_MEGA_CNT_HS && CNT_SUM == MACH1_EXP_MEGA_CNT_SUM),
                  "PCAP=128 must reproduce the certified counter layout exactly");
    // STD: standard-output-basis expert payloads - the gate/up/down walk rows
    // are already standard-basis, so the three output FWHTs and their
    // 1/sqrt(m) fall away and sv acts as a direct row scale (see
    // mach1_rt_stdout_on for the wrong-basis timing-only contract)
    static_assert(!STD || QONCE, "RT_STDOUT rides the audited QONCE mega only");

    extern __shared__ char smem[];
    uint32_t * p4s  = (uint32_t *) smem;
    float2   * lv2s = (float2 *)  (smem + TABW*sizeof(uint32_t));
    float    * shu  = (float *)   (smem + TABW*sizeof(uint32_t) + 256*sizeof(float2));
    float    * sho  = shu + MACH1_EXP_MEGA_MAXD;   // HALF: mff floats only (dispatch-guarded)

    __shared__ float red[WG/128][16][4];
    __shared__ float lv[16];
    __shared__ int   s_last;
    __shared__ int   s_fin;

    const int tid  = threadIdx.x;
    const int vb   = tid >> 7;
    const int vtid = tid & 127;
    const int bar  = vb + 1;
    const int bx   = blockIdx.x;
    const int NB   = gridDim.x;

    if (tid < 16) {
        lv[tid] = a.p4lv[tid];
    }
    // ZDP stages the z-nibble table in the same slot; lv2s is then unread.
    // r230: routing the tile through a runtime shared-or-global pointer
    // compiled the gathers to generic-address loads and halved the arm -
    // the table pointer must stay compile-time shared.
    const uint32_t * p4g = ZDP ? a.zp4 : a.p4;
    for (int i = tid; i < TABW; i += WG) {
        p4s[i] = p4g[i];
    }
    __syncthreads();
    for (int b = tid; b < 256; b += WG) {
        lv2s[b] = make_float2(lv[b & 0xF], lv[(b >> 4) & 0xF]);
    }
    __syncthreads();

    const int n        = a.n;
    const int mff      = a.mff;
    const int md       = a.md;
    const int n_used   = a.n_used;
    const int ntok     = a.ntok;
    const int P        = n_used*ntok;   // (token, slot) pairs, q = t*n_used + s
    const int rows_gu  = mff/16;
    const int rows_dn  = md/16;
    const int ntask_gu = 2*P*a.tpo_gu;
    const int ntask_dn = P*a.tpo_dn;

    // MEGA_QONCE uses the U1 cooperative producer pattern, but publishes the
    // compact q8 record directly into future-phase slices. All resident blocks
    // share the 2*P producers evenly; this avoids both redundant tile-local
    // quantization and the 15-producer/45-spinner imbalance of a lazy mapping.
    if constexpr (ORDER) {
        // Counting sort of pairs by routed expert, overlapped with the other
        // blocks' producer work. shu is dead until the producer loop below,
        // so its front doubles as the histogram.
        if (bx == 0) {
            int * s_hist = (int *) shu;
            for (int i = tid; i < 129; i += WG) {
                s_hist[i] = 0;
            }
            __syncthreads();
            for (int q = tid; q < P; q += WG) {
                const int e = (int) a.ids[(q % n_used)*a.ids_s0 + (q/n_used)*a.ids_s1] & 127;
                atomicAdd(&s_hist[e + 1], 1);
            }
            __syncthreads();
            if (tid == 0) {
                for (int i = 1; i < 129; ++i) {
                    s_hist[i] += s_hist[i - 1];
                }
            }
            __syncthreads();
            for (int q = tid; q < P; q += WG) {
                const int e = (int) a.ids[(q % n_used)*a.ids_s0 + (q/n_used)*a.ids_s1] & 127;
                a.cnt[CNT_PERM + atomicAdd(&s_hist[e], 1)] = q;
            }
            __threadfence();
            __syncthreads();
            if (tid == 0) {
                atomicAdd(&a.cnt[CNT_PERMF], 1);
            }
        }
    }
    if constexpr (QONCE && PHASE != 2) {
        for (int o = bx; o < 2*P; o += NB) {
            const int proj = o/P;
            const int q    = o - proj*P;
            const int64_t e = a.ids[(q % n_used)*a.ids_s0 + (q/n_used)*a.ids_s1];
            const half  * su = a.su[proj];
            const float * xq = a.x + (int64_t)(q/n_used)*n;
            for (int i = tid; i < n; i += WG) {
                shu[i] = __half2float(su[e*n + i]) * xq[i];
            }
            __syncthreads();
            const float scn = __fsqrt_rn((float) n);
            // QIP: scaled shu is dead after the pack, so the normalization
            // rides the FWHT's last butterfly store and the scaled-shared
            // write/read pass and its barrier disappear. Folding it into the
            // pack loads instead was priced and killed: it concentrates the
            // divisions in the n/16 pack threads.
            if constexpr (QIP) {
                mach1_fwht_block<true>(shu, n, tid, WG, scn);
            } else {
                mach1_fwht_block(shu, n, tid, WG);
                for (int i = tid; i < n; i += WG) {
                    shu[i] = __fdiv_rn(shu[i], scn);
                }
                __syncthreads();
            }
            uint32_t * q8g = (uint32_t *) (proj == 0 ? a.scr_d : a.wbuf) + (int64_t) q*md;
            if (tid < n/16) {
                mach1_exp_mega_pack_q8_16(shu + 16*tid, q8g + 6*tid);
                // Only these threads wrote the record. Fence their stores
                // before the block barrier hands publication to tid 0;
                // fencing the other 7/8 of this 1024-thread CTA is pure cost.
                __threadfence();
            }
            __syncthreads();
            if (tid == 0) {
                if constexpr (PFLAGS) {
                    a.cnt[CNT_RF + o] = 1;
                } else {
                    atomicAdd(&a.cnt[CNT_UGU], 1);
                }
            }
        }
        if constexpr (!PFLAGS) {
            if (tid == 0) {
                while (*(volatile int *) &a.cnt[CNT_UGU] < 2*P) { }
            }
        }
        if (tid == 0) {
            if constexpr (ORDER) {
                while (*(volatile int *) &a.cnt[CNT_PERMF] < 1) { }
            }
        }
        __syncthreads();
        __threadfence();
    } else if (a.u1) {
    // MEGA_U1: each (proj, pair) u is identical for every tile task of that
    // item, so compute the 2*P vectors once and let tasks read them.
    // Bit-exact: same input, same block-wide FWHT, one writer per vector.
        for (int o = bx; o < 2*P; o += NB) {
            const int proj = o/P;
            const int q    = o - proj*P;
            const int64_t e = a.ids[(q % n_used)*a.ids_s0 + (q/n_used)*a.ids_s1];
            const half  * su = a.su[proj];
            const float * xq = a.x + (int64_t)(q/n_used)*n;
            for (int i = tid; i < n; i += WG) {
                shu[i] = __half2float(su[e*n + i]) * xq[i];
            }
            __syncthreads();
            mach1_fwht_block(shu, n, tid, WG);
            const float scn = __fsqrt_rn((float) n);
            for (int i = tid; i < n; i += WG) {
                a.ubuf[(int64_t) o*n + i] = __fdiv_rn(shu[i], scn);
            }
            __threadfence();
            __syncthreads();
            if (tid == 0) {
                atomicAdd(&a.cnt[CNT_UGU], 1);
            }
        }
        if (tid == 0) {
            while (*(volatile int *) &a.cnt[CNT_UGU] < 2*P) { }
        }
        __syncthreads();
        __threadfence();
    }

    for (int t = bx; PHASE != 2 && t < ntask_gu; t += NB) {
        const int o    = t/a.tpo_gu;
        const int tr0  = (t - o*a.tpo_gu)*a.vb_gu;
        const int proj = o/P;
        // pair: slot q%n_used of token q/n_used; ORDER visits pairs in
        // expert-sorted order while keeping the pair identity in q
        const int q    = ORDER ? a.cnt[CNT_PERM + (o - proj*P)] : o - proj*P;

        if constexpr (PFLAGS) {
            // wait for this pair's own record instead of the producer barrier
            if (tid == 0) {
                while (*(volatile int *) &a.cnt[CNT_RF + proj*P + q] == 0) { }
            }
            __syncthreads();
            __threadfence();
        }

        const int64_t  e = a.ids[(q % n_used)*a.ids_s0 + (q/n_used)*a.ids_s1];
        const uint32_t g = mach1_group_of(a.remap, a.ids, q, n_used, a.n_kept[proj], a.ids_s0, a.ids_s1);

        if constexpr (QONCE) {
            // Gate/up q8 records live in the future down-output slices. At
            // the p16 gate these are 768 words inside a 2048-word pair slice.
            uint32_t * q8g = (uint32_t *) (proj == 0 ? a.scr_d : a.wbuf) + (int64_t) q*md;
            const int tiles_q = n/16;
            // Global is AoS for coalesced producer stores; shared is SoA so
            // all eight virtual blocks issue conflict-free plane loads.
            uint32_t * q8s = (uint32_t *) shu;
            for (int i = tid; i < 6*tiles_q; i += WG) {
                const int ti = i/6;
                const int k  = i - 6*ti;
                q8s[k*tiles_q + ti] = __ldcg(q8g + i);
            }
            __syncthreads();
        } else if (a.u1) {
            for (int i = tid; i < n; i += WG) {
                shu[i] = __ldcg(&a.ubuf[(int64_t) o*n + i]);
            }
            __syncthreads();
        } else {
        const half  * su = a.su[proj];
        const float * xq = a.x + (int64_t)(q/n_used)*n;   // x is [n, 1, ntok] (matcher)
        for (int i = tid; i < n; i += WG) {
            shu[i] = __half2float(su[e*n + i]) * xq[i];
        }
        __syncthreads();
        if (a.probe != 2) {
        mach1_fwht_block(shu, n, tid, WG);
        const float scn = __fsqrt_rn((float) n);
        for (int i = tid; i < n; i += WG) {
            shu[i] = __fdiv_rn(shu[i], scn);
        }
        }
        __syncthreads();
        }

        // MEGA_TW: a task may cover several row chunks so the record staging,
        // fences, and counter traffic amortize; each vblock's named barrier
        // and red slice are private and shu is read-only here, so chunks need
        // no block-wide barrier between them
        for (int w = 0; w < a.vb_gu; w += MACH1_EXP_MEGA_VB) {
        const int  tr  = tr0 + w + vb;
        const bool act = w + vb < a.vb_gu && tr < rows_gu;
        if (a.probe == 1) {
            if (act && vtid < 16) {
                a.scr_gu[((int64_t) proj*P + q)*mff + tr*16 + vtid] = 0.0f;
            }
        } else {
        mach1_exp_mega_tile<128, ZDP, HALF, QONCE, NOSTAGE>(a.kept[proj], p4s, lv2s, shu, a.gamma[proj], &red[vb][0][0],
                            a.scr_gu + ((int64_t) proj*P + q)*mff, e, g,
                            vtid, act ? tr : tr0, mff, n, a.zs0, bar, act, p4g);
        }
        }

        __threadfence();
        __syncthreads();
        const int ntile = min(a.vb_gu, rows_gu - tr0);
        if (tid == 0) {
            s_last = atomicAdd(&a.cnt[q], ntile) + ntile == 2*rows_gu;
        }
        __syncthreads();
        if (!s_last) {
            continue;
        }
        if (tid == 0) {
            a.cnt[q] = 0;   // self-reset for the next region on this stream
        }
        __threadfence();
        for (int i = tid; i < mff; i += WG) {
            shu[i] = __ldcg(&a.scr_gu[(int64_t) q*mff + i]);
            sho[i] = __ldcg(&a.scr_gu[((int64_t) P + q)*mff + i]);
        }
        __syncthreads();
        if constexpr (!STD) {
            mach1_fwht_block(shu, mff, tid, WG);
            mach1_fwht_block(sho, mff, tid, WG);
        }
        const float scm = __fsqrt_rn((float) mff);
        if constexpr (QONCE) {
            // Continue the winning pair's GLU in shared, form the down-u once,
            // and publish only its compact q8 records into hbuf. A shared
            // store/load pins the same fp32 h rounding as the default path.
            for (int i = tid; i < mff; i += WG) {
                const float og = (STD ? shu[i] : __fdiv_rn(shu[i], scm)) * __half2float(a.sv[0][e*mff + i]);
                const float ou = (STD ? sho[i] : __fdiv_rn(sho[i], scm)) * __half2float(a.sv[1][e*mff + i]);
                sho[i] = ggml_cuda_op_silu_single(og) * ou;
            }
            __syncthreads();
            for (int i = tid; i < mff; i += WG) {
                shu[i] = __half2float(a.su[2][e*mff + i]) * sho[i];
            }
            __syncthreads();
            if constexpr (QIP) {
                mach1_fwht_block<true>(shu, mff, tid, WG, scm);
            } else {
                mach1_fwht_block(shu, mff, tid, WG);
                for (int i = tid; i < mff; i += WG) {
                    shu[i] = __fdiv_rn(shu[i], scm);
                }
                __syncthreads();
            }
            uint32_t * q8g = (uint32_t *) a.hbuf + (int64_t) q*mff;
            if (tid < mff/16) {
                mach1_exp_mega_pack_q8_16(shu + 16*tid, q8g + 6*tid);
                // As above, publish from the 32 actual writers rather than
                // issuing a global fence in all 1024 resident threads.
                __threadfence();
            }
        } else if (a.gdf) {
            // GDF: the slot's h never leaves shared - the same block continues
            // straight into the down-u stage (su*h, FWHT) and publishes dbuf,
            // removing one global round-trip and one phase from the slot chain.
            // Values match the u1 path exactly: fp32 h, same expression order.
            for (int i = tid; i < mff; i += WG) {
                const float og = __fdiv_rn(shu[i], scm) * __half2float(a.sv[0][e*mff + i]);
                const float ou = __fdiv_rn(sho[i], scm) * __half2float(a.sv[1][e*mff + i]);
                shu[i] = __half2float(a.su[2][e*mff + i]) * (ggml_cuda_op_silu_single(og) * ou);
            }
            __syncthreads();
            mach1_fwht_block(shu, mff, tid, WG);
            for (int i = tid; i < mff; i += WG) {
                a.dbuf[(int64_t) q*mff + i] = __fdiv_rn(shu[i], scm);
            }
        } else {
        for (int i = tid; i < mff; i += WG) {
            const float og = __fdiv_rn(shu[i], scm) * __half2float(a.sv[0][e*mff + i]);
            const float ou = __fdiv_rn(sho[i], scm) * __half2float(a.sv[1][e*mff + i]);
            a.hbuf[(int64_t) q*mff + i] = ggml_cuda_op_silu_single(og) * ou;
        }
        }
        if constexpr (!QONCE) {
            __threadfence();
        }
        __syncthreads();
        if (tid == 0) {
            a.cnt[CNT_HS + q] = 1; // h[q] or compact down-u ready
            atomicAdd(&a.cnt[CNT_H], 1);
        }
    }

    // released transition: only blocks that carry down work wait, and they
    // wait on h itself rather than on every block arriving. A block still
    // spinning here implies its own down tasks are unstarted, which is why the
    // final block below can safely clear the counter.
    // MEGA_SLOTREL waits per task on its own slot's h instead, so slot chains
    // overlap the gate|up tail of the other slots.
    if (PHASE == 1) {
        return;
    }
    if (bx >= ntask_dn) {
        return;
    }
    if (PHASE == 0) {
    if (tid == 0 && !a.slotrel) {
        while (*(volatile int *) &a.cnt[CNT_H] < P) { }
    }
    __syncthreads();
    __threadfence();
    }

    // MEGA_U1: same argument as the gate|up phase - one down u per pair,
    // computed after h is released, read by every tile task of that pair
    // (GDF already published dbuf from the GLU block, so the stage is skipped)
    if (a.u1 && !a.gdf) {
        for (int q = bx; q < P; q += NB) {
            if (a.slotrel) {
                if (tid == 0) {
                    while (*(volatile int *) &a.cnt[CNT_HS + q] == 0) { }
                }
                __syncthreads();
                __threadfence();
            }
            const int64_t e = a.ids[(q % n_used)*a.ids_s0 + (q/n_used)*a.ids_s1];
            const half * su = a.su[2];
            for (int i = tid; i < mff; i += WG) {
                shu[i] = __half2float(su[e*mff + i]) * __ldcg(&a.hbuf[(int64_t) q*mff + i]);
            }
            __syncthreads();
            mach1_fwht_block(shu, mff, tid, WG);
            const float scn = __fsqrt_rn((float) mff);
            for (int i = tid; i < mff; i += WG) {
                a.dbuf[(int64_t) q*mff + i] = __fdiv_rn(shu[i], scn);
            }
            __threadfence();
            __syncthreads();
            if (tid == 0) {
                atomicAdd(&a.cnt[CNT_UDN], 1);
            }
        }
        if (tid == 0) {
            while (*(volatile int *) &a.cnt[CNT_UDN] < P) { }
        }
        __syncthreads();
        __threadfence();
    }

    for (int t = bx; t < ntask_dn; t += NB) {
        const int qi  = t/a.tpo_dn;
        const int tr0 = (t - qi*a.tpo_dn)*a.vb_dn;
        const int q   = ORDER ? a.cnt[CNT_PERM + qi] : qi;
        const int tk  = q/n_used;

        if (a.slotrel && !a.u1) {
            if (tid == 0) {
                while (*(volatile int *) &a.cnt[CNT_HS + q] == 0) { }
            }
            __syncthreads();
            __threadfence();
        }

        const int64_t  e = a.ids[(q % n_used)*a.ids_s0 + tk*a.ids_s1];
        const uint32_t g = mach1_group_of(a.remap, a.ids, q, n_used, a.n_kept[2], a.ids_s0, a.ids_s1);

        if constexpr (QONCE) {
            const int tiles_q = mff/16;
            const uint32_t * q8g = (const uint32_t *) a.hbuf + (int64_t) q*mff;
            uint32_t * q8s = (uint32_t *) shu;
            for (int i = tid; i < 6*tiles_q; i += WG) {
                const int ti = i/6;
                const int k  = i - 6*ti;
                q8s[k*tiles_q + ti] = __ldcg(q8g + i);
            }
            __syncthreads();
        } else if (a.u1 || a.gdf) {
            for (int i = tid; i < mff; i += WG) {
                shu[i] = __ldcg(&a.dbuf[(int64_t) q*mff + i]);
            }
            __syncthreads();
        } else {
        const half * su = a.su[2];
        for (int i = tid; i < mff; i += WG) {
            shu[i] = __half2float(su[e*mff + i]) * __ldcg(&a.hbuf[(int64_t) q*mff + i]);
        }
        __syncthreads();
        mach1_fwht_block(shu, mff, tid, WG);
        const float scn = __fsqrt_rn((float) mff);
        for (int i = tid; i < mff; i += WG) {
            shu[i] = __fdiv_rn(shu[i], scn);
        }
        __syncthreads();
        }

        for (int w = 0; w < a.vb_dn; w += MACH1_EXP_MEGA_VB) {
        const int  tr  = tr0 + w + vb;
        const bool act = w + vb < a.vb_dn && tr < rows_dn;
        if (a.probe == 1) {
            if (act && vtid < 16) {
                a.scr_d[(int64_t) q*md + tr*16 + vtid] = 0.0f;
            }
        } else {
        mach1_exp_mega_tile<DNCT, ZDP, HALF, QONCE, NOSTAGE>(a.kept[2], p4s, lv2s, shu, a.gamma[2], &red[vb][0][0],
                            a.scr_d + (int64_t) q*md, e, g,
                            vtid, act ? tr : tr0, md, mff, a.zs0, bar, act, p4g);
        }
        }

        __threadfence();
        __syncthreads();
        const int ntile = min(a.vb_dn, rows_dn - tr0);
        if (tid == 0) {
            s_last = atomicAdd(&a.cnt[CNT_DN + q], ntile) + ntile == rows_dn;
        }
        __syncthreads();
        if (!s_last) {
            continue;
        }
        if (tid == 0) {
            a.cnt[CNT_DN + q] = 0;
        }
        __threadfence();
        for (int i = tid; i < md; i += WG) {
            shu[i] = __ldcg(&a.scr_d[(int64_t) q*md + i]);
        }
        __syncthreads();
        if constexpr (!STD) {
            mach1_fwht_block(shu, md, tid, WG);
        }
        const float scm = __fsqrt_rn((float) md);
        const float w   = a.weights[(q % n_used)*a.ws + tk*a.wst];
        for (int i = tid; i < md; i += WG) {
            a.wbuf[(int64_t) q*md + i] = ((STD ? shu[i] : __fdiv_rn(shu[i], scm)) * __half2float(a.sv[2][e*md + i])) * w;
        }

        __threadfence();
        __syncthreads();
        // single-writer final: the LAST pair's completion block sums every
        // token, so all ids/weights/x reads in the grid strictly precede the
        // dst write (see the counter-layout comment - dst may alias them).
        // Safe to reset the shared slots here: every dn task ran, so no block
        // can still be spinning on H or an HS flag.
        if (tid == 0) {
            s_fin = atomicAdd(&a.cnt[CNT_SUM], 1) == P - 1;
        }
        __syncthreads();
        if (!s_fin) {
            continue;
        }
        if (tid == 0) {
            a.cnt[CNT_SUM] = 0;
            a.cnt[CNT_H]   = 0;
            a.cnt[CNT_UGU] = 0;
            a.cnt[CNT_UDN] = 0;
            if constexpr (ORDER) {
                a.cnt[CNT_PERMF] = 0;
            }
            if constexpr (PFLAGS) {
                for (int o2 = 0; o2 < 2*P; ++o2) {
                    a.cnt[CNT_RF + o2] = 0;
                }
            }
            for (int q2 = 0; q2 < P; ++q2) {
                a.cnt[CNT_HS + q2] = 0;
            }
        }
        __threadfence();
        for (int tk2 = 0; tk2 < ntok; ++tk2) {
            const int64_t wb0 = (int64_t) tk2*n_used*md;
            for (int i = tid; i < md; i += WG) {
                float acc = __ldcg(&a.wbuf[wb0 + i]);
                for (int s2 = 1; s2 < n_used; ++s2) {
                    acc = acc + __ldcg(&a.wbuf[wb0 + (int64_t) s2*md + i]);
                }
                a.dst[(int64_t) tk2*a.ds1 + i] = acc;
            }
        }
    }
}

// Per-slot expert megakernel (GGML_MACH1_EXP_SLOT=1, nt == 1). At decode the
// mega runs at most one tile task per block and its wall clock is the serial
// phase chain: stage, FWHT, walk, reduce, grid-wide counter spin, GLU, down
// chain - r213 proved removing work (U1) moves nothing and shortening the
// walk (ZDP) moves little, because the chain is the cost. But each routed
// slot's gate -> up -> GLU -> down chain is INDEPENDENT until the final
// cross-slot sum, so this form gives every slot ONE 1024-thread block and
// runs the whole chain in shared memory: no fences, no spins, no global
// round-trips until wbuf. The walks are mach1_exp_mega_tile over the same
// tiles in the same order, the GLU and scales replicate the mega's lines
// verbatim, and the final sum keeps the mega's slot order - BIT-IDENTICAL
// to the mega by construction, so the smoke gate applies.
template <int DNCT, bool ZDP>
__launch_bounds__(1024, 1)
static __global__ void mach1_exp_slot_kernel(const mach1_exp_mega_args a) {
    constexpr int WG = 1024;
    constexpr int VB = WG/128;

    extern __shared__ char smem[];
    uint32_t * p4s  = (uint32_t *) smem;
    float2   * lv2s = (float2 *)  (smem + 16384*sizeof(uint32_t));
    float    * shu  = (float *)   (smem + 16384*sizeof(uint32_t) + 256*sizeof(float2));
    float    * sho  = shu + 2048;
    float    * sgu  = sho + 2048;   // [512] gate walk out, then the down u
    float    * suu  = sgu + 512;    // [512] up walk out
    float    * sh_h = suu + 512;    // [512] h

    __shared__ float red[VB][16][4];
    __shared__ float lv[16];

    const int tid  = threadIdx.x;
    const int vb   = tid >> 7;
    const int vtid = tid & 127;
    const int bar  = vb + 1;
    // one block per (token, slot) pair: q = tk*n_used + s. nt == 1 keeps the
    // original indexing exactly (tk == 0 zeroes every token-stride term).
    const int q    = blockIdx.x;

    if (tid < 16) {
        lv[tid] = a.p4lv[tid];
    }
    const uint32_t * p4g = ZDP ? a.zp4 : a.p4;
    for (int i = tid; i < 16384; i += WG) {
        p4s[i] = p4g[i];
    }
    __syncthreads();
    for (int b = tid; b < 256; b += WG) {
        lv2s[b] = make_float2(lv[b & 0xF], lv[(b >> 4) & 0xF]);
    }
    __syncthreads();

    const int n      = a.n;
    const int mff    = a.mff;
    const int md     = a.md;
    const int n_used = a.n_used;
    const int tk     = q/n_used;
    const int s      = q - tk*n_used;

    const int64_t  e  = a.ids[s*a.ids_s0 + tk*a.ids_s1];
    const uint32_t g0 = mach1_group_of(a.remap, a.ids, q, n_used, a.n_kept[0], a.ids_s0, a.ids_s1);
    const uint32_t g1 = mach1_group_of(a.remap, a.ids, q, n_used, a.n_kept[1], a.ids_s0, a.ids_s1);
    const uint32_t g2 = mach1_group_of(a.remap, a.ids, q, n_used, a.n_kept[2], a.ids_s0, a.ids_s1);

    // both u vectors (xne1 == 1 at the matcher; token tk's row at decode)
    const float * xq = a.x + (int64_t) tk*n;
    for (int i = tid; i < n; i += WG) {
        shu[i] = __half2float(a.su[0][e*n + i]) * xq[i];
        sho[i] = __half2float(a.su[1][e*n + i]) * xq[i];
    }
    __syncthreads();
    mach1_fwht_block(shu, n, tid, WG);
    mach1_fwht_block(sho, n, tid, WG);
    const float scn = __fsqrt_rn((float) n);
    for (int i = tid; i < n; i += WG) {
        shu[i] = __fdiv_rn(shu[i], scn);
        sho[i] = __fdiv_rn(sho[i], scn);
    }
    __syncthreads();

    const int rows_gu = mff/16;
    for (int tr = vb; tr < rows_gu; tr += VB) {
        mach1_exp_mega_tile<128, ZDP>(a.kept[0], p4s, lv2s, shu, a.gamma[0], &red[vb][0][0],
                            sgu, e, g0, vtid, tr, mff, n, a.zs0, bar, true);
    }
    for (int tr = vb; tr < rows_gu; tr += VB) {
        mach1_exp_mega_tile<128, ZDP>(a.kept[1], p4s, lv2s, sho, a.gamma[1], &red[vb][0][0],
                            suu, e, g1, vtid, tr, mff, n, a.zs0, bar, true);
    }
    __syncthreads();

    // the mega's GLU lines on the shared buffers
    mach1_fwht_block(sgu, mff, tid, WG);
    mach1_fwht_block(suu, mff, tid, WG);
    const float scm = __fsqrt_rn((float) mff);
    for (int i = tid; i < mff; i += WG) {
        const float og = __fdiv_rn(sgu[i], scm) * __half2float(a.sv[0][e*mff + i]);
        const float ou = __fdiv_rn(suu[i], scm) * __half2float(a.sv[1][e*mff + i]);
        sh_h[i] = ggml_cuda_op_silu_single(og) * ou;
    }
    __syncthreads();

    // down u into sgu
    for (int i = tid; i < mff; i += WG) {
        sgu[i] = __half2float(a.su[2][e*mff + i]) * sh_h[i];
    }
    __syncthreads();
    mach1_fwht_block(sgu, mff, tid, WG);
    for (int i = tid; i < mff; i += WG) {
        sgu[i] = __fdiv_rn(sgu[i], scm);
    }
    __syncthreads();

    const int rows_dn = md/16;
    for (int tr = vb; tr < rows_dn; tr += VB) {
        mach1_exp_mega_tile<DNCT, ZDP>(a.kept[2], p4s, lv2s, sgu, a.gamma[2], &red[vb][0][0],
                            shu, e, g2, vtid, tr, md, mff, a.zs0, bar, true);
    }
    __syncthreads();

    mach1_fwht_block(shu, md, tid, WG);
    const float scd = __fsqrt_rn((float) md);
    const float w   = a.weights[s*a.ws + tk*a.wst];
    for (int i = tid; i < md; i += WG) {
        a.wbuf[(int64_t) q*md + i] = (__fdiv_rn(shu[i], scd) * __half2float(a.sv[2][e*md + i])) * w;
    }
    __threadfence();
    __syncthreads();

    // single-writer final: the LAST pair's block sums every token, so all
    // ids/weights/x reads strictly precede the dst write (dst may alias them)
    const int P = n_used*a.ntok;
    __shared__ int s_fin;
    if (tid == 0) {
        s_fin = atomicAdd(&a.cnt[8], 1) == P - 1;
    }
    __syncthreads();
    if (!s_fin) {
        return;
    }
    if (tid == 0) {
        a.cnt[8] = 0;
    }
    __threadfence();
    for (int tk2 = 0; tk2 < a.ntok; ++tk2) {
        const int64_t wb0 = (int64_t) tk2*n_used*md;
        for (int i = tid; i < md; i += WG) {
            float acc = __ldcg(&a.wbuf[wb0 + i]);
            for (int s2 = 1; s2 < n_used; ++s2) {
                acc = acc + __ldcg(&a.wbuf[wb0 + (int64_t) s2*md + i]);
            }
            a.dst[(int64_t) tk2*a.ds1 + i] = acc;
        }
    }
}

#define MACH1_EXP_SLOT_SMEM (16384*sizeof(uint32_t) + 256*sizeof(float2) + (2048 + 2048 + 512 + 512 + 512)*sizeof(float))


// llama.cpp records the decode graph with stream capture, and a launch API the
// capture rejects poisons the whole capture instead of failing locally - so
// probe cooperative capture once, on a private stream. A captured node is
// recorded and never executed, so the zeroed arguments are never dereferenced.
static bool mach1_exp_mega_capture_ok(const void * fn, const int wg, const size_t smem) {
    cudaStream_t st = nullptr;
    if (cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    bool ok = false;
    if (cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed) == cudaSuccess) {
        mach1_exp_mega_args a = {};
        void * kargs[] = { (void *) &a };
        ok = cudaLaunchCooperativeKernel(fn,
                                         dim3(1, 1, 1), dim3(wg, 1, 1),
                                         kargs, smem, st) == cudaSuccess;
        cudaGraph_t gr = nullptr;
        if (cudaStreamEndCapture(st, &gr) == cudaSuccess && gr != nullptr) {
            CUDA_CHECK(cudaGraphDestroy(gr));
        } else {
            ok = false;
        }
    }
    cudaStreamDestroy(st);
    cudaGetLastError();
    return ok;
}

// grid for the megakernel, or 0 when this device cannot carry it (no
// cooperative launch, not enough opt-in shared memory, or a capture that
// rejects the launch API). Cached per device; the probes are host-side calls
// with no stream ordering, so they are legal wherever the region first matches.
static int mach1_exp_mega_grid(int device, const void * fn, const int slot, const int wg, const size_t smem) {
    static std::mutex mtx;
    // Slots 10/11 belong to the round-333 two-plane kernels. QONCE uses 12,
    // its QIP producer variant 13, the standard-output STD variant 14, the
    // DNCT=32 down-phase form 15, the nt32 wide-counter (PCAP=256) QONCE
    // forms 16 (DNCT=128) and 17 (DNCT=32), the pair-ORDER QONCE forms
    // 18 (DNCT=128) and 19 (DNCT=32), the PFLAGS forms 20 (DNCT=128)
    // and 21 (DNCT=32), and the NOSTAGE coop forms 22 (DNCT=128) and
    // 23 (DNCT=32): prompt and p16 decode can probe several functions in
    // the same process, so sharing a slot would reuse the wrong function's
    // occupancy/capture result.
    static int cache[GGML_CUDA_MAX_DEVICES][24] = { { 0 } };

    std::lock_guard<std::mutex> lock(mtx);
    if (cache[device][slot] != 0) {
        return cache[device][slot] > 0 ? cache[device][slot] : 0;
    }
    cache[device][slot] = -1;
    int coop = 0;
    int bps  = 0;
    if (ggml_cuda_info().devices[device].smpbo >= smem &&
        cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, device) == cudaSuccess && coop != 0 &&
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int) smem) == cudaSuccess &&
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, fn, wg, smem) == cudaSuccess &&
        bps > 0 && mach1_exp_mega_capture_ok(fn, wg, smem)) {
        cache[device][slot] = bps*ggml_cuda_info().devices[device].nsm;
        cudaFuncAttributes at;
        if ((mach1_env_int("GGML_MACH1_TIME", 0) == 1 || mach1_debug_on()) &&
            cudaFuncGetAttributes(&at, fn) == cudaSuccess) {
            fprintf(stderr, "mach1-time: exp_mega slot=%d wg=%d regs=%d spill=%d B smem=%d B %d blk/SM\n",
                    slot, wg, at.numRegs, (int) at.localSizeBytes,
                    (int) at.sharedSizeBytes, bps);
        }
    }
    cudaGetLastError();
    return cache[device][slot] > 0 ? cache[device][slot] : 0;
}

// vbpb: a block covers vbpb consecutive row tiles of one item. Packing a block
// full minimises blocks, which is exactly wrong - a step's thread-work is
// fixed, so fewer blocks means fewer SMs carrying it. Spread each phase over
// the whole grid instead and let the surplus virtual blocks idle.
static void mach1_exp_mega_split(const int nb, const int nops, const int rows, int & vbpb, int & tpo,
                                 const int vbmax = MACH1_EXP_MEGA_VB) {
    const int want = (nb + nops - 1)/nops;
    vbpb = std::max(1, std::min(vbmax, (rows + want - 1)/want));
    tpo  = (rows + vbpb - 1)/vbpb;
}

// GG lane fork (GGML_MACH1_GG_FORK): the cold fused kernel and the hot banks'
// decode/gather/GEMM/scatter chain touch disjoint scr_v rows (hotmask
// partitions pairs by group), so the cold kernel can run on a side stream
// CONCURRENT with the chain instead of serializing ahead of it. The GG path
// never runs under capture (gg_ok guard), so lazy creation here is safe.
struct mach1_gg_fork {
    cudaStream_t s2 = nullptr;
    cudaEvent_t  e0 = nullptr;   // main -> s2: eu16 + hotmask ready
    cudaEvent_t  e1 = nullptr;   // s2 -> main: cold kernel done
};

static mach1_gg_fork * mach1_gg_fork_get(int device) {
    static mach1_gg_fork st[GGML_CUDA_MAX_DEVICES];
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    mach1_gg_fork & f = st[device];
    if (f.s2 == nullptr) {
        if (cudaStreamCreateWithFlags(&f.s2, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&f.e0, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&f.e1, cudaEventDisableTiming) != cudaSuccess) {
            cudaGetLastError();
            f.s2 = nullptr;
            return nullptr;
        }
    }
    return &f;
}

// GG int8 support probe (GGML_MACH1_GG_I8): the grouped-batched API rejected
// 16F on CUDA 12.8 (status 15) - int8 is probed here for both APIs at first
// engagement. Per-bucket GemmBatchedEx is the shipping form; the grouped call
// is a logged receipt only. The tiny GEMM also checks the TN int8 layout
// numerically (all-ones A/B -> C == k).
static bool mach1_gg_i8_ok() {
    static int ok = -1;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    if (ok >= 0) {
        return ok != 0;
    }
    ok = 0;
    int8_t  * ab = nullptr;
    int32_t * c  = nullptr;
    void   ** pv = nullptr;
    cublasHandle_t h = nullptr;
    if (cudaMalloc((void **) &ab, 2*32*32) == cudaSuccess &&
        cudaMalloc((void **) &c, 32*32*sizeof(int32_t)) == cudaSuccess &&
        cudaMalloc((void **) &pv, 3*sizeof(void *)) == cudaSuccess &&
        cublasCreate(&h) == CUBLAS_STATUS_SUCCESS) {
        CUDA_CHECK(cudaMemset(ab, 1, 2*32*32));
        void * hp[3] = { ab, ab + 32*32, c };
        CUDA_CHECK(cudaMemcpy(pv, hp, sizeof(hp), cudaMemcpyHostToDevice));
        const int32_t onei = 1, zeroi = 0;
        const cublasStatus_t st = cublasGemmBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N,
            32, 32, 32,
            &onei,  (const void **) pv,       CUDA_R_8I,  32,
                    (const void **) (pv + 1), CUDA_R_8I,  32,
            &zeroi, (      void **) (pv + 2), CUDA_R_32I, 32,
            1, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
        int32_t c00 = 0;
        if (st == CUBLAS_STATUS_SUCCESS) {
            CUDA_CHECK(cudaMemcpy(&c00, c, sizeof(int32_t), cudaMemcpyDeviceToHost));
        }
        ok = (st == CUBLAS_STATUS_SUCCESS && c00 == 32) ? 1 : 0;
#if CUBLAS_VER_MAJOR > 12 || (CUBLAS_VER_MAJOR == 12 && CUBLAS_VER_MINOR >= 5)
        const cublasOperation_t ta[1] = { CUBLAS_OP_T }, tb[1] = { CUBLAS_OP_N };
        const int md[1] = { 32 }, nd[1] = { 32 }, kd[1] = { 32 };
        const int la[1] = { 32 }, lb[1] = { 32 }, lc[1] = { 32 }, gsz[1] = { 1 };
        const int32_t al[1] = { 1 }, be[1] = { 0 };
        const void * ga[1] = { ab };
        const void * gb[1] = { ab + 32*32 };
        void       * gc[1] = { c };
        const cublasStatus_t stg = cublasGemmGroupedBatchedEx(h, ta, tb, md, nd, kd,
            al, ga, CUDA_R_8I, la, gb, CUDA_R_8I, lb,
            be, gc, CUDA_R_32I, lc, 1, gsz, CUBLAS_COMPUTE_32I);
#else
        // cublasGemmGroupedBatchedEx needs cuBLAS >= 12.5; stg only feeds the
        // debug log, the probe verdict above comes from GemmBatchedEx.
        const cublasStatus_t stg = CUBLAS_STATUS_NOT_SUPPORTED;
#endif
        if (mach1_debug_on()) {
            fprintf(stderr, "mach1: GG i8 probe GemmBatchedEx status=%d c00=%d, GroupedBatchedEx status=%d\n",
                    (int) st, c00, (int) stg);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    if (h != nullptr) {
        cublasDestroy(h);
    }
    if (ab != nullptr) {
        CUDA_CHECK(cudaFree(ab));
    }
    if (c != nullptr) {
        CUDA_CHECK(cudaFree(c));
    }
    if (pv != nullptr) {
        CUDA_CHECK(cudaFree(pv));
    }
    cudaGetLastError();
    return ok != 0;
}

void ggml_cuda_op_mach1_exp_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kept  = dst->src[0];
    const ggml_tensor * dem   = dst->src[1];
    const ggml_tensor * su    = dst->src[2];
    const ggml_tensor * sv    = dst->src[3];
    const ggml_tensor * tlut  = dst->src[4];
    const ggml_tensor * remap = dst->src[5];
    const ggml_tensor * ids   = dst->src[6];
    const ggml_tensor * x     = dst->src[7];
    const ggml_tensor * gamma = dst->src[8];   // payload v3 (V8 walk) only

    const bool v8 = tlut->ne[0] == 8;
    GGML_ASSERT(v8 == (gamma != nullptr));

    GGML_ASSERT(kept->type  == GGML_TYPE_I16);
    GGML_ASSERT(su->type    == GGML_TYPE_F16);
    GGML_ASSERT(sv->type    == GGML_TYPE_F16);
    GGML_ASSERT(tlut->type  == (v8 ? GGML_TYPE_F16 : GGML_TYPE_F32));
    GGML_ASSERT(remap->type == GGML_TYPE_I32);
    GGML_ASSERT(ids->type   == GGML_TYPE_I32);
    GGML_ASSERT(x->type     == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(kept));
    GGML_ASSERT(ggml_is_contiguous(su));
    GGML_ASSERT(ggml_is_contiguous(sv));
    GGML_ASSERT(ggml_is_contiguous(tlut));
    GGML_ASSERT(ggml_is_contiguous(remap));
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(dst));
    // the walk kernel hardcodes the rates like the Vulkan shader
    GGML_ASSERT(kept->ne[0] == (v8 ? 24 : 32));
    if (dem != nullptr) {
        GGML_ASSERT(dem->type == GGML_TYPE_I16);
        GGML_ASSERT(ggml_is_contiguous(dem));
        GGML_ASSERT(dem->ne[0] == 16);
    }
    if (gamma != nullptr) {
        GGML_ASSERT(gamma->type == GGML_TYPE_F16);
        GGML_ASSERT(ggml_is_contiguous(gamma));
    }

    if (mach1_env_int("GGML_MACH1_DEBUG", 0) >= 2) {
        mach1_lattice_report(v8 ? "exp_tlut8" : "exp_tlut2", tlut->data, ggml_nelements(tlut),
                             tlut->type == GGML_TYPE_F16, ctx.stream());
        if (gamma != nullptr) {
            mach1_lattice_report("wave_gamma", gamma->data, ggml_nelements(gamma), true, ctx.stream());
        }
    }

    const int n        = (int) su->ne[0];
    const int m        = (int) sv->ne[0];
    const int n_kept   = (int) kept->ne[2];
    const int n_groups = n_kept + (int)(dem ? dem->ne[2] : 0);
    const int n_used   = (int) ids->ne[0];
    const int n_tok    = (int) ids->ne[1];
    const int P        = n_used*n_tok;
    const int xne1     = (int) x->ne[1];
    const int ids_s0   = (int)(ids->nb[0]/sizeof(int32_t));
    const int ids_s1   = (int)(ids->nb[1]/sizeof(int32_t));
    GGML_ASSERT(n <= 2048 && m <= 2048);   // exp_u/exp_out shared bounds
    GGML_ASSERT(n_groups <= 512);          // exp_group shared bound

    ggml_cuda_pool_alloc<int32_t> scr_i_alloc(ctx.pool(), (size_t) 4*n_groups + P);
    ggml_cuda_pool_alloc<float>   scr_f_alloc(ctx.pool(), (size_t) P*n + (size_t) P*m);
    int32_t * scr_i = scr_i_alloc.get();
    float   * scr_u = scr_f_alloc.get();
    float   * scr_v = scr_u + (size_t) P*n;

    cudaStream_t stream = ctx.stream();

    // p4 repack of the V8 tlut (one-time, cached); nullptr => fp16 gathers
    mach1_p4_table p4t = {nullptr, nullptr};
    if (v8) {
        GGML_ASSERT(tlut->ne[1] == 32768);
        p4t = mach1_exp_p4_table(tlut->data, stream);
    }
    const bool p4 = p4t.packed != nullptr;

    const uint16_t * kept_d = (const uint16_t *) kept->data;
    const uint16_t * dem_d  = dem != nullptr ? (const uint16_t *) dem->data : kept_d;

    // the split path's integer walk reads the nibble table from GLOBAL - worth
    // less than the megakernel's shared copy, and only reachable when the
    // region did not fuse (round 46, arm PA)
    const int zmode = p4t.zpacked != nullptr ? mach1_exp_zdp_mode() : 0;

    char shp[96];
    snprintf(shp, sizeof(shp), " m=%d n=%d n_used=%d n_tok=%d ng=%d v8=%d p4=%d zdp=%d",
             m, n, n_used, n_tok, n_groups, (int) v8, (int) p4, zmode);

    // GGML_MACH1_FOLD=1 (UNCERTIFIED, V8 only): the whole EXP op as one GEMV
    // per used slot from a per-group W_eff bank (gamma, su/sv and both
    // Hadamards folded at build; exp_out folded away). ~537 MB fp16 per
    // tensor - the budget check keeps bandwidth-poor / small-VRAM parts on
    // the exact chain per tensor.
    if (mach1_fold_enabled() && v8 && n_tok == 1) {
        const int n_experts = (int) remap->ne[0];
        const auto exp_fold_fill = [&](half * bdst) {
            ggml_cuda_pool_alloc<int32_t> ones_alloc(ctx.pool(), (size_t) n_groups);
            CUDA_CHECK(cudaMemsetAsync(ones_alloc.get(), 1, (size_t) n_groups*sizeof(int32_t), stream));
            const ggml_cuda_kernel_launch_params dec_params =
                ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, n_groups),
                                               dim3(256, 1, 1), 0, stream);
            if (p4) {
                mach1_launch(mach1_exp_dense_decode_kernel<true, true>, dec_params,
                    kept_d, dem_d, (const void *) tlut->data, p4t.packed, p4t.levels,
                    (const int32_t *) ones_alloc.get(), bdst, m, n, n_kept, n_groups);
            } else {
                mach1_launch(mach1_exp_dense_decode_kernel<true, false>, dec_params,
                    kept_d, dem_d, (const void *) tlut->data,
                    (const uint32_t *) nullptr, (const float *) nullptr,
                    (const int32_t *) ones_alloc.get(), bdst, m, n, n_kept, n_groups);
            }
            ggml_cuda_pool_alloc<int32_t> inv_alloc(ctx.pool(), (size_t) n_groups);
            mach1_launch(mach1_fold_inv_remap_kernel,
                ggml_cuda_kernel_launch_params(dim3((n_experts + 255)/256, 1, 1),
                                               dim3(256, 1, 1), 0, stream),
                (const int32_t *) remap->data, inv_alloc.get(), n_experts, n_kept);
            std::vector<int32_t> hinv(n_groups);
            CUDA_CHECK(cudaMemcpyAsync(hinv.data(), inv_alloc.get(),
                                       (size_t) n_groups*sizeof(int32_t),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            ggml_cuda_pool_alloc<float> f32(ctx.pool(), (size_t) m*n);
            const int Mb = m/16, ty = n/16;
            for (int g = 0; g < n_groups; ++g) {
                const int64_t e  = hinv[g];
                half *        wg = bdst + (int64_t) g*m*n;
                mach1_launch(mach1_fold_widen_gamma_kernel,
                    ggml_cuda_kernel_launch_params(dim3((unsigned)(((int64_t) m*n + 255)/256), 1, 1),
                                                   dim3(256, 1, 1), 0, stream),
                    (const half *) wg, (const half *) gamma->data + e*(Mb + ty), f32.get(), m, n);
                mach1_launch(mach1_fold_rows_h_kernel<256>,
                    ggml_cuda_kernel_launch_params(dim3(m, 1, 1), dim3(256, 1, 1), 0, stream),
                    f32.get(), (const half *) su->data + e*n, n, 1.0f/sqrtf((float) n));
                mach1_launch(mach1_fold_cols_h_kernel<256>,
                    ggml_cuda_kernel_launch_params(dim3(n, 1, 1), dim3(256, 1, 1), 0, stream),
                    (const float *) f32.get(), (const half *) sv->data + e*m, wg, m, n, 1.0f/sqrtf((float) m));
            }
        };
        // NOTE: the experts stay on the fp16 fold bank. Serving them from
        // Q8_0 through mul_mat_vec_q wants its ids-batched MUL_MAT_ID form
        // (src0 [n, m, n_groups] + ids), which needs the bank to be a real
        // graph tensor - that is the load-time materialization work item, not
        // something to fake per op: resolving the group host-side would cost a
        // stream sync on every one of the ~120 expert ops per token.
        half * wb = mach1_bank_get(kept->data, (size_t) n_groups*m*n, ctx.device, stream, exp_fold_fill);
        if (wb != nullptr) {
            mach1_timed(stream, std::string("exp_fold_mv") + shp, [&]() {
            mach1_launch(mach1_exp_fold_mv_kernel,
                ggml_cuda_kernel_launch_params(dim3((unsigned)((m + 7)/8), 1, n_used), dim3(256, 1, 1), 0, stream),
                (const half *) wb, (const int32_t *) remap->data, (const int32_t *) ids->data,
                (const float *) x->data, (float *) dst->data,
                m, n, xne1, n_used, n_kept, ids_s0, ids_s1);
            });
            return;
        }
    }

    // decode: distinct top-k slots need no grouping — one fused kernel per
    // slot replaces exp_group/exp_u/exp_walk (exp_out is unchanged). Unlike
    // the rt fusion, the redundant-u grids here are small (m/16 = 32 for
    // gate/up, and the down projection's FWHT is over n = 512). MEASURED
    // (H200, paired, 3 rounds): 64.8/64.8/64.5 vs base 60.1/58.5/58.4 tok/s
    // tg128, token stream identical — default ON.
    static const bool fuse_on = mach1_env_int("GGML_MACH1_FUSE_EXP", 1) != 0 &&
                                mach1_env_int("GGML_MACH1_ABLATE", 0) == 0;
    if (fuse_on && v8 && n_tok == 1) {
        // persistent tier: one launch per op, resident nibble codebook,
        // exp_out folded (see mach1_exp_persist_kernel)
        static const bool persist = mach1_env_int("GGML_MACH1_EXP_PERSIST", 0) != 0;
        if (persist && p4 && n_used <= 8 && n <= 2048 && m <= 2048 &&
            !mach1_ablate(MACH1_ABLATE_EXP_WALK) && !mach1_ablate(MACH1_ABLATE_EXP_OUT)) {
            const size_t smem = 16384*sizeof(uint32_t) + 256*sizeof(float2)
                              + (size_t) n_used*n*sizeof(float) + (size_t) m*sizeof(float);
            // worst case across shapes: 8 slots x n = 2048 plus m = 2048 out buffer
            const size_t smem_max = 16384*sizeof(uint32_t) + 256*sizeof(float2)
                                  + (size_t) 8*2048*sizeof(float) + (size_t) 2048*sizeof(float);
            const size_t smpbo = ggml_cuda_info().devices[ctx.device].smpbo;
            static bool limit_set[GGML_CUDA_MAX_DEVICES] = { false };
            if (!limit_set[ctx.device] && smem_max <= smpbo) {
                CUDA_CHECK(cudaFuncSetAttribute(mach1_exp_persist_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_max));
                limit_set[ctx.device] = true;
            }
            if (limit_set[ctx.device] && smem <= smem_max) {
                CUDA_CHECK(cudaMemsetAsync(scr_i, 0, (size_t) n_used*sizeof(int32_t), stream));
                const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
                mach1_timed(stream, std::string("exp_persist") + shp, [&]() {
                mach1_launch(mach1_exp_persist_kernel,
                    ggml_cuda_kernel_launch_params(dim3(nsm, 1, 1), dim3(1024, 1, 1), smem, stream),
                    kept_d, p4t.packed, p4t.levels,
                    (const half *) su->data, (const half *) sv->data,
                    (const int32_t *) remap->data, (const int32_t *) ids->data,
                    (const float *) x->data, scr_v, (float *) dst->data,
                    (const half *) gamma->data, scr_i,
                    m, n, xne1, n_used, n_kept, ids_s0, ids_s1);
                });
                return;
            }
        }
        const ggml_cuda_kernel_launch_params fparams =
            ggml_cuda_kernel_launch_params(dim3(m/16, 1, n_used), dim3(128, 1, 1), 0, stream);
        static const bool expfast = mach1_env_int("GGML_MACH1_EXPFAST", 0) != 0;
        static const bool exprows = mach1_env_int("GGML_MACH1_EXP_ROWS", 0) != 0;
        const int tiles_y_e = n/16;
        if (exprows && p4 && n % 16 == 0 && tiles_y_e <= 128) {
            // CT is the smallest power-of-two multiple of 32 that covers
            // tiles_y, so no thread is idle in the walk; RG then takes the
            // block to 512/1024 threads instead of 128.
            mach1_timed(stream, std::string("exp_rows") + shp, [&]() {
            const dim3 gsz(m/16, 1, n_used);
            if (tiles_y_e <= 32) {
                mach1_launch(mach1_exp_fused_rows_kernel<true, 512, 16>,
                    ggml_cuda_kernel_launch_params(gsz, dim3(512, 1, 1), 0, stream),
                    kept_d, (const void *) tlut->data, p4t.packed, p4t.levels,
                    (const half *) su->data, (const int32_t *) remap->data,
                    (const int32_t *) ids->data, (const float *) x->data,
                    scr_v, (const half *) gamma->data, m, n, xne1, n_used, n_kept, ids_s0, ids_s1);
            } else if (tiles_y_e <= 64) {
                mach1_launch(mach1_exp_fused_rows_kernel<true, 1024, 16>,
                    ggml_cuda_kernel_launch_params(gsz, dim3(1024, 1, 1), 0, stream),
                    kept_d, (const void *) tlut->data, p4t.packed, p4t.levels,
                    (const half *) su->data, (const int32_t *) remap->data,
                    (const int32_t *) ids->data, (const float *) x->data,
                    scr_v, (const half *) gamma->data, m, n, xne1, n_used, n_kept, ids_s0, ids_s1);
            } else {
                mach1_launch(mach1_exp_fused_rows_kernel<true, 1024, 8>,
                    ggml_cuda_kernel_launch_params(gsz, dim3(1024, 1, 1), 0, stream),
                    kept_d, (const void *) tlut->data, p4t.packed, p4t.levels,
                    (const half *) su->data, (const int32_t *) remap->data,
                    (const int32_t *) ids->data, (const float *) x->data,
                    scr_v, (const half *) gamma->data, m, n, xne1, n_used, n_kept, ids_s0, ids_s1);
            }
            });
        } else
        mach1_timed(stream, std::string("exp_fused") + shp, [&]() {
        if (p4 && zmode == 2) {
            mach1_launch(mach1_exp_fused_kernel<true, true, 2>, fparams,
                kept_d, (const void *) tlut->data, p4t.packed, p4t.levels,
                (const half *) su->data, (const int32_t *) remap->data,
                (const int32_t *) ids->data, (const float *) x->data,
                scr_v, (const half *) gamma->data, m, n, xne1, n_used, n_kept, ids_s0, ids_s1,
                p4t.zpacked, p4t.zstep);
        } else if (p4 && zmode == 1) {
            mach1_launch(mach1_exp_fused_kernel<true, true, 1>, fparams,
                kept_d, (const void *) tlut->data, p4t.packed, p4t.levels,
                (const half *) su->data, (const int32_t *) remap->data,
                (const int32_t *) ids->data, (const float *) x->data,
                scr_v, (const half *) gamma->data, m, n, xne1, n_used, n_kept, ids_s0, ids_s1,
                p4t.zpacked, p4t.zstep);
        } else if (p4 && expfast) {
            mach1_launch(mach1_exp_fused_kernel<true, true>, fparams,
                kept_d, (const void *) tlut->data, p4t.packed, p4t.levels,
                (const half *) su->data, (const int32_t *) remap->data,
                (const int32_t *) ids->data, (const float *) x->data,
                scr_v, (const half *) gamma->data, m, n, xne1, n_used, n_kept, ids_s0, ids_s1,
                (const uint32_t *) nullptr, 0.0f);
        } else if (p4) {
            mach1_launch(mach1_exp_fused_kernel<true, false>, fparams,
                kept_d, (const void *) tlut->data, p4t.packed, p4t.levels,
                (const half *) su->data, (const int32_t *) remap->data,
                (const int32_t *) ids->data, (const float *) x->data,
                scr_v, (const half *) gamma->data, m, n, xne1, n_used, n_kept, ids_s0, ids_s1,
                (const uint32_t *) nullptr, 0.0f);
        } else {
            mach1_launch(mach1_exp_fused_kernel<false, false>, fparams,
                kept_d, (const void *) tlut->data,
                (const uint32_t *) nullptr, (const float *) nullptr,
                (const half *) su->data, (const int32_t *) remap->data,
                (const int32_t *) ids->data, (const float *) x->data,
                scr_v, (const half *) gamma->data, m, n, xne1, n_used, n_kept, ids_s0, ids_s1,
                (const uint32_t *) nullptr, 0.0f);
        }
        });
        if (!mach1_ablate(MACH1_ABLATE_EXP_OUT))
        mach1_timed(stream, std::string("exp_out") + shp, [&]() {
        if (mach1_fwht_wg512()) {
            mach1_launch(mach1_exp_out_kernel<512>,
                ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(512, 1, 1), 0, stream),
                (const half *) sv->data, (const int32_t *) ids->data, scr_v, (float *) dst->data,
                m, n_used, ids_s0, ids_s1);
        } else {
            mach1_launch(mach1_exp_out_kernel<256>,
                ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(256, 1, 1), 0, stream),
                (const half *) sv->data, (const int32_t *) ids->data, scr_v, (float *) dst->data,
                m, n_used, ids_s0, ids_s1);
        }
        });
        return;
    }

    if (!mach1_ablate(MACH1_ABLATE_EXP_GRP))
    mach1_timed(stream, std::string("exp_group") + shp, [&]() {
    mach1_launch(mach1_exp_group_kernel,
        ggml_cuda_kernel_launch_params(dim3(1, 1, 1), dim3(256, 1, 1), 0, stream),
        (const int32_t *) remap->data, (const int32_t *) ids->data, scr_i,
        n_used, n_tok, n_kept, n_groups, ids_s0, ids_s1);
    });

    if (!mach1_ablate(MACH1_ABLATE_EXP_U))
    mach1_timed(stream, std::string("exp_u") + shp, [&]() {
    if (mach1_fwht_wg512()) {
        mach1_launch(mach1_exp_u_kernel<512>,
            ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(512, 1, 1), 0, stream),
            (const half *) su->data, (const int32_t *) ids->data, (const float *) x->data,
            scr_u, n, n_used, xne1, ids_s0, ids_s1);
    } else {
        mach1_launch(mach1_exp_u_kernel<256>,
            ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(256, 1, 1), 0, stream),
            (const half *) su->data, (const int32_t *) ids->data, (const float *) x->data,
            scr_u, n, n_used, xne1, ids_s0, ids_s1);
    }
    });

    static const int dense_min0 = mach1_env_int("GGML_MACH1_DENSE_MIN", 1024);
    // PPLOW: the dense-apply floor is a PAIR count (n_used*n_tok) and so falls
    // with the ubatch - at nt = 128 it is already borderline. A ubatch that
    // clears the PP floor is a prefill shape by construction (the PP floor is
    // never below 64, every decode width tops out at 32), so admitting it here
    // cannot reach a decode batch.
    const int dense_min = (mach1_pplow_on() && n_tok >= mach1_pp_min()) ? 1 : dense_min0;
    ggml_cuda_pool_alloc<half> bank_alloc(ctx.pool());
    // fused dense: decode-to-shared apply, no [n_groups, m, n] bank round trip.
    // n == 512 only (r262): the single-slice form wins 18% on the down op,
    // but the k-sliced n = 2048 form LOSES 16% - at large nt the bank's apply
    // reads mostly hit L2, so the round-trip saving is smaller than the
    // slicing costs (4x blocks, the partials pass, per-block fixed work).
    static const bool fused_dense = mach1_env_int("GGML_MACH1_EXP_FUSED_DENSE", 1) != 0;
    // integer dense: z-nibble tiles + dp4a (see the kernel comment) - the
    // n = 2048 gate/up shapes where the fp16 tile would need k-slicing
    static const int zdp_apply = mach1_env_int("GGML_MACH1_EXP_ZDP_APPLY", 2);
    if (P >= dense_min && zdp_apply != 0 && v8 && p4 && p4t.zpacked != nullptr &&
        (n == 2048 || n == 512)) {
        // level 2: quantize each (pair, tile) once instead of once per row block
        ggml_cuda_pool_alloc<uint32_t> q8_alloc(ctx.pool());
        uint32_t * scr_q8 = nullptr;
        if (zdp_apply >= 2) {
            const int64_t total = (int64_t) P*(n/16);
            scr_q8 = q8_alloc.alloc((size_t) total*6);
            mach1_timed(stream, std::string("exp_quant_u") + shp, [&]() {
            mach1_launch(mach1_exp_quant_u_kernel,
                ggml_cuda_kernel_launch_params(dim3((unsigned)((total + 255)/256), 1, 1),
                                               dim3(256, 1, 1), 0, stream),
                (const float *) scr_u, scr_q8, n, total);
            });
        }
        const ggml_cuda_kernel_launch_params zp =
            ggml_cuda_kernel_launch_params(dim3(m/16, n_groups, 1), dim3(128, 1, 1), 0, stream);
        // ZPC: the pair-chunk width. acc[PC][16] is PC*16 registers - at 8
        // that is 128 for the accumulators alone, spill territory; a narrower
        // chunk trades one more swz pass for register headroom.
        static const int zpc = mach1_env_int("GGML_MACH1_EXP_ZPC", 4);
        if (mach1_env_int("GGML_MACH1_TIME", 0) == 1) {
            static bool once_att = false;
            if (!once_att) {
                once_att = true;
                cudaFuncAttributes at;
                const struct { const char * name; const void * fn; } vars[] = {
                    { "pc2",    (const void *) mach1_exp_zdp_apply_kernel<2, 1, 1> },
                    { "pc4",    (const void *) mach1_exp_zdp_apply_kernel<4, 1, 1> },
                    { "pc8",    (const void *) mach1_exp_zdp_apply_kernel<8, 1, 1> },
                    { "pc4lb4", (const void *) mach1_exp_zdp_apply_kernel<4, 1, 1, 4> },
                    { "pc4lb6", (const void *) mach1_exp_zdp_apply_kernel<4, 1, 1, 6> },
                    { "mma0",   (const void *) mach1_exp_zdp_mma_kernel<0> },
                    { "mma1",   (const void *) mach1_exp_zdp_mma_kernel<1> },
                };
                for (const auto & v : vars) {
                    if (cudaFuncGetAttributes(&at, v.fn) == cudaSuccess) {
                        fprintf(stderr, "mach1-time: exp_zdp_apply %s regs=%d spill=%d B\n",
                                v.name, at.numRegs, (int) at.localSizeBytes);
                    }
                }
                cudaGetLastError();
            }
        }
        // PP lane: grouped-GEMM form (GGML_MACH1_EXP_APPLY_GG). One D2H fetch
        // of the group counts per call (prefill runs graph-free, so the sync
        // is legal) picks the HOT groups (cnt >= GG_MIN); those decode once to
        // transient fp16 banks and apply as per-group cuBLAS GEMMs over
        // gathered rows, cold groups keep the fused fp16 kernel via hotmask.
        // Pair counts pad up a bucket ladder so every cuBLAS handle only ever
        // sees ONE shape (see rt_apply_tc: the heuristic re-runs on change).
        static const bool zgg  = mach1_env_int("GGML_MACH1_EXP_APPLY_GG", 0) != 0;
        // PP lane: fp16-A form (LUT expansion, B from a fp16 copy of scr_u;
        // no q8 records, no du shuffles). Takes priority over the s8 form.
        static const bool zf16 = mach1_env_int("GGML_MACH1_EXP_APPLY_FP16", 0) != 0;
        // the GG host sync is illegal mid-capture; graph gating in ggml-cuda.cu
        // keeps these cgraphs direct, this guard covers any other capture path
        bool gg_ok = zgg && n_tok >= mach1_pp_gg_min();
        if (gg_ok) {
            cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
            if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) {
                cudaGetLastError();
                gg_ok = false;
            }
        }
        if (gg_ok) {
            mach1_pplow_mark("exp grouped-GEMM", n_tok);
            if (mach1_debug_on()) {
                static std::once_flag once;
                std::call_once(once, []() {
                    fprintf(stderr, "mach1: PP expert grouped-GEMM ENGAGED (hot groups)\n");
                });
            }
            if (!mach1_ablate(MACH1_ABLATE_EXP_WALK))
            mach1_timed(stream, std::string("exp_gg") + shp, [&]() {
                std::vector<int32_t> hcnt(n_groups);
                mach1_timed(stream, std::string("exp_gg_sync") + shp, [&]() {
                CUDA_CHECK(cudaMemcpyAsync(hcnt.data(), scr_i, (size_t) n_groups*sizeof(int32_t),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                });
                // default 64: the L40S sweep is flat within noise for 32-128
                // hot-pair thresholds and 16 measurably loses (bank decode +
                // GEMM launches for groups the fused kernel absorbs cheaper)
                static const int    gg_min0 = std::max(mach1_env_int("GGML_MACH1_GG_MIN", 64), 1);
                static const size_t gg_cap  = (size_t) mach1_env_int("GGML_MACH1_GG_MB", 200) << 20;
                static const int    zprobe  = mach1_env_int("GGML_MACH1_ZMMA_PROBE", 0);
                static const int    gg_fork = mach1_env_int("GGML_MACH1_GG_FORK", 0);
                static const bool   gg_cs8  = mach1_env_int("GGML_MACH1_GG_COLD_S8", 0) != 0;
                static const bool   gg_i8e  = mach1_env_int("GGML_MACH1_GG_I8", 0) != 0;
                // s8 cold slice: the fused s8 MMA form takes the cold groups
                // (integer lattice class); needs the QONCE records
                const bool cold_s8 = gg_cs8 && scr_q8 != nullptr;
                // int8 hot lane: i8 banks + per-row-quantized B, 32I batched
                // GEMMs - Ada's imma rate is ~2x its fp16 TC rate
                static const bool gg_i8 = gg_i8e && mach1_gg_i8_ok();
                if (mach1_debug_on()) {
                    static std::once_flag once_i8;
                    std::call_once(once_i8, [&]() {
                        if (gg_i8) {
                            fprintf(stderr, "mach1: GG INT8 hot lane ENGAGED (i8 banks, per-row B scales, 32I batched GEMM)\n");
                        } else if (gg_i8e) {
                            fprintf(stderr, "mach1: GG INT8 hot lane requested but cuBLAS i8 probe FAILED - fp16 lane\n");
                        }
                        if (cold_s8) {
                            fprintf(stderr, "mach1: GG cold slice s8 MMA ENGAGED\n");
                        } else if (gg_cs8) {
                            fprintf(stderr, "mach1: GG cold s8 requested but no QONCE records - fp16 cold\n");
                        }
                    });
                }
                // i8 banks are half the bytes, so the same cap holds 2x the hot set
                const size_t gg_esz = gg_i8 ? sizeof(int8_t) : sizeof(half);
                int gg_min = gg_min0;
                int n_hot  = 0;
                for (;;) {
                    n_hot = 0;
                    for (int g = 0; g < n_groups; ++g) {
                        n_hot += hcnt[g] >= gg_min;
                    }
                    if ((size_t) n_hot*m*n*gg_esz <= gg_cap) {
                        break;
                    }
                    gg_min *= 2;   // shrink the hot set until the banks fit
                }
                int n_cold = 0;
                for (int g = 0; g < n_groups; ++g) {
                    n_cold += hcnt[g] > 0 && hcnt[g] < gg_min;
                }
                // fp16 copy of scr_u: B rows for the fp16 GEMMs, and the cold
                // groups' fused fp16 kernel reads it too; unused when both
                // the hot lane and the cold slice run integer forms
                ggml_cuda_pool_alloc<half> eu16_alloc(ctx.pool());
                half * eu16 = nullptr;
                if (!gg_i8 || !cold_s8) {
                    eu16 = eu16_alloc.alloc((size_t) P*n);
                    mach1_timed(stream, std::string("exp_gg_u16") + shp, [&]() {
                        const int64_t k = (int64_t) P*n;
                        mach1_launch(mach1_rt_u16_kernel,
                            ggml_cuda_kernel_launch_params(
                                dim3((unsigned) ((k + 255)/256), 1, 1), dim3(256, 1, 1), 0, stream),
                            (const float *) scr_u, eu16, k);
                    });
                }
                ggml_cuda_pool_alloc<int8_t>  hmask_alloc(ctx.pool());
                ggml_cuda_pool_alloc<int32_t> meta_alloc(ctx.pool());
                ggml_cuda_pool_alloc<half>    gbank_alloc(ctx.pool());
                ggml_cuda_pool_alloc<half>    ub_alloc(ctx.pool());
                ggml_cuda_pool_alloc<float>   cb_alloc(ctx.pool());
                ggml_cuda_pool_alloc<int8_t>  gbank8_alloc(ctx.pool());
                ggml_cuda_pool_alloc<int8_t>  ub8_alloc(ctx.pool());
                ggml_cuda_pool_alloc<int32_t> cb32_alloc(ctx.pool());
                ggml_cuda_pool_alloc<float>   wsc_alloc(ctx.pool());
                ggml_cuda_pool_alloc<float>   bsc_alloc(ctx.pool());
                int8_t * hotmask = nullptr;
                if (n_hot > 0) {
                    std::vector<int32_t> hubo(n_hot), hbk(n_hot);
                    std::vector<int8_t>  hm(n_groups, 0);
                    std::vector<int32_t> hmeta((size_t) 4*n_hot);
                    int rows_pad = 0;
                    for (int g = 0, hs = 0, off = 0; g < n_groups; off += hcnt[g], ++g) {
                        if (hcnt[g] < gg_min) {
                            continue;
                        }
                        int b = 32;
                        while (b < hcnt[g]) {
                            b = (b & (b - 1)) ? b/3*4 : b*3/2;   // 32 48 64 96 128 ...
                        }
                        hm[g] = 1;
                        hmeta[hs]           = g;
                        hmeta[n_hot + hs]   = off;
                        hmeta[2*n_hot + hs] = rows_pad;
                        hmeta[3*n_hot + hs] = hcnt[g];
                        hubo[hs] = rows_pad;
                        hbk[hs]  = b;
                        rows_pad += b;
                        ++hs;
                    }
                    hmeta.resize((size_t) 4*n_hot + rows_pad);
                    for (int hs = 0; hs < n_hot; ++hs) {
                        for (int r = 0; r < hbk[hs]; ++r) {
                            hmeta[4*n_hot + hubo[hs] + r] = hs;
                        }
                    }
                    hotmask = hmask_alloc.alloc((size_t) n_groups);
                    int32_t * meta = meta_alloc.alloc(hmeta.size());
                    CUDA_CHECK(cudaMemcpyAsync(hotmask, hm.data(), (size_t) n_groups,
                                               cudaMemcpyHostToDevice, stream));
                    CUDA_CHECK(cudaMemcpyAsync(meta, hmeta.data(), hmeta.size()*sizeof(int32_t),
                                               cudaMemcpyHostToDevice, stream));
                    half    * gbank  = nullptr;
                    half    * ub     = nullptr;
                    float   * cb     = nullptr;
                    int8_t  * gbank8 = nullptr;
                    int8_t  * ub8    = nullptr;
                    int32_t * cb32   = nullptr;
                    float   * wsc    = nullptr;
                    float   * bsc    = nullptr;
                    if (gg_i8) {
                        gbank8 = gbank8_alloc.alloc((size_t) n_hot*m*n);
                        ub8    = ub8_alloc.alloc((size_t) rows_pad*n);
                        cb32   = cb32_alloc.alloc((size_t) rows_pad*m);
                        wsc    = wsc_alloc.alloc((size_t) n_hot*(m/16));
                        bsc    = bsc_alloc.alloc((size_t) rows_pad);
                    } else {
                        gbank = gbank_alloc.alloc((size_t) n_hot*m*n);
                        ub    = ub_alloc.alloc((size_t) rows_pad*n);
                        cb    = cb_alloc.alloc((size_t) rows_pad*m);
                    }
                    if (mach1_debug_on()) {
                        static int dumps = 0;
                        if (dumps++ < 8) {
                            fprintf(stderr, "mach1: exp_gg m=%d n=%d gg_min=%d n_hot=%d n_cold=%d rows_pad=%d P=%d\n",
                                    m, n, gg_min, n_hot, n_cold, rows_pad, P);
                        }
                    }
                    // cold groups first: the fused kernel is the op's biggest
                    // GPU slice, queueing it ahead hides the GEMM host dispatch;
                    // GG_FORK moves it to a side stream so its GPU time overlaps
                    // the whole hot chain, not just the host dispatch
                    mach1_gg_fork * ggf = nullptr;
                    if (n_cold > 0 && gg_fork != 0) {
                        ggf = mach1_gg_fork_get(ctx.device);
                    }
                    if (n_cold > 0) {
                        mach1_timed(stream, std::string("exp_gg_cold") + shp, [&]() {
                        ggml_cuda_kernel_launch_params zc = zp;
                        if (ggf != nullptr) {
                            CUDA_CHECK(cudaEventRecord(ggf->e0, stream));
                            CUDA_CHECK(cudaStreamWaitEvent(ggf->s2, ggf->e0, 0));
                            zc.stream = ggf->s2;
                        }
                        if (cold_s8) {
                            if (n == 512) {
                                mach1_launch(mach1_exp_zdp_mma_kernel<1>, zc,
                                    kept_d, p4t.zpacked, p4t.zstep,
                                    (const int32_t *) scr_i, (const uint32_t *) scr_q8, scr_v,
                                    (const half *) gamma->data, m, n, n_groups,
                                    (const int8_t *) hotmask, zprobe);
                            } else {
                                mach1_launch(mach1_exp_zdp_mma_kernel<0>, zc,
                                    kept_d, p4t.zpacked, p4t.zstep,
                                    (const int32_t *) scr_i, (const uint32_t *) scr_q8, scr_v,
                                    (const half *) gamma->data, m, n, n_groups,
                                    (const int8_t *) hotmask, zprobe);
                            }
                        } else {
                            mach1_launch(mach1_exp_zdp_fp16_kernel, zc,
                                kept_d, p4t.zpacked, p4t.zstep,
                                (const int32_t *) scr_i, (const half *) eu16, scr_v,
                                (const half *) gamma->data, m, n, n_groups,
                                (const int8_t *) hotmask, zprobe);
                        }
                        });
                    }
                    mach1_timed(stream, std::string("exp_gg_dec") + shp, [&]() {
                    if (gg_i8) {
                        mach1_launch(mach1_exp_gg_decode_i8_kernel,
                            ggml_cuda_kernel_launch_params(dim3(m/16, n_hot, 1), dim3(128, 1, 1), 0, stream),
                            kept_d, p4t.zpacked, p4t.zstep, (const int32_t *) scr_i,
                            (const int32_t *) meta, gbank8, wsc, (const half *) gamma->data, m, n, n_groups);
                        mach1_launch(mach1_exp_gg_gather_i8_kernel,
                            ggml_cuda_kernel_launch_params(dim3(rows_pad, 1, 1), dim3(256, 1, 1), 0, stream),
                            (const float *) scr_u, (const int32_t *) scr_i, (const int32_t *) meta,
                            ub8, bsc, n, n_groups, n_hot);
                    } else {
                        mach1_launch(mach1_exp_gg_decode_kernel,
                            ggml_cuda_kernel_launch_params(dim3(m/16, n_hot, 1), dim3(128, 1, 1), 0, stream),
                            kept_d, p4t.zpacked, p4t.zstep, (const int32_t *) scr_i,
                            (const int32_t *) meta, gbank, (const half *) gamma->data, m, n, n_groups);
                        mach1_launch(mach1_exp_gg_gather_kernel,
                            ggml_cuda_kernel_launch_params(dim3(rows_pad, 1, 1), dim3(256, 1, 1), 0, stream),
                            (const half *) eu16, (const int32_t *) scr_i, (const int32_t *) meta,
                            ub, n, n_groups, n_hot);
                    }
                    });
                    mach1_timed(stream, std::string("exp_gg_gemm") + shp, [&]() {
                    // slots sharing a pad bucket run as ONE cublasGemmBatchedEx
                    // (device pointer arrays) - ~8 dispatches instead of n_hot;
                    // the grouped-batched API rejects 16F inputs (status 15)
                    static const int gg_batched = mach1_env_int("GGML_MACH1_GG_BATCHED", 1);
                    if (gg_batched != 0 || gg_i8) {
                        std::vector<int> order(n_hot);
                        for (int i = 0; i < n_hot; ++i) {
                            order[i] = i;
                        }
                        std::stable_sort(order.begin(), order.end(),
                                         [&](int a, int b) { return hbk[a] < hbk[b]; });
                        std::vector<const void *> hpab((size_t) 2*n_hot);
                        std::vector<void *>       hpc(n_hot);
                        for (int i = 0; i < n_hot; ++i) {
                            const int hs = order[i];
                            if (gg_i8) {
                                hpab[i]         = gbank8 + (size_t) hs*m*n;
                                hpab[n_hot + i] = ub8    + (size_t) hubo[hs]*n;
                                hpc[i]          = cb32   + (size_t) hubo[hs]*m;
                            } else {
                                hpab[i]         = gbank + (size_t) hs*m*n;
                                hpab[n_hot + i] = ub    + (size_t) hubo[hs]*n;
                                hpc[i]          = cb    + (size_t) hubo[hs]*m;
                            }
                        }
                        ggml_cuda_pool_alloc<const void *> pab_alloc(ctx.pool(), (size_t) 2*n_hot);
                        ggml_cuda_pool_alloc<void *>       pc_alloc(ctx.pool(), (size_t) n_hot);
                        CUDA_CHECK(cudaMemcpyAsync(pab_alloc.get(), hpab.data(),
                                                   (size_t) 2*n_hot*sizeof(void *), cudaMemcpyHostToDevice, stream));
                        CUDA_CHECK(cudaMemcpyAsync(pc_alloc.get(), hpc.data(),
                                                   (size_t) n_hot*sizeof(void *), cudaMemcpyHostToDevice, stream));
                        static std::map<int64_t, cublasHandle_t> bmap;
                        static std::mutex bmu;
                        const float   onef = 1.0f, zerof = 0.0f;
                        const int32_t onei = 1,    zeroi = 0;
                        for (int i0 = 0; i0 < n_hot;) {
                            const int b = hbk[order[i0]];
                            int i1 = i0 + 1;
                            while (i1 < n_hot && hbk[order[i1]] == b) {
                                ++i1;
                            }
                            cublasHandle_t cbh;
                            {
                                std::lock_guard<std::mutex> lk(bmu);
                                auto & hh = bmap[((int64_t) m << 40) | ((int64_t) n << 20) |
                                                 ((int64_t) b << 1) | (gg_i8 ? 1 : 0)];
                                if (hh == nullptr) {
                                    CUBLAS_CHECK(cublasCreate(&hh));
                                    static void * ws = nullptr;
                                    static const size_t wssz = 32u*1024u*1024u;
                                    if (ws == nullptr) {
                                        CUDA_CHECK(cudaMalloc(&ws, wssz));
                                    }
                                    CUBLAS_CHECK(cublasSetWorkspace(hh, ws, wssz));
                                }
                                cbh = hh;
                            }
                            CUBLAS_CHECK(cublasSetStream(cbh, stream));
                            if (gg_i8) {
                                CUBLAS_CHECK(cublasGemmBatchedEx(cbh, CUBLAS_OP_T, CUBLAS_OP_N,
                                    m, b, n,
                                    &onei,  (const void **) (pab_alloc.get() + i0),         CUDA_R_8I,  n,
                                            (const void **) (pab_alloc.get() + n_hot + i0), CUDA_R_8I,  n,
                                    &zeroi, (      void **) (pc_alloc.get() + i0),          CUDA_R_32I, m,
                                    i1 - i0,
                                    CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
                            } else {
                                CUBLAS_CHECK(cublasGemmBatchedEx(cbh, CUBLAS_OP_T, CUBLAS_OP_N,
                                    m, b, n,
                                    &onef,  (const void **) (pab_alloc.get() + i0),         CUDA_R_16F, n,
                                            (const void **) (pab_alloc.get() + n_hot + i0), CUDA_R_16F, n,
                                    &zerof, (      void **) (pc_alloc.get() + i0),          CUDA_R_32F, m,
                                    i1 - i0,
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                            }
                            i0 = i1;
                        }
                    } else {
                    static std::map<int64_t, cublasHandle_t> hmap;
                    static std::mutex hmu;
                    const float onef = 1.0f, zerof = 0.0f;
                    for (int hs = 0; hs < n_hot; ++hs) {
                        cublasHandle_t cbh;
                        {
                            std::lock_guard<std::mutex> lk(hmu);
                            auto & hh = hmap[((int64_t) m << 40) | ((int64_t) n << 20) | hbk[hs]];
                            if (hh == nullptr) {
                                CUBLAS_CHECK(cublasCreate(&hh));
                                static void * ws = nullptr;
                                static const size_t wssz = 32u*1024u*1024u;
                                if (ws == nullptr) {
                                    CUDA_CHECK(cudaMalloc(&ws, wssz));
                                }
                                CUBLAS_CHECK(cublasSetWorkspace(hh, ws, wssz));
                            }
                            cbh = hh;
                        }
                        CUBLAS_CHECK(cublasSetStream(cbh, stream));
                        CUBLAS_CHECK(cublasGemmEx(cbh, CUBLAS_OP_T, CUBLAS_OP_N,
                            m, hbk[hs], n,
                            &onef,  gbank + (size_t) hs*m*n,     CUDA_R_16F, n,
                                    ub    + (size_t) hubo[hs]*n, CUDA_R_16F, n,
                            &zerof, cb    + (size_t) hubo[hs]*m, CUDA_R_32F, m,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                    }
                    }
                    });
                    mach1_timed(stream, std::string("exp_gg_scat") + shp, [&]() {
                    if (gg_i8) {
                        mach1_launch(mach1_exp_gg_scatter_i8_kernel,
                            ggml_cuda_kernel_launch_params(dim3(rows_pad, 1, 1), dim3(256, 1, 1), 0, stream),
                            (const int32_t *) cb32, (const int32_t *) scr_i, (const int32_t *) meta,
                            (const float *) wsc, (const float *) bsc, scr_v, m, n_groups, n_hot);
                    } else {
                        mach1_launch(mach1_exp_gg_scatter_kernel,
                            ggml_cuda_kernel_launch_params(dim3(rows_pad, 1, 1), dim3(256, 1, 1), 0, stream),
                            (const float *) cb, (const int32_t *) scr_i, (const int32_t *) meta,
                            scr_v, m, n_groups, n_hot);
                    }
                    });
                    if (ggf != nullptr) {
                        // join before the epilogue reads scr_v and before the
                        // pool's later reuse of eu16/scr_v on the main stream
                        CUDA_CHECK(cudaEventRecord(ggf->e1, ggf->s2));
                        CUDA_CHECK(cudaStreamWaitEvent(stream, ggf->e1, 0));
                    }
                }
                if (n_hot == 0) {
                    mach1_timed(stream, std::string("exp_gg_cold") + shp, [&]() {
                    if (cold_s8) {
                        if (n == 512) {
                            mach1_launch(mach1_exp_zdp_mma_kernel<1>, zp,
                                kept_d, p4t.zpacked, p4t.zstep,
                                (const int32_t *) scr_i, (const uint32_t *) scr_q8, scr_v,
                                (const half *) gamma->data, m, n, n_groups,
                                (const int8_t *) hotmask, zprobe);
                        } else {
                            mach1_launch(mach1_exp_zdp_mma_kernel<0>, zp,
                                kept_d, p4t.zpacked, p4t.zstep,
                                (const int32_t *) scr_i, (const uint32_t *) scr_q8, scr_v,
                                (const half *) gamma->data, m, n, n_groups,
                                (const int8_t *) hotmask, zprobe);
                        }
                    } else {
                        mach1_launch(mach1_exp_zdp_fp16_kernel, zp,
                            kept_d, p4t.zpacked, p4t.zstep,
                            (const int32_t *) scr_i, (const half *) eu16, scr_v,
                            (const half *) gamma->data, m, n, n_groups,
                            (const int8_t *) hotmask, zprobe);
                    }
                    });
                }
            });
        // PPLOW: a GG request that only its OWN floor gated out must not drag
        // the apply onto the fp16 lane - fp16 is moot inside the GG branch but
        // standalone it loses to s8 MMA on Ada (ada-s1: -14.3 percent). With
        // PPLOW off the two floors are equal, so this reads as the old zgg.
        } else if ((zf16 || (zgg && n_tok >= mach1_pp_gg_min())) && n_tok >= mach1_pp_min()) {
            mach1_pplow_mark("exp apply fp16", n_tok);
            if (mach1_debug_on()) {
                static std::once_flag once;
                std::call_once(once, []() {
                    fprintf(stderr, "mach1: PP expert zdp apply fp16 MMA ENGAGED (LUT expand, no records)\n");
                });
            }
            ggml_cuda_pool_alloc<half> eu16_alloc(ctx.pool(), (size_t) P*n);
            half * eu16 = eu16_alloc.get();
            static const int zprobe = mach1_env_int("GGML_MACH1_ZMMA_PROBE", 0);
            if (!mach1_ablate(MACH1_ABLATE_EXP_WALK))
            mach1_timed(stream, std::string("exp_zdp_f16") + shp, [&]() {
            const int64_t k = (int64_t) P*n;
            mach1_launch(mach1_rt_u16_kernel,
                ggml_cuda_kernel_launch_params(
                    dim3((unsigned) ((k + 255)/256), 1, 1), dim3(256, 1, 1), 0, stream),
                (const float *) scr_u, eu16, k);
            mach1_launch(mach1_exp_zdp_fp16_kernel, zp,
                kept_d, p4t.zpacked, p4t.zstep,
                (const int32_t *) scr_i, (const half *) eu16, scr_v,
                (const half *) gamma->data, m, n, n_groups,
                (const int8_t *) nullptr, zprobe);
            });
        } else {
        // PP lane: s8 MMA form of the apply (requires the QONCE records)
        static const bool zmma = mach1_env_int("GGML_MACH1_EXP_APPLY_MMA", 0) != 0;
        if (zmma && scr_q8 != nullptr && n_tok >= mach1_pp_min()) {
            mach1_pplow_mark("exp apply s8 MMA", n_tok);
            if (mach1_debug_on()) {
                static std::once_flag once;
                std::call_once(once, []() {
                    fprintf(stderr, "mach1: PP expert zdp apply s8 MMA ENGAGED (true-int8 tiles)\n");
                });
            }
            static const int zprobe = mach1_env_int("GGML_MACH1_ZMMA_PROBE", 0);
            if (!mach1_ablate(MACH1_ABLATE_EXP_WALK))
            mach1_timed(stream, std::string("exp_zdp_mma") + shp, [&]() {
            if (n == 512) {
                mach1_launch(mach1_exp_zdp_mma_kernel<1>, zp,
                    kept_d, p4t.zpacked, p4t.zstep,
                    (const int32_t *) scr_i, (const uint32_t *) scr_q8, scr_v,
                    (const half *) gamma->data, m, n, n_groups,
                    (const int8_t *) nullptr, zprobe);
            } else {
                mach1_launch(mach1_exp_zdp_mma_kernel<0>, zp,
                    kept_d, p4t.zpacked, p4t.zstep,
                    (const int32_t *) scr_i, (const uint32_t *) scr_q8, scr_v,
                    (const half *) gamma->data, m, n, n_groups,
                    (const int8_t *) nullptr, zprobe);
            }
            });
        } else {
        if (!mach1_ablate(MACH1_ABLATE_EXP_WALK))
        mach1_timed(stream, std::string("exp_zdp_apply") + shp, [&]() {
        static const int zlb = mach1_env_int("GGML_MACH1_EXP_ZLB", 1);
#define M1_ZAPPLY(PCV, QO, TPTV, LBV) \
        mach1_launch(mach1_exp_zdp_apply_kernel<PCV, QO, TPTV, LBV>, zp, \
            kept_d, p4t.zpacked, p4t.zstep, \
            (const int32_t *) scr_i, (const float *) scr_u, \
            QO ? (const uint32_t *) scr_q8 : (const uint32_t *) nullptr, scr_v, \
            (const half *) gamma->data, m, n, n_groups)
#define M1_ZAPPLY_PC(QO, TPTV) \
        do { \
            if      (zpc <= 2)             { M1_ZAPPLY(2, QO, TPTV, 1); } \
            else if (zpc <= 4 && zlb == 4) { M1_ZAPPLY(4, QO, TPTV, 4); } \
            else if (zpc <= 4 && zlb >= 6) { M1_ZAPPLY(4, QO, TPTV, 6); } \
            else if (zpc <= 4)             { M1_ZAPPLY(4, QO, TPTV, 1); } \
            else if (zlb == 4)             { M1_ZAPPLY(8, QO, TPTV, 4); } \
            else if (zlb >= 6)             { M1_ZAPPLY(8, QO, TPTV, 6); } \
            else                           { M1_ZAPPLY(8, QO, TPTV, 1); } \
        } while (0)
        if (n == 512) {
            // 32 tiles against 128 threads: TPT = 4 (tile, row-group) map
            if (scr_q8 != nullptr) { M1_ZAPPLY_PC(1, 4); } else { M1_ZAPPLY_PC(0, 4); }
        } else {
            if (scr_q8 != nullptr) { M1_ZAPPLY_PC(1, 1); } else { M1_ZAPPLY_PC(0, 1); }
        }
#undef M1_ZAPPLY_PC
#undef M1_ZAPPLY
        });
        }
        }
    } else if (P >= dense_min && fused_dense && v8 && n == 512) {
        const ggml_cuda_kernel_launch_params fp =
            ggml_cuda_kernel_launch_params(dim3(m/16, 1, n_groups), dim3(128, 1, 1), 0, stream);
        if (!mach1_ablate(MACH1_ABLATE_EXP_WALK))
        mach1_timed(stream, std::string("exp_fused_apply") + shp, [&]() {
        if (p4) {
            mach1_launch(mach1_exp_fused_apply_kernel<true, 8>, fp,
                kept_d, (const void *) tlut->data, p4t.packed, p4t.levels,
                (const int32_t *) scr_i, (const float *) scr_u, scr_v,
                (const half *) gamma->data, m, n, n_groups, 1);
        } else {
            mach1_launch(mach1_exp_fused_apply_kernel<false, 8>, fp,
                kept_d, (const void *) tlut->data,
                (const uint32_t *) nullptr, (const float *) nullptr,
                (const int32_t *) scr_i, (const float *) scr_u, scr_v,
                (const half *) gamma->data, m, n, n_groups, 1);
        }
        });
    } else if (P >= dense_min) {
        // dense prefill: decode active groups once, then apply
        half * bank = bank_alloc.alloc((size_t) n_groups*m*n);
        const ggml_cuda_kernel_launch_params dec_params =
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, n_groups),
                                           dim3(256, 1, 1), 0, stream);
        mach1_timed(stream, std::string("exp_dense_decode") + shp, [&]() {
        if (v8) {
            if (p4) {
                mach1_launch(mach1_exp_dense_decode_kernel<true, true>, dec_params,
                    kept_d, dem_d, (const void *) tlut->data, p4t.packed, p4t.levels,
                    (const int32_t *) scr_i, bank, m, n, n_kept, n_groups);
            } else {
                mach1_launch(mach1_exp_dense_decode_kernel<true, false>, dec_params,
                    kept_d, dem_d, (const void *) tlut->data,
                    (const uint32_t *) nullptr, (const float *) nullptr,
                    (const int32_t *) scr_i, bank, m, n, n_kept, n_groups);
            }
        } else {
            mach1_launch(mach1_exp_dense_decode_kernel<false, false>, dec_params,
                kept_d, dem_d, (const void *) tlut->data,
                (const uint32_t *) nullptr, (const float *) nullptr,
                (const int32_t *) scr_i, bank, m, n, n_kept, n_groups);
        }
        });
        const ggml_cuda_kernel_launch_params app_params =
            ggml_cuda_kernel_launch_params(dim3(m/16, n_groups, 1), dim3(128, 1, 1), 0, stream);
        static const int exp_pc = mach1_env_int("GGML_MACH1_EXP_PC", 8);
        if (!mach1_ablate(MACH1_ABLATE_EXP_WALK))
    mach1_timed(stream, std::string("exp_apply") + shp, [&]() {
        if (v8) {
            if (exp_pc == 1) {
                mach1_launch(mach1_exp_apply_kernel<true, 1>, app_params,
                    (const half *) bank, (const int32_t *) scr_i, (const float *) scr_u, scr_v,
                    (const half *) gamma->data, m, n, n_groups);
            } else if (exp_pc == 4) {
                mach1_launch(mach1_exp_apply_kernel<true, 4>, app_params,
                    (const half *) bank, (const int32_t *) scr_i, (const float *) scr_u, scr_v,
                    (const half *) gamma->data, m, n, n_groups);
            } else {
                mach1_launch(mach1_exp_apply_kernel<true, 8>, app_params,
                    (const half *) bank, (const int32_t *) scr_i, (const float *) scr_u, scr_v,
                    (const half *) gamma->data, m, n, n_groups);
            }
        } else {
            if (exp_pc == 1) {
                mach1_launch(mach1_exp_apply_kernel<false, 1>, app_params,
                    (const half *) bank, (const int32_t *) scr_i, (const float *) scr_u, scr_v,
                    (const half *) nullptr, m, n, n_groups);
            } else if (exp_pc == 4) {
                mach1_launch(mach1_exp_apply_kernel<false, 4>, app_params,
                    (const half *) bank, (const int32_t *) scr_i, (const float *) scr_u, scr_v,
                    (const half *) nullptr, m, n, n_groups);
            } else {
                mach1_launch(mach1_exp_apply_kernel<false, 8>, app_params,
                    (const half *) bank, (const int32_t *) scr_i, (const float *) scr_u, scr_v,
                    (const half *) nullptr, m, n, n_groups);
            }
        }
        });
    } else {
        const ggml_cuda_kernel_launch_params walk_params =
            ggml_cuda_kernel_launch_params(dim3(m/16, 1, n_groups), dim3(128, 1, 1), 0, stream);
        // narrow inputs: warp-per-row-tile variant keeps whole blocks busy
        const bool walk_warp = v8 && n/16 <= 32 && m % 64 == 0;
        const ggml_cuda_kernel_launch_params warp_params =
            ggml_cuda_kernel_launch_params(dim3(m/64, 1, n_groups), dim3(128, 1, 1), 0, stream);
        if (!mach1_ablate(MACH1_ABLATE_EXP_WALK))
    mach1_timed(stream, std::string("exp_walk") + shp, [&]() {
        if (walk_warp) {
            if (p4) {
                mach1_launch(mach1_exp_walk_v8_warp_kernel<true>, warp_params,
                    kept_d, (const void *) tlut->data, p4t.packed, p4t.levels,
                    (const int32_t *) scr_i, scr_u, scr_v,
                    (const half *) gamma->data, m, n, n_groups);
            } else {
                mach1_launch(mach1_exp_walk_v8_warp_kernel<false>, warp_params,
                    kept_d, (const void *) tlut->data,
                    (const uint32_t *) nullptr, (const float *) nullptr,
                    (const int32_t *) scr_i, scr_u, scr_v,
                    (const half *) gamma->data, m, n, n_groups);
            }
        } else if (v8 && mach1_env_int("GGML_MACH1_WALK_RG", 4) == 4 && n/16 <= 128) {
            const ggml_cuda_kernel_launch_params rows_params =
                ggml_cuda_kernel_launch_params(dim3(m/16, 1, n_groups), dim3(512, 1, 1), 0, stream);
            if (p4) {
                mach1_launch(mach1_exp_walk_rows_kernel<true, 512, 4>, rows_params,
                    kept_d, (const void *) tlut->data, p4t.packed, p4t.levels,
                    (const int32_t *) scr_i, scr_u, scr_v,
                    (const half *) gamma->data, m, n, n_groups);
            } else {
                mach1_launch(mach1_exp_walk_rows_kernel<false, 512, 4>, rows_params,
                    kept_d, (const void *) tlut->data,
                    (const uint32_t *) nullptr, (const float *) nullptr,
                    (const int32_t *) scr_i, scr_u, scr_v,
                    (const half *) gamma->data, m, n, n_groups);
            }
        } else if (v8) {
            if (p4) {
                mach1_launch(mach1_exp_walk_kernel<true, true>, walk_params,
                    kept_d, dem_d, (const void *) tlut->data, p4t.packed, p4t.levels,
                    (const int32_t *) scr_i, scr_u, scr_v,
                    (const half *) gamma->data, m, n, n_kept, n_groups);
            } else {
                mach1_launch(mach1_exp_walk_kernel<true, false>, walk_params,
                    kept_d, dem_d, (const void *) tlut->data,
                    (const uint32_t *) nullptr, (const float *) nullptr,
                    (const int32_t *) scr_i, scr_u, scr_v,
                    (const half *) gamma->data, m, n, n_kept, n_groups);
            }
        } else {
            mach1_launch(mach1_exp_walk_kernel<false, false>, walk_params,
                kept_d, dem_d, (const void *) tlut->data,
                (const uint32_t *) nullptr, (const float *) nullptr,
                (const int32_t *) scr_i, scr_u, scr_v,
                (const half *) nullptr, m, n, n_kept, n_groups);
        }
        });
    }

    if (!mach1_ablate(MACH1_ABLATE_EXP_OUT))
    mach1_timed(stream, std::string("exp_out") + shp, [&]() {
    if (mach1_fwht_wg512()) {
        mach1_launch(mach1_exp_out_kernel<512>,
            ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(512, 1, 1), 0, stream),
            (const half *) sv->data, (const int32_t *) ids->data, scr_v, (float *) dst->data,
            m, n_used, ids_s0, ids_s1);
    } else {
        mach1_launch(mach1_exp_out_kernel<256>,
            ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(256, 1, 1), 0, stream),
            (const half *) sv->data, (const int32_t *) ids->data, scr_v, (float *) dst->data,
            m, n_used, ids_s0, ids_s1);
    }
    });
}

// Whole-layer expert FFN fusion (GGML_MACH1_EXP_FFN=1): matcher + executor
// for the decode-time expert subgraph emitted by src/models/mach1.cpp,
//   [i]   MACH1_EXP_MM gate            (this node)
//   [i+1] MACH1_EXP_MM up              (same x, ids, remap, tlut)
//   [i+2] GLU swiglu_split(gate, up)
//   [i+3] MACH1_EXP_MM down            (x = the GLU dst)
//   [i+4] MUL (down, router weights [1, n_used, 1])
//   [i+5 ..]           n_used VIEW     (slot columns of the MUL dst)
//   [i+5+n_used ..]    n_used-1 ADD    (the slot-sum chain)
// executed as the two launches above. Matched one node at a time; any
// difference (basis nodes, the swiglu_limit clamp variant, batched decode)
// falls back to the per-op path. Returns nodes to skip or 0.
// MEASUREMENT ONLY: which guard rejected the last expert-FFN fuse attempt.
// The matcher below has a dozen structurally distinct "give up" paths and a
// per-op census cannot tell them apart - a fused region is skipped before it
// is ever timed, so a non-engaging matcher looks exactly like an absent one.
// Round 133 found all three EXP_MM nodes per layer being dispatched
// individually in the H200 decode census even though GGML_MACH1_EXP_MEGA
// defaults to 1, which means one of these guards is firing on this model and
// costing ~80 launches per token on the dominant tier.  Recording __LINE__
// names the exact guard; graphstat (GGML_MACH1_TIME >= 4) reports it.
static int g_m1_ffn_reject_line = 0;

int ggml_cuda_mach1_exp_ffn_reject_line() { return g_m1_ffn_reject_line; }

#define M1_FFN_REJECT() do { g_m1_ffn_reject_line = __LINE__; return 0; } while (0)

// Cross-region fork (GGML_MACH1_FORK=1): the shexp gate|up chain reads only
// the layer input and weights, so it can run CONCURRENTLY with the routed
// expert region on a second stream - a capturable fork-join. The expert
// matcher records ex on the main stream before its launch; the shexp
// matcher runs gu on s2 behind ex, records egu, and joins main before the
// down launch (which legitimately reads the experts' output). The gu
// scratch is a private allocation, not pool memory, so a concurrent
// main-stream op can never alias it.
struct mach1_fork_state {
    cudaStream_t s2    = nullptr;
    cudaEvent_t  ex    = nullptr;
    cudaEvent_t  egu   = nullptr;
    cudaEvent_t  edn   = nullptr;   // FORK_DN: down walk done on s2
    cudaEvent_t  ezx   = nullptr;   // FORK_Z: layer input ready on main
    cudaEvent_t  ez    = nullptr;   // FORK_Z: z walk done on s2
    float      * scr   = nullptr;
    size_t       scrn  = 0;
    float      * zscr  = nullptr;   // FORK_Z: private scr_v for the z walk
    size_t       zscrn = 0;
    int        * zcnt  = nullptr;   // FORK_Z: private fold counter (main's bank races)
    bool         armed = false;
    bool         z_armed = false;   // FORK_Z: ez pending, joined by the gdn_full exec
};

static bool mach1_fork_on() {
    static const bool on = mach1_env_int("GGML_MACH1_FORK", 1) != 0;
    return on;
}

// GGML_MACH1_FORK_NT: max decode tokens the mega/shexp fork covers. The
// machinery is nt-ready as-is: the s2 gu at nt > 1 counts on the ffn bank
// tail (+392..395, disjoint from the mega's slots [0..388] running
// concurrently on main), the egu join stream-orders gu before down exactly
// like the single-stream path, and the private scratch grows on request.
// Clamped to 4 - the shexp counter tail has 4 slots.
static int64_t mach1_fork_nt() {
    static const int64_t cap = std::max<int64_t>(1, std::min<int64_t>(4,
        mach1_env_int("GGML_MACH1_FORK_NT", 1)));
    return cap;
}

static mach1_fork_state * mach1_fork_get(int device, cudaStream_t stream, size_t scr_floats) {
    static mach1_fork_state st[GGML_CUDA_MAX_DEVICES];
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    mach1_fork_state & f = st[device];
    if (f.s2 == nullptr) {
        // resources must be created outside capture, like the counter banks
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) {
            cudaGetLastError();
            return nullptr;
        }
        if (cudaStreamCreateWithFlags(&f.s2, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&f.ex,  cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&f.egu, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&f.edn, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&f.ezx, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&f.ez,  cudaEventDisableTiming) != cudaSuccess) {
            cudaGetLastError();
            f.s2 = nullptr;
            return nullptr;
        }
    }
    if (f.scrn < scr_floats) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) {
            cudaGetLastError();
            return nullptr;
        }
        if (f.scr != nullptr) {
            CUDA_CHECK(cudaFree(f.scr));
            f.scr = nullptr;
        }
        if (cudaMalloc(&f.scr, scr_floats*sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            f.scrn = 0;
            return nullptr;
        }
        f.scrn = scr_floats;
    }
    return &f;
}

// FORK_Z: private scratch + fold counter for the z walk on s2. The main
// stream's counter bank races with a concurrent s2 walk, so z gets its own.
static bool mach1_fork_zres(mach1_fork_state * f, cudaStream_t stream, size_t floats) {
    if (f->zcnt != nullptr && f->zscrn >= floats) {
        return true;
    }
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) {
        cudaGetLastError();
        return false;
    }
    if (f->zcnt == nullptr) {
        if (cudaMalloc(&f->zcnt, sizeof(int)) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        CUDA_CHECK(cudaMemsetAsync(f->zcnt, 0, sizeof(int), stream));
    }
    if (f->zscrn < floats) {
        if (f->zscr != nullptr) {
            CUDA_CHECK(cudaFree(f->zscr));
            f->zscr = nullptr;
        }
        if (cudaMalloc(&f->zscr, floats*sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            f->zscrn = 0;
            return false;
        }
        f->zscrn = floats;
    }
    return true;
}

static bool mach1_fork_z_on() {
    static const bool on = mach1_env_int("GGML_MACH1_FORK_Z", 0) != 0;
    return on;
}

// GGML_MACH1_FORK_Z_NT: max decode tokens for the batched-region z fork
// (ops[1] of the qkvz pair rides s2 under the conv/core chain; the legacy
// nt == 1 vtiled fork keeps its own FORK_Z gate). 0 = off.
static int64_t mach1_fork_z_nt() {
    static const int64_t cap = std::max<int64_t>(0, std::min<int64_t>(4,
        mach1_env_int("GGML_MACH1_FORK_Z_NT", 0)));
    return cap;
}

// GGML_MACH1_FORK_P16: run the p16 shared-expert gate/up region (u_tcb,
// two TT walks, out+glu) on s2 under the routed expert mega instead of
// after it. The region reads the layer input and its own weights only, and
// the join is recorded before the down rt node - the first consumer of its
// output - so this is schedule, not values.
static bool mach1_fork_p16_on() {
    // default ON (gf-wall-s3): +4.7 pct at B8 (1.185x vs q4) and +5.5 at
    // B16 (1.478x) same-draw, neutral B2/B4 - the gate/up region hides under
    // the expert mega on s2. Token-exact certified (gf-nt-s1 np 2/8/16).
    static const bool on = mach1_env_int("GGML_MACH1_FORK_P16", 1) != 0;
    return on;
}

// GGML_MACH1_FORK_STAMP: %globaltimer stamps around the mega (main) and the
// forked region (s2). The decode graph is replayed, so no host code runs
// during a step - the stamps are written to a device ring and dumped from
// the next uncaptured matcher call. Overlap is [gu_begin, gu_end] meeting
// [mega_begin, mega_end] on the same device clock.
#define MACH1_STAMP_SLOTS 256

static __global__ void mach1_stamp_kernel(uint64_t * p, const int slot, const int tag) {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    p[2*slot]     = t;
    p[2*slot + 1] = (uint64_t) tag;
}

struct mach1_stamp_state {
    uint64_t * dev   = nullptr;
    int        slot  = 0;
    int        n     = 0;
    int        dumps = 0;
};

static bool mach1_stamp_on() {
    static const bool on = mach1_env_int("GGML_MACH1_FORK_STAMP", 0) != 0;
    return on;
}

static mach1_stamp_state * mach1_stamp_get(cudaStream_t stream) {
    static mach1_stamp_state st;
    if (!mach1_stamp_on()) {
        return nullptr;
    }
    if (st.dev == nullptr) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) {
            cudaGetLastError();
            return nullptr;
        }
        if (cudaMalloc(&st.dev, 2*MACH1_STAMP_SLOTS*sizeof(uint64_t)) != cudaSuccess) {
            cudaGetLastError();
            return nullptr;
        }
        CUDA_CHECK(cudaMemset(st.dev, 0, 2*MACH1_STAMP_SLOTS*sizeof(uint64_t)));
    }
    return &st;
}

static void mach1_stamp(cudaStream_t stream, cudaStream_t any, const int tag) {
    mach1_stamp_state * st = mach1_stamp_get(any);
    if (st == nullptr) {
        return;
    }
    mach1_launch(mach1_stamp_kernel,
        ggml_cuda_kernel_launch_params(dim3(1, 1, 1), dim3(1, 1, 1), 0, stream),
        st->dev, st->slot, tag);
    st->slot = (st->slot + 1) % MACH1_STAMP_SLOTS;
    st->n++;
}

// dump from an uncaptured call: the ring then holds the last executed step
static void mach1_stamp_dump(cudaStream_t stream) {
    mach1_stamp_state * st = mach1_stamp_get(stream);
    if (st == nullptr || st->dumps >= 3 || st->n < 160) {
        return;
    }
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) {
        cudaGetLastError();
        return;
    }
    st->dumps++;
    st->n = 0;
    std::vector<uint64_t> h(2*MACH1_STAMP_SLOTS);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(h.data(), st->dev, h.size()*sizeof(uint64_t), cudaMemcpyDeviceToHost));
    // globaltimer is device-wide and monotonic, so sorting by time recovers
    // EXECUTION order across both streams; the k-th begin pairs with the
    // k-th end of the same tag.
    std::vector<std::pair<uint64_t, int>> ev;
    for (int i = 0; i < MACH1_STAMP_SLOTS; ++i) {
        if (h[2*i + 1] != 0) {
            ev.emplace_back(h[2*i], (int) h[2*i + 1]);
        }
    }
    std::sort(ev.begin(), ev.end());
    std::vector<uint64_t> b1, e1, b2, e2;
    for (const auto & e : ev) {
        switch (e.second) {
            case 1: b1.push_back(e.first); break;
            case 2: e1.push_back(e.first); break;
            case 3: b2.push_back(e.first); break;
            case 4: e2.push_back(e.first); break;
            default: break;
        }
    }
    const size_t nl = std::min(std::min(b1.size(), e1.size()), std::min(b2.size(), e2.size()));
    const uint64_t t0 = ev.empty() ? 0 : ev.front().first;
    long long msum = 0, gsum = 0, osum = 0;
    for (size_t i = 0; i < nl; ++i) {
        if (e1[i] <= b1[i] || e2[i] <= b2[i]) {
            continue;
        }
        const uint64_t lo = std::max(b1[i], b2[i]);
        const uint64_t hi = std::min(e1[i], e2[i]);
        const long long ov = hi > lo ? (long long) (hi - lo) : 0;
        msum += (long long) (e1[i] - b1[i]);
        gsum += (long long) (e2[i] - b2[i]);
        osum += ov;
        fprintf(stderr, "mach1-stamp: L%02d mega=[%lld,%lld] %lld ns  gu=[%lld,%lld] %lld ns  overlap=%lld ns\n",
                (int) i, (long long) (b1[i] - t0), (long long) (e1[i] - t0), (long long) (e1[i] - b1[i]),
                (long long) (b2[i] - t0), (long long) (e2[i] - t0), (long long) (e2[i] - b2[i]), ov);
    }
    fprintf(stderr, "mach1-stamp: SUMMARY layers=%d mega=%lld ns gu=%lld ns overlap=%lld ns (%.1f pct of gu)\n",
            (int) nl, msum, gsum, osum, gsum > 0 ? 100.0*(double) osum/(double) gsum : 0.0);
}

int ggml_cuda_mach1_exp_ffn_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    static const bool two   = mach1_env_int("GGML_MACH1_EXP_FFN", 0) != 0;
    static const bool mega  = mach1_env_int("GGML_MACH1_EXP_MEGA", 1) != 0;
    static const bool on    = (two || mega) && mach1_env_int("GGML_MACH1_ABLATE", 0) == 0;
    if (!on || mach1_fold_enabled()) {
        M1_FFN_REJECT();
    }
    mach1_stamp_dump(ctx.stream());

    const ggml_tensor * gate = cgraph->nodes[node_idx];
    const ggml_tensor * ids  = gate->src[6];
    const int n_used = (int) ids->ne[0];
    const int n_tok  = (int) ids->ne[1];
    if (mach1_env_int("GGML_MACH1_DEBUG", 0) >= 3 && n_tok > 1) {
        static bool once_ntprev = false;
        if (!once_ntprev) {
            once_ntprev = true;
            for (int k = std::max(0, node_idx - 18); k < std::min(cgraph->n_nodes, node_idx + 4 + 2*n_used); ++k) {
                const ggml_tensor * t = cgraph->nodes[k];
                fprintf(stderr, "mach1-ntprev: [%+d] %-12s %-28s ne=%lld,%lld,%lld nb=%zu,%zu,%zu src0=%s\n",
                        k - node_idx, ggml_op_name(t->op), t->name,
                        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
                        t->nb[0], t->nb[1], t->nb[2],
                        t->src[0] ? t->src[0]->name : "-");
            }
        }
    }
    if (mach1_env_int("GGML_MACH1_DEBUG", 0) >= 3 && n_tok == 1) {
        static bool once2 = false;
        if (!once2) {
            once2 = true;
            for (int k = std::max(0, node_idx - 18); k < std::min(cgraph->n_nodes, node_idx + 2); ++k) {
                const ggml_tensor * t = cgraph->nodes[k];
                fprintf(stderr, "mach1-ffnprev: [%+d] %-12s %-28s ne=%lld,%lld,%lld src0=%s\n",
                        k - node_idx, ggml_op_name(t->op), t->name,
                        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
                        t->src[0] ? t->src[0]->name : "-");
            }
        }
    }
    // MEGA_NT: admit small decode batches (speculative verify, parallel
    // sequences) into the megakernel as (token, slot) pairs. The certified
    // counter layout carries 128 pairs; GGML_MACH1_NT32=1 admits up to 256
    // pairs through the widened QONCE layout (PCAP=256) below.
    static const int mega_nt = std::max(1, std::min(32, mach1_env_int("GGML_MACH1_MEGA_NT", 16)));
    const int pair_cap = mach1_nt32_on() ? 256 : 128;
    if (n_tok < 1 || n_tok > mega_nt || n_used < 2 || n_used > 8 || n_used*n_tok > pair_cap) {
        M1_FFN_REJECT();
    }
    const int len = 4 + 2*n_used;
    if (node_idx + len > cgraph->n_nodes) {
        M1_FFN_REJECT();
    }

    const ggml_tensor * up   = cgraph->nodes[node_idx + 1];
    const ggml_tensor * glu  = cgraph->nodes[node_idx + 2];
    const ggml_tensor * down = cgraph->nodes[node_idx + 3];
    const ggml_tensor * mul  = cgraph->nodes[node_idx + 4];
    if (up->op != GGML_OP_MACH1_EXP_MM || glu->op != GGML_OP_GLU ||
        down->op != GGML_OP_MACH1_EXP_MM || mul->op != GGML_OP_MUL) {
        M1_FFN_REJECT();
    }
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || glu->src[0] != gate || glu->src[1] != up ||
        ggml_get_op_params_i32(glu, 1) != 0 || glu->type != GGML_TYPE_F32) {
        M1_FFN_REJECT();
    }

    // one x, one router, one codebook across the three ops; no demoted tier
    // (v8/p4 payloads have none) and no basis nodes in between
    const ggml_tensor * x     = gate->src[7];
    const ggml_tensor * remap = gate->src[5];
    const ggml_tensor * tlut  = gate->src[4];
    // MEASUREMENT ONLY (GGML_MACH1_EXPFFN_FORCE=1): admit a demoted tier the
    // executor does not implement, so the region fuses anyway and the WORTH
    // of whole-layer expert fusion can be priced before the demoted path is
    // built.  Safe to force in the structural sense - the executor reads
    // src[0], src[2], src[3] and src[8] only and never dereferences src[1],
    // so this omits the demoted correction rather than reading bad memory.
    // The src[8] guard below is NOT forceable: the executor does read it.
    // OUTPUT IS NUMERICALLY WRONG under this flag - timing arm only, never
    // run it with smoke.
    static const bool force_dem = mach1_env_int("GGML_MACH1_EXPFFN_FORCE", 0) != 0;
    if (!force_dem &&
        (gate->src[1] != nullptr || up->src[1] != nullptr || down->src[1] != nullptr)) {
        M1_FFN_REJECT();
    }
    if (up->src[7] != x || down->src[7] != glu || up->src[6] != ids || down->src[6] != ids ||
        up->src[5] != remap || down->src[5] != remap || up->src[4] != tlut || down->src[4] != tlut) {
        M1_FFN_REJECT();
    }
    if (tlut->ne[0] != 8 || tlut->ne[1] != 32768 ||
        gate->src[8] == nullptr || up->src[8] == nullptr || down->src[8] == nullptr) {
        M1_FFN_REJECT();
    }
    if (gate->src[0]->ne[0] != 24 || up->src[0]->ne[0] != 24 || down->src[0]->ne[0] != 24) {
        M1_FFN_REJECT();
    }
    for (const ggml_tensor * t : { gate, up, down }) {
        if (t->src[0]->type != GGML_TYPE_I16 || t->src[2]->type != GGML_TYPE_F16 ||
            t->src[3]->type != GGML_TYPE_F16 || t->src[8]->type != GGML_TYPE_F16 ||
            !ggml_is_contiguous(t->src[0]) || !ggml_is_contiguous(t->src[2]) ||
            !ggml_is_contiguous(t->src[3]) || !ggml_is_contiguous(t->src[8])) {
            M1_FFN_REJECT();
        }
    }
    if (ids->type != GGML_TYPE_I32 || remap->type != GGML_TYPE_I32 || !ggml_is_contiguous(remap)) {
        M1_FFN_REJECT();
    }
    if (x->ne[1] != 1 || x->ne[2] != n_tok || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x)) {
        M1_FFN_REJECT();
    }

    // gate/up share m and n; every dimension fits the 2048-float shared bufs
    const int n   = (int) gate->src[2]->ne[0];
    const int mff = (int) gate->src[3]->ne[0];
    const int md  = (int) down->src[3]->ne[0];
    if ((int) up->src[2]->ne[0] != n || (int) up->src[3]->ne[0] != mff ||
        (int) down->src[2]->ne[0] != mff) {
        M1_FFN_REJECT();
    }
    if (n > 2048 || mff > 2048 || md > 2048 || n % 16 != 0 || mff % 16 != 0 || md % 16 != 0) {
        M1_FFN_REJECT();
    }
    if (glu->ne[0] != mff || glu->ne[1] != n_used || glu->ne[2] != n_tok || !ggml_is_contiguous(glu)) {
        M1_FFN_REJECT();
    }

    const ggml_tensor * weights = mul->src[1];
    if (mul->src[0] != down || weights->type != GGML_TYPE_F32 ||
        weights->ne[0] != 1 || weights->ne[1] != n_used || weights->ne[2] != n_tok ||
        ggml_nelements(weights) != (int64_t) n_used*n_tok ||
        mul->ne[0] != md || mul->ne[1] != n_used || mul->ne[2] != n_tok) {
        M1_FFN_REJECT();
    }

    for (int s = 0; s < n_used; ++s) {
        const ggml_tensor * v = cgraph->nodes[node_idx + 5 + s];
        if (v->op != GGML_OP_VIEW || v->view_src != mul || v->ne[0] != md || v->ne[1] != n_tok ||
            v->nb[1] != mul->nb[2] ||
            (const char *) v->data != (const char *) mul->data + s*mul->nb[1]) {
            M1_FFN_REJECT();
        }
    }
    const ggml_tensor * acc = cgraph->nodes[node_idx + 5];
    for (int k = 0; k < n_used - 1; ++k) {
        const ggml_tensor * a = cgraph->nodes[node_idx + 5 + n_used + k];
        if (a->op != GGML_OP_ADD || a->src[0] != acc || a->src[1] != cgraph->nodes[node_idx + 6 + k] ||
            a->type != GGML_TYPE_F32) {
            M1_FFN_REJECT();
        }
        acc = a;
    }
    ggml_tensor * out = cgraph->nodes[node_idx + len - 1];
    if (out->ne[0] != md || out->ne[1] != n_tok || out->nb[0] != sizeof(float)) {
        M1_FFN_REJECT();
    }

    // no intermediate may be a graph output or have external consumers
    ggml_op ops[20];
    ops[0] = GGML_OP_MACH1_EXP_MM;
    ops[1] = GGML_OP_MACH1_EXP_MM;
    ops[2] = GGML_OP_GLU;
    ops[3] = GGML_OP_MACH1_EXP_MM;
    ops[4] = GGML_OP_MUL;
    for (int s = 0; s < n_used; ++s) {
        ops[5 + s] = GGML_OP_VIEW;
    }
    for (int k = 0; k < n_used - 1; ++k) {
        ops[5 + n_used + k] = GGML_OP_ADD;
    }
    const int out_idx = node_idx + len - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, len, ops, &out_idx, 1)) {
        M1_FFN_REJECT();
    }

    cudaStream_t stream = ctx.stream();
    int * cnt = mach1_ffn_counters(ctx.device, stream);
    if (cnt == nullptr) {
        M1_FFN_REJECT();
    }
    const mach1_p4_table p4t = mach1_exp_p4_table(tlut->data, stream);
    if (p4t.packed == nullptr) {
        M1_FFN_REJECT();
    }

    if (mach1_fork_on() &&
        (n_tok <= mach1_fork_nt() || (mach1_fork_p16_on() && mach1_nt_ok(n_tok)))) {
        mach1_fork_state * f = mach1_fork_get(ctx.device, stream, 0);
        if (f != nullptr && cudaEventRecord(f->ex, stream) == cudaSuccess) {
            f->armed = true;
        } else {
            cudaGetLastError();
        }
    }

    const int ids_s0 = (int)(ids->nb[0]/sizeof(int32_t));
    const int ids_s1 = (int)(ids->nb[1]/sizeof(int32_t));
    const int ws     = (int)(weights->nb[1]/sizeof(float));
    const int wst    = (int)(weights->nb[2]/sizeof(float));
    const int ds1    = (int)(out->nb[1]/sizeof(float));
    const int P      = n_used*n_tok;   // (token, slot) pairs

    static const bool mega_u1 = mach1_env_int("GGML_MACH1_MEGA_U1", 0) != 0;
    static const bool mega_slotrel = mach1_env_int("GGML_MACH1_MEGA_SLOTREL", 0) != 0;
    static const bool mega_gdf = mach1_env_int("GGML_MACH1_MEGA_GDF", 0) != 0;
    static const bool mega_qonce_env = mach1_env_int("GGML_MACH1_MEGA_QONCE", 0) != 0;
    ggml_cuda_pool_alloc<float> scr_alloc(ctx.pool(),
        (size_t) 3*P*mff + (size_t) 2*P*md +
        (mega_u1 ? (size_t) 2*P*n + (size_t) P*mff :
         (mega_gdf ? (size_t) P*mff : 0)));
    float * scr_gu = scr_alloc.get();             // [2, P, mff]
    float * hbuf   = scr_gu + (size_t) 2*P*mff;   // [P, mff]
    float * scr_d  = hbuf + (size_t) P*mff;       // [P, md]
    float * wbuf   = scr_d + (size_t) P*md;       // [P, md]
    float * ubuf   = mega_u1 ? wbuf + (size_t) P*md : nullptr;
    float * dbuf   = mega_u1 ? ubuf + (size_t) 2*P*n
                   : (mega_gdf ? wbuf + (size_t) P*md : nullptr);

    char shp[112];

    // one persistent cooperative launch for the whole region (see
    // mach1_exp_mega_kernel); falls back to the two launches below when the
    // device cannot carry it
    // the down phase covers tiles_y = mff/16 column tiles with 128 threads;
    // at mff <= 512 that leaves 3/4 of them idle, so give them row groups
    static const bool dnrows = mach1_env_int("GGML_MACH1_EXP_DNROWS", 0) != 0;
    // ZDP: the integer-lattice walk (dp4a against q8 activations). Requires the
    // ladder repack; numerics are tolerance-class like TC_FWHT, not bit-exact.
    // EXP_DP4A selects the same walk with 15-bit activations (two int8 planes,
    // 4 dp4a) - round 46's P7 arm, 86x tighter than the q8 form. It implies
    // the integer walk, and it is carried by the plain mega only, so it turns
    // the MEGA_HALF/MEGA_SPLIT/EXP_SLOT variants off rather than silently
    // demoting itself to q8.
    const int  zmode = p4t.zpacked != nullptr ? mach1_exp_zdp_mode() : 0;
    const bool zdp   = zmode != 0;
    const bool dn32 = dnrows && mff <= 512;
    // MEGA_HALF (ZDP only): WG 512 with the 8192-row staged table - 44 KB, so
    // two blocks share an SM and each phase splits its tile rows twice as
    // wide. The high table rows read L2; the words are identical, so the
    // stream is bit-identical to the full-table form.
    // 1 = WG 512 + half table (the widened form); the r241 hang bisection:
    // 2 = WG 512 alone (full table), 3 = half table alone (WG 1024),
    // 4 = the level-1 binary with the grid capped at nsm (residency check:
    //     if the coop occupancy promise of 2/SM is the hang, 4 must run)
    static const int mh_env = mach1_env_int("GGML_MACH1_MEGA_HALF", 0);
    const int mh = (mh_env > 0 && zmode == 1 && mff <= 512 &&
                    n <= MACH1_EXP_MEGA_MAXD && md <= MACH1_EXP_MEGA_MAXD) ? mh_env : 0;
    static const bool slot_on = mach1_env_int("GGML_MACH1_EXP_SLOT", 0) != 0;
    static const bool msplit_env = mach1_env_int("GGML_MACH1_MEGA_SPLIT", 0) != 0;
    // Bounded production path: exact p16 shape only. The three q8 images
    // alias future-phase slices (not extra scratch), and the global H release
    // is part of their lifetime proof, so exclude every alternative scheduler.
    // QONCE's explicit opt-in selects its audited q8 record even when the
    // newer default EXP_DP4A mode is 15-bit; with QONCE off, zmode == 2 keeps
    // the round-333 path unchanged.
    // nt32: GGML_MACH1_NT32 admits (16, 32] tokens into the SAME audited
    // QONCE record/scheduling with the widened PCAP=256 counter layout.
    // ntlo: [2, 16) tokens ride the certified PCAP=128 layout at runtime
    // P = n_used*n_tok <= 120 - the producer/counter structure is P-general,
    // and without it the sub-16 mega re-quantizes per tile task. The
    // nt == 16 admission and its instantiations are untouched.
    const bool qonce_nt = n_tok == 16 ||
                          (mach1_nt32_on() && n_tok > 16 && n_tok <= 32) ||
                          (mach1_ntlo_on() && n_tok >= 2 && n_tok < 16) ||
                          mach1_ntlow_ok(n_tok);
    const bool mega_qonce = mega_qonce_env && mega && zdp && !force_dem &&
                            qonce_nt && n_used == 8 && n == 2048 && mff == 512 && md == 2048 &&
                            !mega_u1 && !mega_gdf && mh_env == 0 &&
                            !slot_on;
    // QONCE split: the two plain PHASE launches (WG 512, half table, 2/SM)
    // for the QONCE class - the kernel boundary replaces the H spin. The
    // split instantiations carry the certified PCAP=128 layout, so the wide
    // (P > 128) admission stays on the cooperative kernel.
    const bool qsplit = mega_qonce && msplit_env && n_tok >= 2 && P <= 128;
    if (n_tok >= 2 && n_tok < 16 && mach1_debug_on()) {
        static std::once_flag once_lodiag;
        std::call_once(once_lodiag, [&]() {
            fprintf(stderr, "mach1-ntlo: exp_ffn reached n_tok=%d ntlo=%d qonce_nt=%d mega=%d zdp=%d qenv=%d\n",
                    n_tok, mach1_ntlo_on() ? 1 : 0, qonce_nt ? 1 : 0, mega ? 1 : 0,
                    zdp ? 1 : 0, mega_qonce_env ? 1 : 0);
        });
    }
    // pairs beyond the certified 128 exist only in the wide-QONCE layout
    const bool mega_wide = P > 128;
    if (mega_wide && !mega_qonce) {
        M1_FFN_REJECT();
    }
    // QONCE_FWSCALE: the producer folds the FWHT normalization into the last
    // butterfly store (bitwise records, one shared pass and one barrier fewer
    // per pack site)
    static const bool mega_qip_env = mach1_env_int("GGML_MACH1_QONCE_FWSCALE", 0) != 0;
    const bool mega_qip = mega_qonce && mega_qip_env && !mach1_rt_stdout_on() && !dn32 && !mega_wide && !qsplit;
    const bool mega_std = mega_qonce && mach1_rt_stdout_on() && !dn32 && !mega_wide && !qsplit;
    // MEGA_ORDER: expert-sorted pair visitation (plain QONCE, certified
    // 128-pair layout only). Values bitwise identical; order is the lever.
    static const bool mega_order_env = mach1_env_int("GGML_MACH1_MEGA_ORDER", 0) != 0;
    const bool mega_ord = mega_order_env && mega_qonce && !mega_wide && !mega_qip && !mega_std && !qsplit;
    // PFLAGS: producer barrier -> per-pair record flags (see the kernel note)
    static const bool mega_pf_env = mach1_env_int("GGML_MACH1_QONCE_PFLAGS", 0) != 0;
    const bool mega_pf = mega_pf_env && mega_qonce && !mega_wide && !mega_qip && !mega_std && !mega_ord && !qsplit;
    // rung 9: the cooperative QONCE mega with the z-table served from L2/L1
    // (TABW=0, carveout prefers L1). Same launch, same barrier structure,
    // word-identical values - the qs2-wall split A/B measured the L1 table
    // 12-16 percent faster than the smem table at equal occupancy.
    static const bool mega_qns_env = mach1_env_int("GGML_MACH1_MEGA_NOSTAGE", 0) != 0;
    const bool mega_qns = mega_qns_env && mega_qonce && !mega_wide && !mega_qip && !mega_std &&
                          !mega_ord && !mega_pf && !qsplit;
    const void * megafn =
          (mega_wide && dn32) ? (const void *) mach1_exp_mega_kernel<32, 1, MACH1_EXP_MEGA_WG, false, 1, 0, true, false, false, 256>
        : mega_wide  ? (const void *) mach1_exp_mega_kernel<128, 1, MACH1_EXP_MEGA_WG, false, 1, 0, true, false, false, 256>
        : mega_std   ? (const void *) mach1_exp_mega_kernel<128, 1, MACH1_EXP_MEGA_WG, false, 1, 0, true, false, true>
        : mega_qip   ? (const void *) mach1_exp_mega_kernel<128, 1, MACH1_EXP_MEGA_WG, false, 1, 0, true, true>
        : (mega_pf && dn32) ? (const void *) mach1_exp_mega_kernel<32, 1, MACH1_EXP_MEGA_WG, false, 1, 0, true, false, false, 128, false, true>
        : mega_pf ? (const void *) mach1_exp_mega_kernel<128, 1, MACH1_EXP_MEGA_WG, false, 1, 0, true, false, false, 128, false, true>
        : (mega_ord && dn32) ? (const void *) mach1_exp_mega_kernel<32, 1, MACH1_EXP_MEGA_WG, false, 1, 0, true, false, false, 128, true>
        : mega_ord ? (const void *) mach1_exp_mega_kernel<128, 1, MACH1_EXP_MEGA_WG, false, 1, 0, true, false, false, 128, true>
        : (mega_qns && dn32) ? (const void *) mach1_exp_mega_kernel<32, 1, MACH1_EXP_MEGA_WG, false, 1, 0, true, false, false, 128, false, false, true>
        : mega_qns ? (const void *) mach1_exp_mega_kernel<128, 1, MACH1_EXP_MEGA_WG, false, 1, 0, true, false, false, 128, false, false, true>
        : (mega_qonce && dn32) ? (const void *) mach1_exp_mega_kernel<32, 1, MACH1_EXP_MEGA_WG, false, 1, 0, true>
        : mega_qonce ? (const void *) mach1_exp_mega_kernel<128, 1, MACH1_EXP_MEGA_WG, false, 1, 0, true>
        : (mh == 1 || mh == 4) ? (dn32 ? (const void *) mach1_exp_mega_kernel<32,  1, 512, true>
                                       : (const void *) mach1_exp_mega_kernel<128, 1, 512, true>)
        : mh == 2 ? (dn32 ? (const void *) mach1_exp_mega_kernel<32,  1, 512, false>
                          : (const void *) mach1_exp_mega_kernel<128, 1, 512, false>)
        : mh == 3 ? (dn32 ? (const void *) mach1_exp_mega_kernel<32,  1, 1024, true, 1>
                          : (const void *) mach1_exp_mega_kernel<128, 1, 1024, true, 1>)
        : zmode == 2 ? (dn32 ? (const void *) mach1_exp_mega_kernel<32,  2>
                             : (const void *) mach1_exp_mega_kernel<128, 2>)
        : dn32 ? (zdp ? (const void *) mach1_exp_mega_kernel<32,  1>
                      : (const void *) mach1_exp_mega_kernel<32,  0>)
               : (zdp ? (const void *) mach1_exp_mega_kernel<128, 1>
                      : (const void *) mach1_exp_mega_kernel<128, 0>);
    const int    mega_wg   = (mh == 1 || mh == 2 || mh == 4) ? 512 : MACH1_EXP_MEGA_WG;
    const size_t mega_smem = mega_qns ? MACH1_EXP_MEGA_SMEM_NOSTAGE
                           : (mh == 1 || mh == 3 || mh == 4) ? MACH1_EXP_MEGA_SMEM_HALF : MACH1_EXP_MEGA_SMEM;
    const int    mega_slot = (mega_wide && dn32) ? 17
                           : mega_wide ? 16
                           : mega_std ? 14
                           : mega_qip ? 13
                           : (mega_pf && dn32) ? 21
                           : mega_pf ? 20
                           : (mega_ord && dn32) ? 19
                           : mega_ord ? 18
                           : (mega_qns && dn32) ? 23
                           : mega_qns ? 22
                           : (mega_qonce && dn32) ? 15
                           : mega_qonce ? 12
                           : mh > 0 ? (2 + 2*((mh == 4) ? 1 : mh) + (dn32 ? 1 : 0))
                           : zmode == 2 ? 10 + (dn32 ? 1 : 0)
                                        : (dn32 ? 2 : 0) + (zdp ? 1 : 0);
    // per-slot form: one plain 1024-thread block per routed (token, slot)
    // pair, the whole chain in shared - see the kernel comment. Grid-sync-free
    // and capture safe, so no cooperative probe is needed; only smpbo gates it.
    // GGML_MACH1_EXP_SLOT_NT (default 1) admits decode batches: at nt in
    // [2, slot_nt] the P = n_used*n_tok independent blocks replace the
    // P-flat cooperative mega (ncu: 165 us at both P=64 and P=128, eligible
    // warps 1.65/8, mio+barrier stalls dominant). The q8 z-walk form matches
    // the QONCE numeric class those widths already run.
    static const int slot_nt = std::max(1, std::min(8, mach1_env_int("GGML_MACH1_EXP_SLOT_NT", 1)));
    // mff <= 512: the slot kernel's sgu/suu/sh_h are fixed 512-entry smem slices
    const bool slot = slot_on && mega && mff <= 512 &&
                      (n_tok == 1 ? zmode != 2
                                  : (n_tok <= slot_nt && zdp && mach1_ntlo_on())) &&
                      ggml_cuda_info().devices[ctx.device].smpbo >= MACH1_EXP_SLOT_SMEM;
    // MEGA_SPLIT: the widened shape as TWO PLAIN launches cut at the H
    // boundary - the kernel boundary is the join, so no cooperative launch,
    // no residency promise, and 2 blocks/SM whenever the hardware agrees
    // (the r241 coop form hangs when the occupancy promise is not honored)
    // MEGA_SPLIT at nt >= 2 ONLY: the single mega won the nt = 1 race
    // (round 38 class); at batched decode the DN spin waves grow with P and
    // the kernel-boundary join gets another look. The pair decomposition is
    // the nt-mega's (runtime ntok/strides), so only the grids scale by P.
    const bool msplit = msplit_env && zmode == 1 && n_tok >= 2 && mff <= 512 &&
                        n <= MACH1_EXP_MEGA_MAXD && md <= MACH1_EXP_MEGA_MAXD;
    int nb = (mega && !slot && !msplit && !qsplit) ? mach1_exp_mega_grid(ctx.device, megafn, mega_slot, mega_wg, mega_smem) : 0;
    if (mh == 4 && nb > 0) {
        nb = std::min(nb, ggml_cuda_info().devices[ctx.device].nsm);
    }
    // GGML_MACH1_MEGA_RSV: hold blocks back from the cooperative grid at
    // decode. The mega runs at 1 block/SM with the whole register file, so
    // nothing else can co-reside while it owns every SM; the held-back
    // blocks are the room a concurrent stream needs. The task loop is
    // `for (t = bx; t < ntask; t += NB)` and every output row tile belongs
    // to exactly one task, so the grid size does not move any value.
    static const int mega_rsv = mach1_env_int("GGML_MACH1_MEGA_RSV", 0);
    if (mega_rsv > 0 && nb > 0 && n_tok <= 32) {
        const int nb0 = nb;
        nb = std::max(1, nb - mega_rsv);
        if (mach1_debug_on()) {
            static std::once_flag once;
            std::call_once(once, [&]() {
                fprintf(stderr, "mach1: expert mega grid RESERVED %d of %d blocks (nb=%d)\n",
                        nb0 - nb, nb0, nb);
            });
        }
    }
    snprintf(shp, sizeof(shp), " m=%d n=%d md=%d n_used=%d nt=%d dnct=%d zdp=%d slot=%d q1=%d", mff, n, md,
             n_used, n_tok, dn32 ? 32 : 128, mega_qonce ? 1 : zmode, slot ? 1 : 0,
             mega_qip ? 2 : (mega_qonce ? 1 : 0));
    if (mega_qip && nb > 0 && mach1_debug_on()) {
        static std::once_flag once;
        std::call_once(once, []() {
            fprintf(stderr, "mach1: QONCE producer FWHT scale fold ENGAGED (bitwise records, zero scratch delta)\n");
        });
    }
    if (mega_std && nb > 0 && mach1_debug_on()) {
        static std::once_flag once;
        std::call_once(once, []() {
            fprintf(stderr, "mach1: expert mega standard-output ENGAGED (TIMING_ONLY WRONG_BASIS NOT_QUALITY)\n");
        });
    }
    if (mega_wide && nb > 0 && mach1_debug_on()) {
        static std::once_flag once;
        std::call_once(once, [&]() {
            fprintf(stderr, "mach1: QONCE mega nt32 WIDE ENGAGED (nt=%d P=%d pcap=256)\n", n_tok, P);
        });
    }
    if (mega_qonce && n_tok < 16 && nb > 0 && mach1_debug_on()) {
        static std::once_flag once_lo;
        std::call_once(once_lo, [&]() {
            fprintf(stderr, "mach1: ntlo QONCE mega nt=%d P=%d ENGAGED\n", n_tok, P);
        });
    }
    if (mega_ord && nb > 0 && mach1_debug_on()) {
        static std::once_flag once_ord;
        std::call_once(once_ord, [&]() {
            fprintf(stderr, "mach1: QONCE mega pair-order ENGAGED (P=%d)\n", P);
        });
    }
    if (mega_pf && nb > 0 && mach1_debug_on()) {
        static std::once_flag once_pf;
        std::call_once(once_pf, [&]() {
            fprintf(stderr, "mach1: QONCE mega pflags ENGAGED (P=%d)\n", P);
        });
    }
    if (mega_qns && nb > 0) {
        static bool qns_attr = false;
        if (!qns_attr) {
            CUDA_CHECK(cudaFuncSetAttribute(mach1_exp_mega_kernel<32, 1, MACH1_EXP_MEGA_WG, false, 1, 0, true, false, false, 128, false, false, true>,
                cudaFuncAttributePreferredSharedMemoryCarveout, 0));
            CUDA_CHECK(cudaFuncSetAttribute(mach1_exp_mega_kernel<128, 1, MACH1_EXP_MEGA_WG, false, 1, 0, true, false, false, 128, false, false, true>,
                cudaFuncAttributePreferredSharedMemoryCarveout, 0));
            qns_attr = true;
        }
        if (mach1_debug_on()) {
            static std::once_flag once_qns;
            std::call_once(once_qns, [&]() {
                fprintf(stderr, "mach1: QONCE mega nostage coop ENGAGED (nt=%d P=%d tab=L1)\n", n_tok, P);
            });
        }
    }
    if (mega_qonce && mach1_ntlow_ok(n_tok) && nb > 0 && mach1_debug_on()) {
        static std::once_flag once;
        std::call_once(once, [&]() {
            fprintf(stderr, "mach1: QONCE mega NTLOW ENGAGED (nt=%d P=%d pcap=128)\n", n_tok, P);
        });
    }
    static const int mega_tw_env = mach1_env_int("GGML_MACH1_MEGA_TW", 1);
    if (mega_qonce && nb > 0 && mach1_debug_on() &&
        (mega_tw_env > 1 || dn32 || mega_slotrel)) {
        static std::once_flag once;
        std::call_once(once, [&]() {
            fprintf(stderr, "mach1: QONCE mega scheduling ENGAGED (tw=%d dnct=%d slotrel=%d)\n",
                    mega_tw_env, dn32 ? 32 : 128, mega_slotrel ? 1 : 0);
        });
    }
    if (n_tok > 1 && mach1_env_int("GGML_MACH1_DEBUG", 0) >= 2) {
        static bool once_nt = false;
        if (!once_nt) {
            once_nt = true;
            fprintf(stderr, "mach1-ntmega: nt=%d P=%d ids_s0=%d ids_s1=%d ws=%d wst=%d ds1=%d nb=%d "
                    "x=%p ids=%p w=%p dst=%p\n", n_tok, P, ids_s0, ids_s1, ws, wst, ds1, nb,
                    (const void *) x->data, (const void *) ids->data,
                    (const void *) weights->data, (const void *) out->data);
            // graphs-off debug lane only: pull the routed ids and weights the
            // kernel is about to read and print them - garbage here means the
            // region's inputs are wrong before the mega ever runs
            cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
            if (cudaStreamIsCapturing(stream, &cap) == cudaSuccess && cap == cudaStreamCaptureStatusNone) {
                const int span_i = (n_used - 1)*ids_s0 + (n_tok - 1)*ids_s1 + 1;
                const int span_w = (n_used - 1)*ws + (n_tok - 1)*wst + 1;
                std::vector<int32_t> hids(span_i);
                std::vector<float>   hw(span_w);
                CUDA_CHECK(cudaMemcpyAsync(hids.data(), ids->data, span_i*sizeof(int32_t),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(hw.data(), weights->data, span_w*sizeof(float),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                for (int tk = 0; tk < n_tok; ++tk) {
                    fprintf(stderr, "mach1-ntmega: tok %d ids:", tk);
                    for (int s = 0; s < n_used; ++s) {
                        fprintf(stderr, " %d", hids[s*ids_s0 + tk*ids_s1]);
                    }
                    fprintf(stderr, " w:");
                    for (int s = 0; s < n_used; ++s) {
                        fprintf(stderr, " %.4f", hw[s*ws + tk*wst]);
                    }
                    fprintf(stderr, "\n");
                }
            } else {
                cudaGetLastError();
            }
        }
    }
    if (slot || nb > 0 || (mega && (msplit || qsplit))) {
        mach1_exp_mega_args a = {};
        a.kept[0]   = (const uint16_t *) gate->src[0]->data;
        a.kept[1]   = (const uint16_t *) up->src[0]->data;
        a.kept[2]   = (const uint16_t *) down->src[0]->data;
        a.su[0]     = (const half *) gate->src[2]->data;
        a.su[1]     = (const half *) up->src[2]->data;
        a.su[2]     = (const half *) down->src[2]->data;
        a.sv[0]     = (const half *) gate->src[3]->data;
        a.sv[1]     = (const half *) up->src[3]->data;
        a.sv[2]     = (const half *) down->src[3]->data;
        a.gamma[0]  = (const half *) gate->src[8]->data;
        a.gamma[1]  = (const half *) up->src[8]->data;
        a.gamma[2]  = (const half *) down->src[8]->data;
        a.p4        = p4t.packed;
        a.p4lv      = p4t.levels;
        a.zp4       = p4t.zpacked;
        a.zs0       = p4t.zstep;
        a.remap     = (const int32_t *) remap->data;
        a.ids       = (const int32_t *) ids->data;
        a.x         = (const float *) x->data;
        a.weights   = (const float *) weights->data;
        a.scr_gu    = scr_gu;
        a.hbuf      = hbuf;
        a.scr_d     = scr_d;
        a.wbuf      = wbuf;
        a.ubuf      = ubuf;
        a.dbuf      = dbuf;
        a.dst       = (float *) out->data;
        a.cnt       = cnt;
        a.u1        = mega_u1 ? 1 : 0;
        a.slotrel   = mega_slotrel ? 1 : 0;
        a.gdf       = (mega_gdf && dbuf != nullptr) ? 1 : 0;
        a.probe     = mach1_env_int("GGML_MACH1_MEGA_PROBE", 0);
        a.n_kept[0] = (int) gate->src[0]->ne[2];
        a.n_kept[1] = (int) up->src[0]->ne[2];
        a.n_kept[2] = (int) down->src[0]->ne[2];
        a.mff       = mff;
        a.n         = n;
        a.md        = md;
        a.n_used    = n_used;
        a.ids_s0    = ids_s0;
        a.ids_s1    = ids_s1;
        a.ws        = ws;
        a.ntok      = n_tok;
        a.wst       = wst;
        a.ds1       = ds1;
        if (mega && (msplit || qsplit) && !slot) {
            // one task per block, all four vblocks of the 512-thread block active
            a.vb_gu  = 4;
            a.tpo_gu = (mff/16 + 3)/4;
            a.vb_dn  = 4;
            a.tpo_dn = (md/16 + 3)/4;
            const int grid_gu = 2*P*a.tpo_gu;
            const int grid_dn = P*a.tpo_dn;
            // qsplit runs the PFLAGS producer edge, not the CNT_UGU barrier:
            // a plain ntask-wide launch makes no residency promise, and the
            // barrier deadlocks once 2*P producer blocks exceed what fits
            // (P=128 is 256 producers vs ~142 resident at 44 KB/block). With
            // per-record flags task block b waits only on record b/tpo_gu <= b,
            // so every scheduled prefix of blocks is self-sufficient - progress
            // at any residency. Values are the certified PFLAGS records
            // (pf-nt-s1 token gate). The carveout hint asks for the 100 KB
            // shared config so two 44 KB blocks share an SM (the split's whole
            // point); it is a perf hint only, correctness needs no residency.
            // MEGA_NOSTAGE (qsplit only): 1 = no staged table at MINB 2,
            // 2 = the same at MINB 3 (regs capped harder for 3 blocks/SM).
            // The table serves from L2/L1 via the tile's compile-time global
            // path; carveout prefers L1 so the 64 KB table can live there.
            static const int qs_ns_env = mach1_env_int("GGML_MACH1_MEGA_NOSTAGE", 0);
            const int qsns = qsplit ? std::max(0, std::min(2, qs_ns_env)) : 0;
            const size_t qs_smem = qsns ? MACH1_EXP_MEGA_SMEM_NOSTAGE : MACH1_EXP_MEGA_SMEM_HALF;
            if (qsplit) {
                static bool qs_attr = false;
                if (!qs_attr) {
                    CUDA_CHECK(cudaFuncSetAttribute(mach1_exp_mega_kernel<32,  1, 512, true, 2, 1, true, false, false, 128, false, true>,
                        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
                    CUDA_CHECK(cudaFuncSetAttribute(mach1_exp_mega_kernel<128, 1, 512, true, 2, 1, true, false, false, 128, false, true>,
                        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
                    CUDA_CHECK(cudaFuncSetAttribute(mach1_exp_mega_kernel<32,  1, 512, true, 2, 2, true, false, false, 128, false, true>,
                        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
                    CUDA_CHECK(cudaFuncSetAttribute(mach1_exp_mega_kernel<128, 1, 512, true, 2, 2, true, false, false, 128, false, true>,
                        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
                    CUDA_CHECK(cudaFuncSetAttribute(mach1_exp_mega_kernel<32,  1, 512, true, 2, 1, true, false, false, 128, false, true, true>,
                        cudaFuncAttributePreferredSharedMemoryCarveout, 0));
                    CUDA_CHECK(cudaFuncSetAttribute(mach1_exp_mega_kernel<128, 1, 512, true, 2, 1, true, false, false, 128, false, true, true>,
                        cudaFuncAttributePreferredSharedMemoryCarveout, 0));
                    CUDA_CHECK(cudaFuncSetAttribute(mach1_exp_mega_kernel<32,  1, 512, true, 2, 2, true, false, false, 128, false, true, true>,
                        cudaFuncAttributePreferredSharedMemoryCarveout, 0));
                    CUDA_CHECK(cudaFuncSetAttribute(mach1_exp_mega_kernel<128, 1, 512, true, 2, 2, true, false, false, 128, false, true, true>,
                        cudaFuncAttributePreferredSharedMemoryCarveout, 0));
                    CUDA_CHECK(cudaFuncSetAttribute(mach1_exp_mega_kernel<32,  1, 512, true, 3, 1, true, false, false, 128, false, true, true>,
                        cudaFuncAttributePreferredSharedMemoryCarveout, 0));
                    CUDA_CHECK(cudaFuncSetAttribute(mach1_exp_mega_kernel<128, 1, 512, true, 3, 1, true, false, false, 128, false, true, true>,
                        cudaFuncAttributePreferredSharedMemoryCarveout, 0));
                    CUDA_CHECK(cudaFuncSetAttribute(mach1_exp_mega_kernel<32,  1, 512, true, 3, 2, true, false, false, 128, false, true, true>,
                        cudaFuncAttributePreferredSharedMemoryCarveout, 0));
                    CUDA_CHECK(cudaFuncSetAttribute(mach1_exp_mega_kernel<128, 1, 512, true, 3, 2, true, false, false, 128, false, true, true>,
                        cudaFuncAttributePreferredSharedMemoryCarveout, 0));
                    qs_attr = true;
                }
            }
            mach1_timed(stream, std::string("exp_mega_gu") + shp, [&]() {
            if (qsplit && qsns == 2 && dn32) {
                mach1_launch(mach1_exp_mega_kernel<32,  1, 512, true, 3, 1, true, false, false, 128, false, true, true>,
                    ggml_cuda_kernel_launch_params(dim3(grid_gu, 1, 1), dim3(512, 1, 1), qs_smem, stream), a);
            } else if (qsplit && qsns == 2) {
                mach1_launch(mach1_exp_mega_kernel<128, 1, 512, true, 3, 1, true, false, false, 128, false, true, true>,
                    ggml_cuda_kernel_launch_params(dim3(grid_gu, 1, 1), dim3(512, 1, 1), qs_smem, stream), a);
            } else if (qsplit && qsns == 1 && dn32) {
                mach1_launch(mach1_exp_mega_kernel<32,  1, 512, true, 2, 1, true, false, false, 128, false, true, true>,
                    ggml_cuda_kernel_launch_params(dim3(grid_gu, 1, 1), dim3(512, 1, 1), qs_smem, stream), a);
            } else if (qsplit && qsns == 1) {
                mach1_launch(mach1_exp_mega_kernel<128, 1, 512, true, 2, 1, true, false, false, 128, false, true, true>,
                    ggml_cuda_kernel_launch_params(dim3(grid_gu, 1, 1), dim3(512, 1, 1), qs_smem, stream), a);
            } else if (qsplit && dn32) {
                mach1_launch(mach1_exp_mega_kernel<32,  1, 512, true, 2, 1, true, false, false, 128, false, true>,
                    ggml_cuda_kernel_launch_params(dim3(grid_gu, 1, 1), dim3(512, 1, 1), MACH1_EXP_MEGA_SMEM_HALF, stream), a);
            } else if (qsplit) {
                mach1_launch(mach1_exp_mega_kernel<128, 1, 512, true, 2, 1, true, false, false, 128, false, true>,
                    ggml_cuda_kernel_launch_params(dim3(grid_gu, 1, 1), dim3(512, 1, 1), MACH1_EXP_MEGA_SMEM_HALF, stream), a);
            } else if (dn32) {
                mach1_launch(mach1_exp_mega_kernel<32,  1, 512, true, 2, 1>,
                    ggml_cuda_kernel_launch_params(dim3(grid_gu, 1, 1), dim3(512, 1, 1), MACH1_EXP_MEGA_SMEM_HALF, stream), a);
            } else {
                mach1_launch(mach1_exp_mega_kernel<128, 1, 512, true, 2, 1>,
                    ggml_cuda_kernel_launch_params(dim3(grid_gu, 1, 1), dim3(512, 1, 1), MACH1_EXP_MEGA_SMEM_HALF, stream), a);
            }
            });
            mach1_timed(stream, std::string("exp_mega_dn") + shp, [&]() {
            if (qsplit && qsns == 2 && dn32) {
                mach1_launch(mach1_exp_mega_kernel<32,  1, 512, true, 3, 2, true, false, false, 128, false, true, true>,
                    ggml_cuda_kernel_launch_params(dim3(grid_dn, 1, 1), dim3(512, 1, 1), qs_smem, stream), a);
            } else if (qsplit && qsns == 2) {
                mach1_launch(mach1_exp_mega_kernel<128, 1, 512, true, 3, 2, true, false, false, 128, false, true, true>,
                    ggml_cuda_kernel_launch_params(dim3(grid_dn, 1, 1), dim3(512, 1, 1), qs_smem, stream), a);
            } else if (qsplit && qsns == 1 && dn32) {
                mach1_launch(mach1_exp_mega_kernel<32,  1, 512, true, 2, 2, true, false, false, 128, false, true, true>,
                    ggml_cuda_kernel_launch_params(dim3(grid_dn, 1, 1), dim3(512, 1, 1), qs_smem, stream), a);
            } else if (qsplit && qsns == 1) {
                mach1_launch(mach1_exp_mega_kernel<128, 1, 512, true, 2, 2, true, false, false, 128, false, true, true>,
                    ggml_cuda_kernel_launch_params(dim3(grid_dn, 1, 1), dim3(512, 1, 1), qs_smem, stream), a);
            } else if (qsplit && dn32) {
                mach1_launch(mach1_exp_mega_kernel<32,  1, 512, true, 2, 2, true, false, false, 128, false, true>,
                    ggml_cuda_kernel_launch_params(dim3(grid_dn, 1, 1), dim3(512, 1, 1), MACH1_EXP_MEGA_SMEM_HALF, stream), a);
            } else if (qsplit) {
                mach1_launch(mach1_exp_mega_kernel<128, 1, 512, true, 2, 2, true, false, false, 128, false, true>,
                    ggml_cuda_kernel_launch_params(dim3(grid_dn, 1, 1), dim3(512, 1, 1), MACH1_EXP_MEGA_SMEM_HALF, stream), a);
            } else if (dn32) {
                mach1_launch(mach1_exp_mega_kernel<32,  1, 512, true, 2, 2>,
                    ggml_cuda_kernel_launch_params(dim3(grid_dn, 1, 1), dim3(512, 1, 1), MACH1_EXP_MEGA_SMEM_HALF, stream), a);
            } else {
                mach1_launch(mach1_exp_mega_kernel<128, 1, 512, true, 2, 2>,
                    ggml_cuda_kernel_launch_params(dim3(grid_dn, 1, 1), dim3(512, 1, 1), MACH1_EXP_MEGA_SMEM_HALF, stream), a);
            }
            });
            if (qsplit && mach1_debug_on()) {
                static std::once_flag once_qs;
                std::call_once(once_qs, [&]() {
                    fprintf(stderr, "mach1: QONCE mega split ENGAGED (nt=%d P=%d wg512 %s %d/SM pflags)\n",
                            n_tok, P, qsns ? "nostage" : "half", qsns == 2 ? 3 : 2);
                });
            }
            return len - 1;
        }
        // MEGA_TW widens each task to tw row chunks so record staging,
        // fences, and counter traffic amortize; the task loops walk the extra
        // chunks with the same per-vblock tiles (see the kernel comment)
        static const int mega_tw = mach1_env_int("GGML_MACH1_MEGA_TW", 1);
        const int tw = (mega_qonce && !slot && mega_tw > 1) ? mega_tw : 1;
        mach1_exp_mega_split(slot ? n_used : nb, 2*P, mff/16, a.vb_gu, a.tpo_gu,
                             slot ? MACH1_EXP_MEGA_VB : tw*(mega_wg/128));
        mach1_exp_mega_split(slot ? n_used : nb, P,   md/16,  a.vb_dn, a.tpo_dn,
                             slot ? MACH1_EXP_MEGA_VB : tw*(mega_wg/128));

        if (slot) {
            static bool slot_attr = false;
            if (!slot_attr) {
                CUDA_CHECK(cudaFuncSetAttribute(mach1_exp_slot_kernel<128, false>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int) MACH1_EXP_SLOT_SMEM));
                CUDA_CHECK(cudaFuncSetAttribute(mach1_exp_slot_kernel<128, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int) MACH1_EXP_SLOT_SMEM));
                slot_attr = true;
            }
            const ggml_cuda_kernel_launch_params sp =
                ggml_cuda_kernel_launch_params(dim3(P, 1, 1), dim3(1024, 1, 1),
                                               MACH1_EXP_SLOT_SMEM, stream);
            mach1_timed(stream, std::string("exp_slot") + shp, [&]() {
            if (zdp) {
                mach1_launch(mach1_exp_slot_kernel<128, true>, sp, a);
            } else {
                mach1_launch(mach1_exp_slot_kernel<128, false>, sp, a);
            }
            });
            if (n_tok > 1 && mach1_debug_on()) {
                static std::once_flag once_snt;
                std::call_once(once_snt, [&]() {
                    fprintf(stderr, "mach1: exp_slot nt=%d P=%d ENGAGED\n", n_tok, P);
                });
            }
            return len - 1;
        }
        void * kargs[] = { (void *) &a };
        const bool stamp = n_tok <= 32;   // decode only: prefill would rotate the ring
        if (stamp) {
            mach1_stamp(stream, stream, 1);
        }
        mach1_timed(stream, std::string("exp_mega") + shp, [&]() {
        CUDA_CHECK(cudaLaunchCooperativeKernel(megafn,
            dim3(nb, 1, 1), dim3(mega_wg, 1, 1), kargs, mega_smem, stream));
        });
        if (stamp) {
            mach1_stamp(stream, stream, 2);
        }
        return len - 1;
    }
    // the two-launch fallback keeps its original per-slot grids - nt == 1 only
    if (!two || n_tok != 1) {
        M1_FFN_REJECT();
    }

    mach1_timed(stream, std::string("exp_ffn_gu") + shp, [&]() {
    mach1_launch(mach1_exp_ffn_gu_kernel,
        ggml_cuda_kernel_launch_params(dim3(mff/16, 2, n_used), dim3(128, 1, 1), 0, stream),
        (const uint16_t *) gate->src[0]->data, (const uint16_t *) up->src[0]->data,
        p4t.packed, p4t.levels,
        (const half *) gate->src[2]->data, (const half *) up->src[2]->data,
        (const half *) gate->src[3]->data, (const half *) up->src[3]->data,
        (const half *) gate->src[8]->data, (const half *) up->src[8]->data,
        (const int32_t *) remap->data, (const int32_t *) ids->data,
        (const float *) x->data, scr_gu, hbuf, cnt,
        mff, n, n_used, (int) gate->src[0]->ne[2], (int) up->src[0]->ne[2], ids_s0, ids_s1);
    });
    mach1_timed(stream, std::string("exp_ffn_down") + shp, [&]() {
    mach1_launch(mach1_exp_ffn_down_kernel,
        ggml_cuda_kernel_launch_params(dim3(md/16, 1, n_used), dim3(128, 1, 1), 0, stream),
        (const uint16_t *) down->src[0]->data, p4t.packed, p4t.levels,
        (const half *) down->src[2]->data, (const half *) down->src[3]->data,
        (const half *) down->src[8]->data,
        (const int32_t *) remap->data, (const int32_t *) ids->data,
        (const float *) hbuf, (const float *) weights->data,
        scr_d, wbuf, (float *) out->data, cnt, cnt + 8,
        md, mff, n_used, (int) down->src[0]->ne[2], ids_s0, ids_s1, ws);
    });

    return len - 1;
}

#undef M1_FFN_REJECT

// Shared-expert branch fusion (GGML_MACH1_SHEXP_FFN=1; decode only): the
// per-layer chain [rt gate, silu, rt up, mul, rt down, sigmoid-gate mul,
// moe add] runs as TWO launches - the matcher is ggml_cuda_mach1_shexp_fuse
// below. The tiny gate mul_mat and its sigmoid stay ordinary nodes that the
// builder (src/models/mach1.cpp) expands ahead of the region; launch F reads
// the computed sigmoid dst.
//
// Launch E "shexp gate|up": grid (m/16, 2, 1), blockIdx.y picks the
// projection. The block body is EXACTLY mach1_rt_walk_rows_v_kernel<WG, 4, 1>
// with per-projection trellis/su - both projections share x, so the UFUSE x'
// prologues differ only in su. The last of the 2*(m/16) blocks (completion
// counter, the OFUSE threadfence pattern) runs BOTH rt_out stages
// (sv .* H_m(v)/sqrt(m)) and the MUL node's silu(y_g)*y_u
// (ggml_cuda_op_silu_single, the UNARY_SILU inline) and writes h - launch
// F's x.
template <int WG, int ZDP = 0>
__global__ void __launch_bounds__(WG) mach1_shexp_gu_kernel(
        const uint16_t * __restrict__ tre_g,
        const uint16_t * __restrict__ tre_u,
        const half     * __restrict__ tlut,     // pre-rounded fp16
        const float    * __restrict__ su_g,
        const float    * __restrict__ su_u,
        const float    * __restrict__ sv_g,
        const float    * __restrict__ sv_u,
        const float    * __restrict__ x,
        float          * __restrict__ scr_v,    // [2, m]
        float          * __restrict__ h,        // [m]
        int            * __restrict__ counter,
        const int m, const int n,
        const uint16_t * __restrict__ ztab,     // ZDP = 1
        const float zs0) {                      // ZDP = 1
    constexpr int RG        = 4;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int CT        = WG / RG;
    constexpr int n_warps   = CT / warp_size;
    constexpr int ROWS      = 16 / RG;
    __shared__ float red[RG][16/RG][CT/32];
    __shared__ float slutA[ZDP ? 1 : 512];
    __shared__ float slutB[ZDP ? 1 : 512];
    __shared__ uint32_t zslutr[ZDP ? 4096 : 1];   // 4-way replicated
    __shared__ float shu[CT*16];
    __shared__ int   is_last;
    extern __shared__ float obuf[];   // [2, m]

    const int proj = blockIdx.y;
    const int t    = blockIdx.z;   // token slice (nt <= 4; strides from the matcher)
    const int tr   = blockIdx.x;
    const int tid  = threadIdx.x;
    const int ct   = tid % CT;
    const int rg   = tid / CT;

    const uint16_t * trellis = proj ? tre_u : tre_g;
    const float    * su      = proj ? su_u  : su_g;
    x       += (int64_t) t*n;
    scr_v   += (int64_t) t*2*m;
    h       += (int64_t) t*m;
    counter += t;
    float * vout = scr_v + (int64_t) proj*m;

    if (ZDP) {
        for (int i = tid; i < 1024; i += WG) {
            const uint32_t w = ztab[i];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                zslutr[(i << 2) + r] = w;
            }
        }
    } else {
        for (int i = tid; i < 512; i += WG) {
            slutA[i] = __half2float(tlut[2*i + 0]);
            slutB[i] = __half2float(tlut[2*i + 1]);
        }
    }
    // exactly mach1_rt_u_kernel's body, into shared instead of scr_u
    for (int i = tid; i < n; i += WG) {
        shu[i] = su[i] * x[i];
    }
    __syncthreads();
    mach1_fwht_block(shu, n, tid, WG);
    const float sc = __fsqrt_rn((float) n);
    for (int i = tid; i < n; i += WG) {
        shu[i] = __fdiv_rn(shu[i], sc);
    }
    __syncthreads();

    const int  tiles_y = n / 16;
    const bool have    = ct < tiles_y;

    float partial[ROWS];
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
        partial[i] = 0.0f;
    }

    if (have) {
        const float * ub = shu + ct*16;
        float uu[16];
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            uu[c] = ub[c];
        }
        const uint4 * base = (const uint4 *) (trellis + ((int64_t) tr*tiles_y + ct)*64);
        uint32_t qq[4];
        float    fsc = 0.0f;
        if (ZDP) {
            fsc = zs0*mach1_rt_zdp_quant(uu, qq)*(1.0f/127.0f);
        }
        uint4 v[3];
#pragma unroll
        for (int q = 0; q < 3; ++q) {
            v[q] = base[(2*rg + q) & 7];
        }
        const uint32_t * V = (const uint32_t *) v;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const uint32_t A  = V[2*r + 0];
            const uint32_t B  = V[2*r + 1];
            const uint32_t C  = V[2*r + 2];
            const uint32_t Ap = __byte_perm(A, B, 0x5432);
            const uint32_t Bp = __byte_perm(B, C, 0x5432);
            uint32_t st[8];
            st[0] = A;   st[1] = __funnelshift_l(A,  A,  8);
            st[2] = Ap;  st[3] = __funnelshift_l(Ap, Ap, 8);
            st[4] = B;   st[5] = __funnelshift_l(B,  B,  8);
            st[6] = Bp;  st[7] = __funnelshift_l(Bp, Bp, 8);
            uint32_t ph[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                ph[j] = st[j]*(st[j] + 1u);
            }
            if (ZDP) {
                partial[r] = (float) mach1_rt_zdp_row_rep<4>(ph, zslutr, qq, ct & 3)*fsc;
            } else {
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t row = (ph[j] >> 6) & 511u;
                float a0 = slutA[row];
                const float a1 = slutB[row];
                if (ph[j] & 0x8000u) {
                    a0 = -a0;
                }
                acc += a0*uu[2*j] + a1*uu[2*j + 1];
            }
            partial[r] = acc;
            }
        }
    }

    const int lane = ct % warp_size;
    const int wid  = ct / warp_size;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const float s = warp_reduce_sum<warp_size>(partial[r]);
        if (lane == 0) {
            red[rg][r][wid] = s;
        }
    }
    __syncthreads();
    if (tid < 16) {
        const int g = tid / ROWS;
        const int r = tid % ROWS;
        float sum = 0.0f;
#pragma unroll
        for (int wj = 0; wj < n_warps; ++wj) {
            sum += red[g][r][wj];
        }
        vout[tr*16 + g*ROWS + r] = sum;
    }

    // last of the 2*(m/16) blocks folds both out stages and the silu-mul
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        is_last = atomicAdd(counter, 1) == 2*(m/16) - 1;
    }
    __syncthreads();
    if (!is_last) {
        return;
    }
    if (tid == 0) {
        *counter = 0;   // self-reset for the down launch on this stream
    }
    __threadfence();
    float * bg = obuf;
    float * bu = obuf + m;
    for (int i = tid; i < m; i += WG) {
        bg[i] = __ldcg(&scr_v[i]);
        bu[i] = __ldcg(&scr_v[(int64_t) m + i]);
    }
    __syncthreads();
    mach1_fwht_block(bg, m, tid, WG);
    mach1_fwht_block(bu, m, tid, WG);
    const float osc = __fsqrt_rn((float) m);
    for (int i = tid; i < m; i += WG) {
        const float yg = __fdiv_rn(bg[i], osc) * sv_g[i];
        const float yu = __fdiv_rn(bu[i], osc) * sv_u[i];
        h[i] = ggml_cuda_op_silu_single(yg) * yu;
    }
}

// Launch F "shexp down + gate + add": the same walk on x = h. The last block
// folds the out stage, the sigmoid-gate MUL (gate[0] is the [1] sigmoid dst
// the ordinary node ahead of the region computed) and the final ADD with
// moe_out, writing the region's output (the ADD node's dst). __fmul_rn and
// __fadd_rn pin the MUL and ADD kernels' separate roundings - a contracted
// fma would change bits.
// RGT = rows per 16-row tile group. The down shape (n = mff = 512) has only
// 32 column tiles, so RGT 4 idles 96 of each 128-thread vblock; RGT 16
// (CT = 32, one warp) fills them. BIT-EXACT: per-(row, tile) partials are the
// same states in the same order, tiles 0-31 reduce in warp 0's shuffle tree
// either way, and the wider form's extra lanes only ever added exact zeros.
template <int WG, int ZDP = 0, int RGT = 4>
__global__ void __launch_bounds__(WG) mach1_shexp_down_kernel(
        const uint16_t * __restrict__ trellis,
        const half     * __restrict__ tlut,     // pre-rounded fp16
        const float    * __restrict__ su,
        const float    * __restrict__ sv,
        const float    * __restrict__ x,        // h from launch E, [n]
        const float    * __restrict__ gate,     // sigmoid dst, [1]
        const float    * __restrict__ moe,      // moe_out, [m]
        float          * __restrict__ scr_v,    // [m]
        float          * __restrict__ dst,      // [m]
        int            * __restrict__ counter,
        const int m, const int n,
        const uint16_t * __restrict__ ztab,     // ZDP = 1
        const float zs0,                        // ZDP = 1
        const int defer) {                      // FORK_DN: raw scr_v only, fold runs on main
    constexpr int RG        = RGT;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int CT        = WG / RG;
    constexpr int n_warps   = CT / warp_size;
    constexpr int ROWS      = 16 / RG;
    __shared__ float red[RG][16/RG][CT/32 < 1 ? 1 : CT/32];
    __shared__ float slutA[ZDP ? 1 : 512];
    __shared__ float slutB[ZDP ? 1 : 512];
    __shared__ uint32_t zslutr[ZDP ? 4096 : 1];   // 4-way replicated
    __shared__ float shu[(CT < 128 ? 128 : CT)*16];   // the u stage still holds n
    __shared__ int   is_last;
    extern __shared__ float obuf[];   // [m]

    const int t   = blockIdx.z;   // token slice (nt <= 4; strides from the matcher)
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;
    const int ct  = tid % CT;
    const int rg  = tid / CT;

    x       += (int64_t) t*n;      // hbuf slice
    gate    += t;                  // sigmoid dst is [1, nt]
    moe     += (int64_t) t*m;
    scr_v   += (int64_t) t*m;
    dst     += (int64_t) t*m;
    counter += t;

    if (ZDP) {
        for (int i = tid; i < 1024; i += WG) {
            const uint32_t w = ztab[i];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                zslutr[(i << 2) + r] = w;
            }
        }
    } else {
        for (int i = tid; i < 512; i += WG) {
            slutA[i] = __half2float(tlut[2*i + 0]);
            slutB[i] = __half2float(tlut[2*i + 1]);
        }
    }
    for (int i = tid; i < n; i += WG) {
        shu[i] = su[i] * x[i];
    }
    __syncthreads();
    mach1_fwht_block(shu, n, tid, WG);
    const float sc = __fsqrt_rn((float) n);
    for (int i = tid; i < n; i += WG) {
        shu[i] = __fdiv_rn(shu[i], sc);
    }
    __syncthreads();

    const int  tiles_y = n / 16;
    const bool have    = ct < tiles_y;

    float partial[ROWS];
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
        partial[i] = 0.0f;
    }

    if (have) {
        const float * ub = shu + ct*16;
        float uu[16];
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            uu[c] = ub[c];
        }
        const uint4 * base = (const uint4 *) (trellis + ((int64_t) tr*tiles_y + ct)*64);
        uint32_t qq[4];
        float    fsc = 0.0f;
        if (ZDP) {
            fsc = zs0*mach1_rt_zdp_quant(uu, qq)*(1.0f/127.0f);
        }
        uint4 v[3];
        // RGT 16: one row per group - the 2-word row stride is half a uint4,
        // so the window starts at rg>>1 and the row offsets by 2*(rg&1); the
        // & 7 wrap keeps the last row's third word circular, exactly as the
        // RGT 4 form's V[8] wraps to the tile's start
#pragma unroll
        for (int q = 0; q < 3; ++q) {
            v[q] = base[((RG == 16 ? (rg >> 1) : 2*rg) + q) & 7];
        }
        const uint32_t * V = (const uint32_t *) v;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int vo = RG == 16 ? 2*(rg & 1) : 2*r;
            const uint32_t A  = V[vo + 0];
            const uint32_t B  = V[vo + 1];
            const uint32_t C  = V[vo + 2];
            const uint32_t Ap = __byte_perm(A, B, 0x5432);
            const uint32_t Bp = __byte_perm(B, C, 0x5432);
            uint32_t st[8];
            st[0] = A;   st[1] = __funnelshift_l(A,  A,  8);
            st[2] = Ap;  st[3] = __funnelshift_l(Ap, Ap, 8);
            st[4] = B;   st[5] = __funnelshift_l(B,  B,  8);
            st[6] = Bp;  st[7] = __funnelshift_l(Bp, Bp, 8);
            uint32_t ph[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                ph[j] = st[j]*(st[j] + 1u);
            }
            if (ZDP) {
                partial[r] = (float) mach1_rt_zdp_row_rep<4>(ph, zslutr, qq, ct & 3)*fsc;
            } else {
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t row = (ph[j] >> 6) & 511u;
                float a0 = slutA[row];
                const float a1 = slutB[row];
                if (ph[j] & 0x8000u) {
                    a0 = -a0;
                }
                acc += a0*uu[2*j] + a1*uu[2*j + 1];
            }
            partial[r] = acc;
            }
        }
    }

    const int lane = ct % warp_size;
    const int wid  = ct / warp_size;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const float s = warp_reduce_sum<warp_size>(partial[r]);
        if (lane == 0) {
            red[rg][r][wid] = s;
        }
    }
    __syncthreads();
    if (tid < 16) {
        const int g = tid / ROWS;
        const int r = tid % ROWS;
        float sum = 0.0f;
#pragma unroll
        for (int wj = 0; wj < n_warps; ++wj) {
            sum += red[g][r][wj];
        }
        scr_v[tr*16 + g*ROWS + r] = sum;
    }

    // FORK_DN: the walk ran on the fork stream ahead of the moe output; the
    // fold (which reads moe) runs as mach1_shexp_out_kernel on main instead
    if (defer) {
        return;
    }
    // last of the m/16 blocks folds out stage * sigmoid gate + moe add
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        is_last = atomicAdd(counter, 1) == m/16 - 1;
    }
    __syncthreads();
    if (!is_last) {
        return;
    }
    if (tid == 0) {
        *counter = 0;   // self-reset for the next fused op on this stream
    }
    __threadfence();
    for (int i = tid; i < m; i += WG) {
        obuf[i] = __ldcg(&scr_v[i]);
    }
    __syncthreads();
    mach1_fwht_block(obuf, m, tid, WG);
    const float osc = __fsqrt_rn((float) m);
    const float gv  = gate[0];
    for (int i = tid; i < m; i += WG) {
        const float y = __fdiv_rn(obuf[i], osc) * sv[i];
        dst[i] = __fadd_rn(moe[i], __fmul_rn(y, gv));
    }
}

// FORK_DN epilogue: exactly the down kernel's fold - out FWHT, sv scale,
// sigmoid-gate MUL, moe ADD - as its own single-block launch on main, so the
// walk above it can run on the fork stream ahead of the moe output. Same
// arithmetic in the same order: bit-identical to the folded path.
template <int WG>
__global__ void __launch_bounds__(WG) mach1_shexp_out_kernel(
        const float * __restrict__ sv,
        const float * __restrict__ gate,     // sigmoid dst, [1, nt]
        const float * __restrict__ moe,      // moe_out, [nt, m]
        const float * __restrict__ scr_v,    // [nt, m]
        float       * __restrict__ dst,      // [nt, m]
        const int m) {
    extern __shared__ float obuf[];
    const int tid = threadIdx.x;
    // token slice on grid.z (0 offsets at nt = 1)
    const int t = blockIdx.z;
    gate  += t;
    moe   += (int64_t) t*m;
    scr_v += (int64_t) t*m;
    dst   += (int64_t) t*m;
    for (int i = tid; i < m; i += WG) {
        obuf[i] = __ldcg(&scr_v[i]);
    }
    __syncthreads();
    mach1_fwht_block(obuf, m, tid, WG);
    const float osc = __fsqrt_rn((float) m);
    const float gv  = gate[0];
    for (int i = tid; i < m; i += WG) {
        const float y = __fdiv_rn(obuf[i], osc) * sv[i];
        dst[i] = __fadd_rn(moe[i], __fmul_rn(y, gv));
    }
}

// Shared-expert region matcher: at cgraph->nodes[node_idx] ==
//   [i]   MACH1_RT_MM gate            (this node)
//   [i+1] UNARY silu(gate)
//   [i+2] MACH1_RT_MM up              (same x, same tlut)
//   [i+3] MUL (silu, up)
//   [i+4] MACH1_RT_MM down            (x = the MUL dst)
//   [i+5] MUL (down, sigmoid gate [1])
//   [i+6] ADD (moe_out, gated)
// executed as the two launches above. The sigmoid and its mul_mat are
// ordinary nodes the builder expands ahead of the region, so topological
// order places them before node_idx and launch F reads a computed dst.
// Matched one node at a time; any difference falls back to the per-op path.
// Returns nodes to skip or 0.
static int mach1_shexp_gu_batch_p16(ggml_backend_cuda_context & ctx,
                                    ggml_tensor * gp, ggml_tensor * up,
                                    ggml_tensor * par, int skip,
                                    mach1_fork_state * gfork);

int ggml_cuda_mach1_shexp_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    // fold/zbank/fuse_rt/coop serve rt ops from other numeric paths - the
    // fused region replicates the default walk path only. The plain rt bank
    // is not in that list: it never serves an op below eff_dense_min and
    // every fusion here is nt <= 16, so the two cannot meet.
    static const bool on = mach1_env_int("GGML_MACH1_SHEXP_FFN", 1) != 0 &&
                           mach1_env_int("GGML_MACH1_ABLATE", 0) == 0 &&
                           mach1_env_int("GGML_MACH1_FUSE_RT", 0) == 0 &&
                           mach1_env_int("GGML_MACH1_COOP", 0) == 0;
    if (!on || mach1_fold_enabled() || mach1_zbank_enabled()) {
        return 0;
    }

    const int len = 7;
    if (node_idx + len > cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * gp    = cgraph->nodes[node_idx];
    const ggml_tensor * silu  = cgraph->nodes[node_idx + 1];
    const ggml_tensor * up    = cgraph->nodes[node_idx + 2];
    const ggml_tensor * par   = cgraph->nodes[node_idx + 3];
    const ggml_tensor * down  = cgraph->nodes[node_idx + 4];
    const ggml_tensor * gated = cgraph->nodes[node_idx + 5];
    ggml_tensor       * out   = cgraph->nodes[node_idx + 6];
    if (silu->op != GGML_OP_UNARY || up->op != GGML_OP_MACH1_RT_MM || par->op != GGML_OP_MUL ||
        down->op != GGML_OP_MACH1_RT_MM || gated->op != GGML_OP_MUL || out->op != GGML_OP_ADD) {
        return 0;
    }
    if (ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU || silu->src[0] != gp || silu->type != GGML_TYPE_F32) {
        return 0;
    }
    if (par->src[0] != silu || par->src[1] != up || par->type != GGML_TYPE_F32 || !ggml_is_contiguous(par)) {
        return 0;
    }

    // one x and one codebook across the three rt ops; decode only
    const ggml_tensor * x    = gp->src[4];
    const ggml_tensor * tlut = gp->src[3];
    if (up->src[4] != x || up->src[3] != tlut || down->src[4] != par || down->src[3] != tlut) {
        return 0;
    }
    if (tlut->ne[0] != 2 || tlut->ne[1] != 512) {
        return 0;
    }
    if (gp->src[0]->ne[0] != 64 || up->src[0]->ne[0] != 64 || down->src[0]->ne[0] != 64) {
        return 0;
    }
    // nt <= 4: each token is an independent z-slice of both full-region
    // launches. The default-off p16 sibling path below batches only the
    // independent gate/up RT transforms and leaves down/gate/add on their
    // ordinary nodes, preserving the fast p16 down MMA path.
    const int shexp_nt = (int) ggml_nrows(x);
    static const int shexp_nt_max = mach1_env_int("GGML_MACH1_SHEXP_NT", 4);
    static const bool p16_sibling = mach1_env_int("GGML_MACH1_P16_SIBLING_BATCH", 0) != 0;
    const bool p16_gu = p16_sibling && (shexp_nt == 16 || (mach1_nt32_on() && shexp_nt == 32) ||
                                        (mach1_ntlo_on() && shexp_nt > 4 && shexp_nt < 16) ||
                                        mach1_nt_ok(shexp_nt));
    if (x->type != GGML_TYPE_F32 || shexp_nt < 1 ||
        (!p16_gu && (shexp_nt > shexp_nt_max || shexp_nt > 4)) ||
        !ggml_is_contiguous(x)) {
        return 0;
    }

    // gate/up share m and n; the dimensions must fit the WG 512 gate|up walk
    // (n/16 <= 128 column tiles) and the 48 KB static+dynamic shared budget
    // of the folded epilogues (2*mff and md floats)
    const int n   = (int) gp->src[1]->ne[0];
    const int mff = (int) gp->src[2]->ne[0];
    const int md  = (int) down->src[2]->ne[0];
    if ((int) up->src[1]->ne[0] != n || (int) up->src[2]->ne[0] != mff ||
        (int) down->src[1]->ne[0] != mff) {
        return 0;
    }
    if (n % 16 != 0 || mff % 16 != 0 || md % 16 != 0) {
        return 0;
    }
    if (n > 2048 || mff > 4096 || md > 4096) {
        return 0;
    }
    if (par->ne[0] != mff || ggml_nrows(par) != shexp_nt) {
        return 0;
    }

    if (p16_gu) {
        // Mach-1-Small's shared gate/up pair. Batch the two same-input U/TT/out
        // pipelines, then materialize their dependent SILU*MUL in one launch.
        // The down RT op resumes as the next ordinary graph node, so its
        // current rt_mma16 implementation is unchanged.
        if (n != 2048 || mff != 512) {
            return 0;
        }
        const ggml_op gu_ops[4] = {
            GGML_OP_MACH1_RT_MM, GGML_OP_UNARY, GGML_OP_MACH1_RT_MM, GGML_OP_MUL,
        };
        const int gu_out = node_idx + 3;
        if (!ggml_can_fuse_subgraph(cgraph, node_idx, 4, gu_ops, &gu_out, 1)) {
            return 0;
        }
        // FORK_P16: the region reads x and its own weights only, so it can
        // ride s2 under the mega the expert matcher armed. The join is
        // recorded before this call returns, ahead of the down rt node.
        // The only byte the fork writes outside its private scratch is par,
        // and x dies exactly here - so the allocator may put par ON x, which
        // the mega is still reading. Refuse the fork on any such overlap.
        mach1_fork_state * gfork = nullptr;
        if (mach1_fork_on() && mach1_fork_p16_on()) {
            const auto ov = [](const ggml_tensor * a, const ggml_tensor * b) {
                const char * ac = (const char *) a->data;
                const char * bc = (const char *) b->data;
                return ac < bc + ggml_nbytes(b) && bc < ac + ggml_nbytes(a);
            };
            const bool clear = !ov(par, x) && !ov(par, out->src[0]) && !ov(par, out->src[1]) &&
                               !ov(par, gated->src[0]) && !ov(par, gated->src[1]);
            mach1_fork_state * f = clear ? mach1_fork_get(ctx.device, ctx.stream(),
                (size_t) 2*shexp_nt*((size_t) mff + n)) : nullptr;
            if (f != nullptr && f->armed) {
                f->armed = false;
                gfork = f;
            }
            if (!clear && mach1_debug_on()) {
                static std::once_flag once;
                std::call_once(once, []() {
                    fprintf(stderr, "mach1: FORK_P16 declined (par aliases a live region tensor)\n");
                });
            }
        }
        return mach1_shexp_gu_batch_p16(
            ctx, cgraph->nodes[node_idx], cgraph->nodes[node_idx + 2],
            cgraph->nodes[node_idx + 3], 3, gfork);
    }

    const ggml_tensor * sig = gated->src[1];
    if (gated->src[0] != down || sig->op != GGML_OP_UNARY ||
        ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID ||
        sig->type != GGML_TYPE_F32 || ggml_nelements(sig) != shexp_nt ||
        !ggml_is_contiguous(sig)) {
        return 0;
    }
    const ggml_tensor * moe = out->src[0];
    if (out->src[1] != gated || moe == gated || out->type != GGML_TYPE_F32 ||
        out->ne[0] != md || ggml_nrows(out) != shexp_nt || !ggml_is_contiguous(out) ||
        moe->type != GGML_TYPE_F32 || !ggml_is_contiguous(moe)) {
        return 0;
    }
    // The region output must not overlay any tensor the fused kernels still
    // read - the allocator may alias dead byte ranges (the nt-mega's r256
    // lesson). At nt > 1 token slices race; at nt == 1 every thread reads
    // sig[0] while other warps write dst, so the check applies at every nt.
    // In-place onto moe is fine: each element is read and written by the
    // same thread in the same iteration.
    {
        const auto overlap = [](const void * a, size_t an, const void * b, size_t bn) {
            const char * ac = (const char *) a;
            const char * bc = (const char *) b;
            return ac < bc + bn && bc < ac + an;
        };
        const size_t outb = (size_t) shexp_nt*md*sizeof(float);
        if (overlap(out->data, outb, x->data,   (size_t) shexp_nt*n*sizeof(float)) ||
            overlap(out->data, outb, sig->data, (size_t) shexp_nt*sizeof(float))   ||
            (out->data != moe->data &&
             overlap(out->data, outb, moe->data, (size_t) shexp_nt*md*sizeof(float)))) {
            return 0;
        }
    }
    // moe is read while dst is written, so it must come from outside the region
    for (int k = 0; k < len - 1; ++k) {
        if (moe == cgraph->nodes[node_idx + k]) {
            return 0;
        }
    }

    // no intermediate may be a graph output or have external consumers
    const ggml_op ops[7] = { GGML_OP_MACH1_RT_MM, GGML_OP_UNARY, GGML_OP_MACH1_RT_MM, GGML_OP_MUL,
                             GGML_OP_MACH1_RT_MM, GGML_OP_MUL, GGML_OP_ADD };
    const int out_idx = node_idx + len - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, len, ops, &out_idx, 1)) {
        return 0;
    }

    cudaStream_t stream = ctx.stream();
    // nt > 1 needs one completion counter per token slice; the ffn bank's
    // tail slots [392..395] are free of the mega layout and self-reset
    int * cnt = shexp_nt == 1 ? mach1_stream_counter(ctx.device, stream)
                              : mach1_ffn_counters(ctx.device, stream);
    if (cnt == nullptr) {
        return 0;
    }
    if (shexp_nt > 1) {
        cnt += 392;
    }

    // fork: gu runs on s2 behind the ex event the expert matcher recorded,
    // from private scratch; down joins main and keeps its graph position
    mach1_fork_state * fork = nullptr;
    if (mach1_fork_on() && shexp_nt <= mach1_fork_nt()) {
        mach1_fork_state * f = mach1_fork_get(ctx.device, stream,
            (size_t) shexp_nt*((size_t) 3*mff + (size_t) md));
        if (f != nullptr && f->armed) {
            f->armed = false;
            fork = f;
        }
    }
    ggml_cuda_pool_alloc<float> scr_alloc(ctx.pool());
    float * scr_gu;
    if (fork != nullptr) {
        scr_gu = fork->scr;
    } else {
        scr_gu = scr_alloc.alloc((size_t) shexp_nt*(3*mff + md));
    }
    float * hbuf   = scr_gu + (size_t) shexp_nt*2*mff;   // [nt, mff]
    float * scr_d  = hbuf + (size_t) shexp_nt*mff;       // [nt, md]
    cudaStream_t gu_stream = stream;
    if (fork != nullptr) {
        CUDA_CHECK(cudaStreamWaitEvent(fork->s2, fork->ex, 0));
        gu_stream = fork->s2;
    }

    char shp[96];
    snprintf(shp, sizeof(shp), " mff=%d n=%d md=%d nt=%d", mff, n, md, shexp_nt);

    const mach1_rt_ztab zt = mach1_zdp_on() ? mach1_rt_z_table(tlut->data, stream)
                                            : mach1_rt_ztab{nullptr, 0.0f};
    const bool zdp = zt.ztab != nullptr;
    mach1_timed(stream, std::string("shexp_gu") + shp, [&]() {
    const ggml_cuda_kernel_launch_params gu =
        ggml_cuda_kernel_launch_params(dim3(mff/16, 2, shexp_nt), dim3(512, 1, 1),
                                       (size_t) 2*mff*sizeof(float), gu_stream);
    if (zdp) {
        mach1_launch(mach1_shexp_gu_kernel<512, 1>, gu,
            (const uint16_t *) gp->src[0]->data, (const uint16_t *) up->src[0]->data,
            (const half *) tlut->data,
            (const float *) gp->src[1]->data, (const float *) up->src[1]->data,
            (const float *) gp->src[2]->data, (const float *) up->src[2]->data,
            (const float *) x->data, scr_gu, hbuf, cnt, mff, n, zt.ztab, zt.zstep);
    } else {
        mach1_launch(mach1_shexp_gu_kernel<512>, gu,
            (const uint16_t *) gp->src[0]->data, (const uint16_t *) up->src[0]->data,
            (const half *) tlut->data,
            (const float *) gp->src[1]->data, (const float *) up->src[1]->data,
            (const float *) gp->src[2]->data, (const float *) up->src[2]->data,
            (const float *) x->data, scr_gu, hbuf, cnt, mff, n,
            (const uint16_t *) nullptr, 0.0f);
    }
    });
    // FORK_DN: the down WALK only needs hbuf (s2's own product), so it stays
    // on s2 ahead of the moe output; the fold - which reads moe - runs as the
    // epilogue kernel on main after the join. Schedule only, values identical.
    static const bool fork_dn_env = mach1_env_int("GGML_MACH1_FORK_DN", 0) != 0;
    const bool dn_fork = fork != nullptr && fork_dn_env;
    if (fork != nullptr && !dn_fork) {
        CUDA_CHECK(cudaEventRecord(fork->egu, fork->s2));
        CUDA_CHECK(cudaStreamWaitEvent(stream, fork->egu, 0));
    }
    const int defer = dn_fork ? 1 : 0;
    // down walk WG mirrors the unfused dispatch: 512 up to 128 column tiles,
    // 1024 above (FWHT values are WG-independent, the walk layout is not)
    static const bool dn16 = mach1_env_int("GGML_MACH1_SHEXP_DN16", 0) != 0;
    mach1_timed(stream, std::string("shexp_down") + shp, [&]() {
    const ggml_cuda_kernel_launch_params fp =
        ggml_cuda_kernel_launch_params(dim3(md/16, 1, shexp_nt),
            dim3(mff/16 <= 128 ? 512 : 1024, 1, 1), (size_t) md*sizeof(float),
            dn_fork ? fork->s2 : stream);
    if (dn16 && mff/16 <= 32 && zdp) {
        mach1_launch(mach1_shexp_down_kernel<512, 1, 16>, fp,
            (const uint16_t *) down->src[0]->data, (const half *) tlut->data,
            (const float *) down->src[1]->data, (const float *) down->src[2]->data,
            (const float *) hbuf, (const float *) sig->data, (const float *) moe->data,
            scr_d, (float *) out->data, cnt, md, mff, zt.ztab, zt.zstep, defer);
    } else if (dn16 && mff/16 <= 32) {
        mach1_launch(mach1_shexp_down_kernel<512, 0, 16>, fp,
            (const uint16_t *) down->src[0]->data, (const half *) tlut->data,
            (const float *) down->src[1]->data, (const float *) down->src[2]->data,
            (const float *) hbuf, (const float *) sig->data, (const float *) moe->data,
            scr_d, (float *) out->data, cnt, md, mff, (const uint16_t *) nullptr, 0.0f, defer);
    } else if (mff/16 <= 128 && zdp) {
        mach1_launch(mach1_shexp_down_kernel<512, 1>, fp,
            (const uint16_t *) down->src[0]->data, (const half *) tlut->data,
            (const float *) down->src[1]->data, (const float *) down->src[2]->data,
            (const float *) hbuf, (const float *) sig->data, (const float *) moe->data,
            scr_d, (float *) out->data, cnt, md, mff, zt.ztab, zt.zstep, defer);
    } else if (mff/16 <= 128) {
        mach1_launch(mach1_shexp_down_kernel<512>, fp,
            (const uint16_t *) down->src[0]->data, (const half *) tlut->data,
            (const float *) down->src[1]->data, (const float *) down->src[2]->data,
            (const float *) hbuf, (const float *) sig->data, (const float *) moe->data,
            scr_d, (float *) out->data, cnt, md, mff, (const uint16_t *) nullptr, 0.0f, defer);
    } else if (zdp) {
        mach1_launch(mach1_shexp_down_kernel<1024, 1>, fp,
            (const uint16_t *) down->src[0]->data, (const half *) tlut->data,
            (const float *) down->src[1]->data, (const float *) down->src[2]->data,
            (const float *) hbuf, (const float *) sig->data, (const float *) moe->data,
            scr_d, (float *) out->data, cnt, md, mff, zt.ztab, zt.zstep, defer);
    } else {
        mach1_launch(mach1_shexp_down_kernel<1024>, fp,
            (const uint16_t *) down->src[0]->data, (const half *) tlut->data,
            (const float *) down->src[1]->data, (const float *) down->src[2]->data,
            (const float *) hbuf, (const float *) sig->data, (const float *) moe->data,
            scr_d, (float *) out->data, cnt, md, mff, (const uint16_t *) nullptr, 0.0f, defer);
    }
    });

    // FORK_DN: join main to the s2 walk, then run the deferred fold - the
    // exact copy of the down kernel's epilogue - on main, where moe is ready
    if (dn_fork) {
        CUDA_CHECK(cudaEventRecord(fork->edn, fork->s2));
        CUDA_CHECK(cudaStreamWaitEvent(stream, fork->edn, 0));
        const int owg = mff/16 <= 128 ? 512 : 1024;
        mach1_timed(stream, std::string("shexp_out") + shp, [&]() {
        const ggml_cuda_kernel_launch_params op =
            ggml_cuda_kernel_launch_params(dim3(1, 1, shexp_nt), dim3(owg, 1, 1),
                                           (size_t) md*sizeof(float), stream);
        if (owg == 512) {
            mach1_launch(mach1_shexp_out_kernel<512>, op,
                (const float *) down->src[2]->data, (const float *) sig->data,
                (const float *) moe->data, (const float *) scr_d, (float *) out->data, md);
        } else {
            mach1_launch(mach1_shexp_out_kernel<1024>, op,
                (const float *) down->src[2]->data, (const float *) sig->data,
                (const float *) moe->data, (const float *) scr_d, (float *) out->data, md);
        }
        });
    }

    return len - 1;
}

// Batched independent same-input rt ops (GGML_MACH1_QKV_BATCH=1; decode
// only). The attention layers emit three independent rt ops on one input
// (wq/wk/wv, build_layer_attn in src/models/mach1.cpp) and the GDN layers
// two (wqkv + wqkv_gate, both trailed by the v_tiled row reorder). Their
// serialization was launch order only - the walks are independent - so one
// launch runs them all on disjoint blocks and the group's wall clock
// approaches max() instead of sum(). blockIdx.x maps to (op, tile row) via
// per-op first-block offsets; the block body is EXACTLY
// mach1_rt_walk_rows_v_kernel with per-op trellis/su/sv pointers (x is
// shared, su is not, so x' = H(su .* x) differs per op and the UFUSE
// prologue runs with the block's own su). Each op's out stage folds in via
// its own completion counter (the certified OFUSE pattern), writing the
// op's dst directly or through the vtiled row scatter. An op outside the
// UFUSE window reads a precomputed scr_u instead, exactly as the unfused
// dispatch does.
struct mach1_rt_batch_op {
    const uint16_t * trellis;
    const float    * su;
    const float    * sv;
    const float    * scr_u;   // non-null: outside the UFUSE window, x' precomputed by rt_u
    float          * scr_v;
    float          * dst;
    const int      * perm;    // optional vtiled row scatter
    int              m;
    int              blk0;    // first block of this op
};

// PROBE (GGML_MACH1_QKVB_PROBE, measurement only - output is garbage):
// 1 = keep the state math, skip the zslut gather and dp4a
// 2 = keep the gather count, make the index lane-uniform (conflict share)
// 3 = skip the folded out stage (tail share)
// MINB pins blocks/SM: the r238 OSPLIT lesson was that the REGISTER file
// (not the fold workspace) caps this walk's residency, so widening means
// telling ptxas the budget. MINB 2 at WG 512 demands 64 registers.
template <int WG, int TCF = 0, int ZDP = 0, int PROBE = 0, int MINB = 1>
__global__ void __launch_bounds__(WG, MINB) mach1_rt_qkv_batch_kernel(
        const mach1_rt_batch_op op0,
        const mach1_rt_batch_op op1,
        const mach1_rt_batch_op op2,    // m = 0 when only two ops are batched
        const half  * __restrict__ tlut,     // pre-rounded fp16
        const float * __restrict__ x,
        int         * __restrict__ counters, // one per op
        const int n,
        const half  * __restrict__ htab,     // TCF = 1 (see the rows_v kernel)
        const uint16_t * __restrict__ ztab,  // ZDP = 1
        const float zs0,                     // ZDP = 1
        const int osplit) {                  // QKVB_OSPLIT: no fold, out runs as its own launch
    constexpr int RG        = 4;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int CT        = WG / RG;
    constexpr int n_warps   = CT / warp_size;
    constexpr int ROWS      = 16 / RG;
    __shared__ float red[RG][16/RG][CT/32];
    __shared__ float slutA[ZDP ? 1 : 512];
    __shared__ float slutB[ZDP ? 1 : 512];
    __shared__ uint32_t zslutr[ZDP ? 2048 : 1];   // 2-way replicated (4 + the TC ws busts 99 KB)
    __shared__ float shu[CT*16];
    __shared__ int   is_last;
    extern __shared__ float obuf[];   // fp fold only; the ZDP fold is chunked through shu

    int opi;
    mach1_rt_batch_op op;
    if (op2.m > 0 && (int) blockIdx.x >= op2.blk0) {
        opi = 2; op = op2;
    } else if ((int) blockIdx.x >= op1.blk0) {
        opi = 1; op = op1;
    } else {
        opi = 0; op = op0;
    }
    // token z-slice (nt <= 4): every per-token pointer moves by its row
    // stride; scr_u/scr_v are laid [nt, n]/[nt, m] by the exec, and the
    // completion counters come in trios per slice
    const int t   = blockIdx.z;
    x        += (int64_t) t*n;
    counters += t*3;
    op.scr_v += (int64_t) t*op.m;
    op.dst   += (int64_t) t*op.m;
    if (op.scr_u != nullptr) {
        op.scr_u += (int64_t) t*n;
    }
    const int tr  = blockIdx.x - op.blk0;
    const int tid = threadIdx.x;
    const int ct  = tid % CT;
    const int rg  = tid / CT;

    const uint16_t * __restrict__ trellis = op.trellis;

    if (ZDP) {
        for (int i = tid; i < 1024; i += WG) {
            const uint32_t w = ztab[i];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                zslutr[(i << 1) + r] = w;
            }
        }
    } else {
        for (int i = tid; i < 512; i += WG) {
            slutA[i] = __half2float(tlut[2*i + 0]);
            slutB[i] = __half2float(tlut[2*i + 1]);
        }
    }
    if (op.scr_u == nullptr && TCF) {
        mach1_tc_fwht_dyn<WG, false>(n, x, op.su, shu, nullptr, nullptr,
                                     htab, (half *) obuf, tid);
    } else if (op.scr_u == nullptr) {
        // exactly mach1_rt_u_kernel's body, into shared instead of scr_u
        for (int i = tid; i < n; i += WG) {
            shu[i] = op.su[i] * x[i];
        }
        __syncthreads();
        mach1_fwht_block(shu, n, tid, WG);
        const float sc = __fsqrt_rn((float) n);
        for (int i = tid; i < n; i += WG) {
            shu[i] = __fdiv_rn(shu[i], sc);
        }
    } else {
        for (int i = tid; i < n; i += WG) {
            shu[i] = op.scr_u[i];
        }
    }
    __syncthreads();

    const int  tiles_y = n / 16;
    const bool have    = ct < tiles_y;

    float partial[ROWS];
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
        partial[i] = 0.0f;
    }

    if (have) {
        const float * ub = shu + ct*16;
        float uu[16];
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            uu[c] = ub[c];
        }
        uint32_t qq[4];
        float    fsc = 0.0f;
        if (ZDP) {
            fsc = zs0*mach1_rt_zdp_quant(uu, qq)*(1.0f/127.0f);
        }
        const uint4 * base = (const uint4 *) (trellis + ((int64_t) tr*tiles_y + ct)*64);
        uint4 v[3];
#pragma unroll
        for (int q = 0; q < 3; ++q) {
            v[q] = base[(2*rg + q) & 7];
        }
        const uint32_t * V = (const uint32_t *) v;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const uint32_t A  = V[2*r + 0];
            const uint32_t B  = V[2*r + 1];
            const uint32_t C  = V[2*r + 2];
            const uint32_t Ap = __byte_perm(A, B, 0x5432);
            const uint32_t Bp = __byte_perm(B, C, 0x5432);
            uint32_t st[8];
            st[0] = A;   st[1] = __funnelshift_l(A,  A,  8);
            st[2] = Ap;  st[3] = __funnelshift_l(Ap, Ap, 8);
            st[4] = B;   st[5] = __funnelshift_l(B,  B,  8);
            st[6] = Bp;  st[7] = __funnelshift_l(Bp, Bp, 8);
            uint32_t ph[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                ph[j] = st[j]*(st[j] + 1u);
            }
            if (PROBE == 1) {
                // consume the states so the chain is not dead code
                partial[r] = (float)((ph[0] + ph[3] + ph[5] + ph[7]) & 0xFFu)*fsc;
            } else if (PROBE == 2) {
                uint32_t phu[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    phu[j] = (ph[j] & ~0xFFC0u) | ((uint32_t)(ct & 1023) << 6);
                }
                partial[r] = (float) mach1_rt_zdp_row_rep<2>(phu, zslutr, qq, ct & 1)*fsc;
            } else if (ZDP) {
                partial[r] = (float) mach1_rt_zdp_row_rep<2>(ph, zslutr, qq, ct & 1)*fsc;
            } else {
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t row = (ph[j] >> 6) & 511u;
                float a0 = slutA[row];
                const float a1 = slutB[row];
                if (ph[j] & 0x8000u) {
                    a0 = -a0;
                }
                acc += a0*uu[2*j] + a1*uu[2*j + 1];
            }
            partial[r] = acc;
            }
        }
    }

    const int lane = ct % warp_size;
    const int wid  = ct / warp_size;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const float s = warp_reduce_sum<warp_size>(partial[r]);
        if (lane == 0) {
            red[rg][r][wid] = s;
        }
    }
    __syncthreads();
    if (tid < 16) {
        const int g = tid / ROWS;
        const int r = tid % ROWS;
        float sum = 0.0f;
#pragma unroll
        for (int wj = 0; wj < n_warps; ++wj) {
            sum += red[g][r][wj];
        }
        op.scr_v[tr*16 + g*ROWS + r] = sum;
    }
    // OSPLIT: the raw v is the kernel's whole product - the out stage runs as
    // a trailing batched TC launch, so the walk carries no fold workspace and
    // packs 3 blocks/SM instead of 1 (the 77 KB TC ws was capping the waves)
    if (osplit) {
        return;
    }
    // the last block of THIS op folds its out stage; the counters of the
    // other ops keep running independently
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        is_last = atomicAdd(&counters[opi], 1) == op.m/16 - 1;
    }
    __syncthreads();
    if (!is_last) {
        return;
    }
    if (tid == 0) {
        counters[opi] = 0;   // self-reset for the next batch on this stream
    }
    __threadfence();
    if (PROBE == 3) {
        // no out fold: prices the serial tail. Zeroed dst keeps downstream finite.
        for (int i = tid; i < op.m; i += WG) {
            if (op.perm == nullptr) {
                op.dst[i] = 0.0f;
            } else {
                op.dst[op.perm[i]] = 0.0f;
            }
        }
        return;
    }
    if (ZDP && !TCF) {
        // chunked in-place fold through shu: no obuf, so the launch carries no
        // dynamic shared. r221 measured this SLOWER than the TC fold on Ada
        // (44.8 vs 34.9 us) - kept selectable for the fold A/B, not default.
        mach1_fwht_chunked<WG>(op.scr_v, shu, op.m, tid);
        const float oscz = __fsqrt_rn((float) op.m);
        if (op.perm == nullptr) {
            for (int i = tid; i < op.m; i += WG) {
                op.dst[i] = __fdiv_rn(op.scr_v[i], oscz) * op.sv[i];
            }
        } else {
            for (int i = tid; i < op.m; i += WG) {
                op.dst[op.perm[i]] = __fdiv_rn(op.scr_v[i], oscz) * op.sv[i];
            }
        }
        return;
    }
    if (TCF) {
        mach1_tc_fwht_dyn<WG, true>(op.m, op.scr_v, nullptr, op.dst, op.sv, op.perm,
                                    htab, (half *) obuf, tid);
        return;
    }
    for (int i = tid; i < op.m; i += WG) {
        obuf[i] = __ldcg(&op.scr_v[i]);
    }
    __syncthreads();
    mach1_fwht_block(obuf, op.m, tid, WG);
    const float osc = __fsqrt_rn((float) op.m);
    if (op.perm == nullptr) {
        for (int i = tid; i < op.m; i += WG) {
            op.dst[i] = __fdiv_rn(obuf[i], osc) * op.sv[i];
        }
    } else {
        // vtiled fold: identical values stored at the permuted row
        for (int i = tid; i < op.m; i += WG) {
            op.dst[op.perm[i]] = __fdiv_rn(obuf[i], osc) * op.sv[i];
        }
    }
}

// QKVB_OSPLIT companions: the u and out transforms of every batched op in ONE
// launch each (block b serves op b), so the group's serial tail is max() of
// the single-block TC transforms instead of their sum, and the walk between
// them runs with no dynamic shared. Bitwise identical to the folded path: the
// same mach1_tc_fwht_dyn body on the same inputs.
struct mach1_rt_u_batch {
    const float * su;
    float       * scr_u;
};

template <int WG, bool PACKED_F16 = false, bool PACKED_Q8 = false>
__global__ void mach1_rt_u_tc_batch_kernel(
        const float * __restrict__ su0, float * __restrict__ scr_u0,
        const float * __restrict__ su1, float * __restrict__ scr_u1,
        const float * __restrict__ su2, float * __restrict__ scr_u2,
        const half * __restrict__ htab, const float * __restrict__ x, const int n,
        const float q8_zstep) {
    extern __shared__ float obuf[];
    static_assert(!(PACKED_F16 && PACKED_Q8),
                  "batched p16 activation formats are exclusive");
    const float * su = blockIdx.x == 0 ? su0 : (blockIdx.x == 1 ? su1 : su2);
    float * scr_u = blockIdx.x == 0 ? scr_u0 : (blockIdx.x == 1 ? scr_u1 : scr_u2);
    if (scr_u == nullptr) {
        return;
    }
    // token slice: scratch rows are [nt, n], grid.z = nt (0 offsets at nt=1)
    const int t = blockIdx.z;
    if constexpr (PACKED_Q8) {
        // Match the standalone IMMA8 handoff exactly: one q8 byte per value
        // followed by one combined float scale per K16, all inside the
        // existing n-float token slot.
        mach1_tc_fwht_apply<32, 64, WG, false, false, true>(
            x + (int64_t) t*n, su, scr_u + (int64_t) t*n,
            nullptr, nullptr, htab + mach1_tc_hoff(32), htab + mach1_tc_hoff(64),
            (half *) obuf, threadIdx.x, q8_zstep);
    } else if constexpr (PACKED_F16) {
        // The p16 compressed spine consumes half rows. Keep them in the front
        // of the existing fp32 scratch allocation, exactly like the standalone
        // rt_mma16 u stage, while the op dimension lets sibling transforms
        // occupy the device together instead of serializing 16-block grids.
        float * packed = (float *) ((half *) scr_u + (int64_t) t*n);
        mach1_tc_fwht_apply<32, 64, WG, false, true>(
            x + (int64_t) t*n, su, packed, nullptr, nullptr,
            htab + mach1_tc_hoff(32), htab + mach1_tc_hoff(64),
            (half *) obuf, threadIdx.x);
    } else {
        mach1_tc_fwht_dyn<WG, false>(n, x + (int64_t) t*n, su, scr_u + (int64_t) t*n,
                                     nullptr, nullptr,
                                     htab, (half *) obuf, threadIdx.x);
    }
}

// p16 attention QKV has one compressed-MMA projection (q, m=8192) and two
// ordinary TT projections (k/v, m=512). All three share n=2048 and x, so one
// grid can overlap their U transforms; q stores whichever packed handoff the
// standalone compressed spine selected, while k/v remain fp32.
template <int WG, bool PACKED_Q8 = false>
__global__ void mach1_rt_u_tc_mixed_p16_kernel(
        const mach1_rt_u_batch u0, const mach1_rt_u_batch u1, const mach1_rt_u_batch u2,
        const half * __restrict__ htab, const float * __restrict__ x,
        const float q8_zstep) {
    extern __shared__ float obuf[];
    const int opi = blockIdx.x;
    const mach1_rt_u_batch & u = opi == 0 ? u0 : (opi == 1 ? u1 : u2);
    const int t = blockIdx.z;
    const float * xt = x + (int64_t) t*2048;
    if (opi == 0) {
        if constexpr (PACKED_Q8) {
            mach1_tc_fwht_apply<32, 64, WG, false, false, true>(
                xt, u.su, u.scr_u + (int64_t) t*2048, nullptr, nullptr,
                htab + mach1_tc_hoff(32), htab + mach1_tc_hoff(64),
                (half *) obuf, threadIdx.x, q8_zstep);
        } else {
            float * packed = (float *) ((half *) u.scr_u + (int64_t) t*2048);
            mach1_tc_fwht_apply<32, 64, WG, false, true>(
                xt, u.su, packed, nullptr, nullptr,
                htab + mach1_tc_hoff(32), htab + mach1_tc_hoff(64),
                (half *) obuf, threadIdx.x);
        }
    } else {
        mach1_tc_fwht_apply<32, 64, WG, false>(
            xt, u.su, u.scr_u + (int64_t) t*2048, nullptr, nullptr,
            htab + mach1_tc_hoff(32), htab + mach1_tc_hoff(64),
            (half *) obuf, threadIdx.x);
    }
}

template <int WG>
__global__ void __launch_bounds__(WG) mach1_rt_out_tc_batch_kernel(
        const mach1_rt_batch_op op0, const mach1_rt_batch_op op1, const mach1_rt_batch_op op2,
        const half * __restrict__ htab) {
    extern __shared__ float obuf[];
    const mach1_rt_batch_op & op = blockIdx.x == 0 ? op0 : (blockIdx.x == 1 ? op1 : op2);
    if (op.m == 0) {
        return;
    }
    // token slice: scr_v and dst rows are [nt, m], grid.z = nt
    const int t = blockIdx.z;
    mach1_tc_fwht_dyn<WG, true>(op.m, op.scr_v + (int64_t) t*op.m, nullptr,
                                op.dst + (int64_t) t*op.m, op.sv, op.perm,
                                htab, (half *) obuf, threadIdx.x);
}

// standard-output-basis sibling epilogue (mach1_rt_stdout_on): the TC body's
// out[perm[i]] = y_i * sv[i] store with y = v, no output Hadamard. grid.y
// chunks m so the wide 8192-row ops spread over SMs instead of leaving one
// block per (op, token).
#define MACH1_RT_RSB_CHUNK 2048
__global__ void mach1_rt_out_rs_batch_kernel(
        const mach1_rt_batch_op op0, const mach1_rt_batch_op op1, const mach1_rt_batch_op op2) {
    const mach1_rt_batch_op & op = blockIdx.x == 0 ? op0 : (blockIdx.x == 1 ? op1 : op2);
    if (op.m == 0) {
        return;
    }
    const int c0 = blockIdx.y*MACH1_RT_RSB_CHUNK;
    if (c0 >= op.m) {
        return;
    }
    const int t   = blockIdx.z;
    const int end = min(op.m, c0 + MACH1_RT_RSB_CHUNK);
    const float * src = op.scr_v + (int64_t) t*op.m;
    float       * dst = op.dst + (int64_t) t*op.m;
    for (int i = c0 + threadIdx.x; i < end; i += blockDim.x) {
        const int oi = op.perm != nullptr ? op.perm[i] : i;
        dst[oi] = op.sv != nullptr ? src[i]*op.sv[i] : src[i];
    }
}

__global__ void mach1_silu_mul_p16_kernel(
        const float * __restrict__ gate, const float * __restrict__ up,
        float * __restrict__ dst, const int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
         i < n; i += (int64_t) blockDim.x*gridDim.x) {
        dst[i] = ggml_cuda_op_silu_single(gate[i])*up[i];
    }
}

// fused shared-expert pair epilogue (GGML_MACH1_SHEXP_OGLU): both out
// transforms in one block, then glu = silu(gate)*up straight from shared.
// Same tc_fwht_dyn body, sv fold, and silu expression in the same order as
// the split path, so values are bit-identical; only the two transformed
// halves' global round trip and the separate silu launch disappear.
template <int WG>
__global__ void mach1_shexp_out_glu_tc_kernel(
        const mach1_rt_batch_op op0, const mach1_rt_batch_op op1,
        const half * __restrict__ htab, float * __restrict__ glu) {
    extern __shared__ float obuf[];
    const int m = op0.m;
    float * gbuf = obuf;
    float * ubuf = obuf + m;
    half  * ws   = (half *) (obuf + 2*m);
    const int t = blockIdx.z;
    mach1_tc_fwht_dyn<WG, true>(m, op0.scr_v + (int64_t) t*m, nullptr, gbuf,
                                op0.sv, op0.perm, htab, ws, threadIdx.x);
    __syncthreads();
    mach1_tc_fwht_dyn<WG, true>(m, op1.scr_v + (int64_t) t*m, nullptr, ubuf,
                                op1.sv, op1.perm, htab, ws, threadIdx.x);
    __syncthreads();
    float * out = glu + (int64_t) t*m;
    for (int i = threadIdx.x; i < m; i += WG) {
        out[i] = ggml_cuda_op_silu_single(gbuf[i])*ubuf[i];
    }
}

// Shared executor for the qkv trio and the GDN qkvz pair: rt_u launches for
// the ops outside the UFUSE window (mirrors ggml_cuda_op_mach1_rt_mm), then
// one batched walk over sum(m/16) blocks. The per-op counters ride the ffn
// counter bank: same-stream kernels serialize and every counter self-resets
// before its kernel exits, so the ranges cannot collide.
// the region's independent walks as ONE grid: op-select by blk0 (the sliced
// batch kernel's pattern), each block running the decode-once TT body with
// its row tile rebased into its op. No ordering between ops is needed - the
// walks share nothing - so this removes (nops-1) launch boundaries per
// region without replicating any per-token work (the r283 rule).
template <int WG, int TT, int ZDP, int MINB = 1>
__global__ void __launch_bounds__(WG, MINB) mach1_rt_walk_tt_pair_kernel(
        const mach1_rt_batch_op op0, const mach1_rt_batch_op op1, const mach1_rt_batch_op op2,
        const half * __restrict__ tlut, const int n, const int nt,
        const uint16_t * __restrict__ ztab, const float zs0) {
    mach1_rt_batch_op op = op0;
    if (op2.m > 0 && (int) blockIdx.x >= op2.blk0) {
        op = op2;
    } else if (op1.m > 0 && (int) blockIdx.x >= op1.blk0) {
        op = op1;
    }
    mach1_rt_walk_tt_body<WG, TT, ZDP, 0, 4, 0>(op.trellis, tlut, op.scr_u, op.scr_v,
        op.m, n, nt, ztab, zs0,
        (const float *) nullptr, (float *) nullptr, (int *) nullptr,
        (const float *) nullptr, (const float *) nullptr,
        (int) blockIdx.x - op.blk0);
}

// the TT walk's template selection (mirrors the wvec TT branch: wg by column
// tiles, ttc by wg and nt, ZDP by table presence) for the batch exec's
// b_nt >= 2 path: one pair-select launch covers every op in the region
static void mach1_rt_walk_tt_launch(cudaStream_t stream,
        const mach1_rt_batch_op ops[3], const half * tlut,
        const int blk, const int n, const int nt, const mach1_rt_ztab & zt) {
    const int  wg  = n/16 <= 128 ? 512 : 1024;
    const int  ttc = (wg == 512 && nt >= 3) ? 4 : 2;
    const bool zdp = zt.ztab != nullptr;
    // GGML_MACH1_WALK_MINB=2: the two-blocks-per-SM pins (wg 512 only - the
    // 1024-thread forms already fill the register file)
    static const int minb = mach1_env_int("GGML_MACH1_WALK_MINB", 1);
    const ggml_cuda_kernel_launch_params tp =
        ggml_cuda_kernel_launch_params(dim3(blk, 1, (nt + ttc - 1)/ttc), dim3(wg, 1, 1), 0, stream);
#define M1_TTW_H(WGT, TTV, ZD) \
    do { if (minb >= 2 && WGT == 512) { \
        static bool carve = false; \
        if (!carve) { carve = true; \
            cudaFuncSetAttribute((const void *) mach1_rt_walk_tt_pair_kernel<WGT, TTV, ZD, 2>, \
                cudaFuncAttributePreferredSharedMemoryCarveout, 100); \
            cudaGetLastError(); \
        } \
        mach1_launch(mach1_rt_walk_tt_pair_kernel<WGT, TTV, ZD, 2>, tp, \
            ops[0], ops[1], ops[2], tlut, n, nt, \
            ZD ? zt.ztab : (const uint16_t *) nullptr, ZD ? zt.zstep : 0.0f); \
    } else { \
        mach1_launch(mach1_rt_walk_tt_pair_kernel<WGT, TTV, ZD>, tp, \
            ops[0], ops[1], ops[2], tlut, n, nt, \
            ZD ? zt.ztab : (const uint16_t *) nullptr, ZD ? zt.zstep : 0.0f); \
    } } while (0)
    if (wg == 512 && ttc == 4) {
        if (zdp) { M1_TTW_H(512, 4, 1); } else { M1_TTW_H(512, 4, 0); }
    } else if (wg == 512) {
        if (zdp) { M1_TTW_H(512, 2, 1); } else { M1_TTW_H(512, 2, 0); }
    } else {
        if (zdp) { M1_TTW_H(1024, 2, 1); } else { M1_TTW_H(1024, 2, 0); }
    }
#undef M1_TTW_H
}

// GGML_MACH1_TC_WG=1024 for the sibling out batch: the qkvz/qkv shapes are
// m = 4096/8192 (A = 64), where 1024 threads keep every warp group on an mi
// row and halve the per-warp MMA row count. Bit-identical - see mach1_tc_wg.
static void mach1_rt_out_tcb_launch(cudaStream_t stream, const dim3 & grid, const size_t osmem,
                                    const mach1_rt_batch_op & op0, const mach1_rt_batch_op & op1,
                                    const mach1_rt_batch_op & op2, const half * tc_htab,
                                    const int mmax) {
    if (mach1_tc_wg() == 1024 && mmax >= 4096) {
        mach1_smem_opt_in((const void *) mach1_rt_out_tc_batch_kernel<1024>, osmem);
        mach1_launch(mach1_rt_out_tc_batch_kernel<1024>,
            ggml_cuda_kernel_launch_params(grid, dim3(1024, 1, 1), osmem, stream),
            op0, op1, op2, tc_htab);
        return;
    }
    mach1_smem_opt_in((const void *) mach1_rt_out_tc_batch_kernel<512>, osmem);
    mach1_launch(mach1_rt_out_tc_batch_kernel<512>,
        ggml_cuda_kernel_launch_params(grid, dim3(512, 1, 1), osmem, stream),
        op0, op1, op2, tc_htab);
}

static int mach1_rt_batch_exec(ggml_backend_cuda_context & ctx, const char * key,
                               const ggml_tensor * const rt[], ggml_tensor * const out[],
                               const int * const perm[], int nops, int skip,
                               mach1_fork_state * zf = nullptr,
                               const bool allow_mma16_pair = false,
                               const bool allow_mma16_trio = false,
                               ggml_tensor * glu_out = nullptr,
                               mach1_fork_state * gfork = nullptr) {
    const int n = (int) rt[0]->src[1]->ne[0];
    // token slices (nt <= 4, matcher-gated): scratch and counters scale by
    // nt and the kernel's blockIdx.z picks the slice
    const int b_nt = (int) ggml_nrows(rt[0]->src[4]);

    cudaStream_t stream = ctx.stream();
    int * cnt = mach1_ffn_counters(ctx.device, stream);
    if (cnt == nullptr) {
        return 0;
    }

    static const int ufuse_maxb = mach1_env_int("GGML_MACH1_UFUSE_MAXB", 256);
    const half * tc_htab = mach1_tc_fwht_tab(ctx.device, stream);
    const mach1_rt_ztab zt = mach1_zdp_on() ? mach1_rt_z_table(rt[0]->src[3]->data, stream)
                                            : mach1_rt_ztab{nullptr, 0.0f};
    const bool zdp = zt.ztab != nullptr;
    static const int qprobe = mach1_env_int("GGML_MACH1_QKVB_PROBE", 0);
    static const bool qchunk = mach1_env_int("GGML_MACH1_QKVB_CHUNK", 0) != 0;
    static const int osplit_lvl = mach1_env_int("GGML_MACH1_QKVB_OSPLIT", 0);

    int  m[3]     = { 0, 0, 0 };
    bool ufuse[3] = { false, false, false };
    size_t scr_elems = 0;
    // wg 512 walks only - the 1024-thread TCF form exceeds the register
    // budget, see the dispatch comment in ggml_cuda_op_mach1_rt_mm
    bool tcw = tc_htab != nullptr && mach1_tc_in_walk() && n/16 <= 128 && b_nt == 1;
    for (int i = 0; i < nops; ++i) {
        m[i] = (int) rt[i]->src[2]->ne[0];
    }
    // Default-off p16 GDN-pair path. It preserves the selected compressed
    // spine handoff and changes only scheduling around it: the two
    // independent same-input u transforms and the two independent out
    // transforms become one op-by-token grid each. No persistent weights or
    // activations are added; the already-required per-op scratch is reused.
    static const bool mma16_pair_on = mach1_env_int("GGML_MACH1_QKVZ_MMA16_BATCH", 0) != 0;
    static const bool rt_mma16_on   = mach1_env_int("GGML_MACH1_RT_MMA16", 0) != 0;
    const auto & rt_mma_dev = ggml_cuda_info().devices[ctx.device];
    // nt32: b_nt == 32 rides the SAME sibling structure - u/out batch grids
    // are z-general, the MMA spines run per 16-token half. ntlo: b_nt in
    // (4, 16) rides it from zero-padded scratch rows (b_nt <= 4 stays on the
    // tuned TT-batch/fork regions). b_nt == 16 keeps its exact admission and
    // launch shapes.
    // nt32/ntlow: other admitted widths ride the SAME sibling structure - the
    // u/out batch grids are z-general and the MMA spines run per token chunk.
    // b_nt == 16 keeps its exact admission and launch shapes.
    const bool bnt_mma_nt = b_nt == 16 || (mach1_nt32_on() && b_nt == 32) ||
                            (mach1_ntlo_on() && b_nt >= mach1_ntlo_min() && b_nt < 16) ||
                            mach1_nt_ok(b_nt);
    bool bnt_mma16 = allow_mma16_pair && mma16_pair_on && rt_mma16_on && zf == nullptr &&
                     bnt_mma_nt && nops == 2 && n == 2048 && tc_htab != nullptr &&
                     ampere_mma_available(rt_mma_dev.cc) && rt_mma_dev.warp_size == 32 &&
                     !mach1_ablate(MACH1_ABLATE_RT_WALK);
    for (int i = 0; i < nops; ++i) {
        bnt_mma16 = bnt_mma16 && (m[i] == 4096 || m[i] == 8192) &&
                      mach1_tc_fwht_dim(m[i]);
    }
    static const bool p16_sibling_on = mach1_env_int("GGML_MACH1_P16_SIBLING_BATCH", 0) != 0;
    bool bnt_mixed16 = allow_mma16_trio && p16_sibling_on && rt_mma16_on && zf == nullptr &&
                       bnt_mma_nt && nops == 3 && n == 2048 &&
                       m[0] == 8192 && m[1] == 512 && m[2] == 512 &&
                       tc_htab != nullptr && ampere_mma_available(rt_mma_dev.cc) &&
                       rt_mma_dev.warp_size == 32 && !mach1_ablate(MACH1_ABLATE_RT_WALK);
    // An nt=16 matcher admission is specific to this path. If a future model
    // misses its exact shape/device contract, leave the ordinary per-op graph
    // untouched instead of silently falling back to the older TT batch path.
    if (allow_mma16_pair && mma16_pair_on && bnt_mma_nt && !bnt_mma16) {
        return 0;
    }
    if (allow_mma16_trio && p16_sibling_on && bnt_mma_nt && !bnt_mixed16) {
        return 0;
    }
    // The sibling schedule must preserve the standalone spine selection. If
    // RT_IMMA8 is enabled, batch U writes the identical q8+K16-scale records
    // and the compressed projections call the same cp-aware IMMA8 host path;
    // otherwise this remains the validated fp16 sibling implementation.
    static const bool rt_imma8_on = mach1_env_int("GGML_MACH1_RT_IMMA8", 0) != 0;
    mach1_rt_ztab bnt_imma8_zt = zt;
    if (rt_imma8_on && (bnt_mma16 || bnt_mixed16) && bnt_imma8_zt.ztab == nullptr) {
        bnt_imma8_zt = mach1_rt_z_table(rt[0]->src[3]->data, stream);
    }
    const bool bnt_imma8 = rt_imma8_on && bnt_imma8_zt.ztab != nullptr &&
                           (bnt_mma16 || bnt_mixed16);
    // OSPLIT: every op takes a precomputed u (one batched TC launch), the walk
    // carries no fold, the out stages run as one batched TC launch after it.
    // b_nt >= 2 ALWAYS takes this structure with the walks routed through the
    // TT form: pair-merged u/out launches shave (nops-1) serial stages each,
    // and decode-once walks avoid the r275 per-slice re-decode loss. Nothing
    // is replicated per token across a grid (the r283 rule).
    const bool bnt_split = !bnt_mma16 && !bnt_mixed16 && b_nt >= 2 &&
                           tc_htab != nullptr && n % 16 == 0 &&
                           n/16 <= 256 && mach1_tc_fwht_dim(n) && qprobe == 0;
    bool osplit = (osplit_lvl > 0 && zdp && qprobe == 0 && tc_htab != nullptr &&
                   n/16 <= 128 && mach1_tc_fwht_dim(n) && b_nt == 1) ||
                  bnt_split || bnt_mma16 || bnt_mixed16;
    for (int i = 0; i < nops; ++i) {
        osplit = osplit && mach1_tc_fwht_dim(m[i]);
    }
    if (bnt_split && !osplit) {
        return 0;   // a shape outside the TC dims: leave it to the unfused path
    }
    if (glu_out != nullptr &&
        (!p16_sibling_on || !bnt_split || !bnt_mma_nt || nops != 2 ||
         n != 2048 || m[0] != 512 || m[1] != 512)) {
        return 0;
    }
    // ntlo pad: the 16-token MMA spines read scratch token rows [b_nt, 16) -
    // carve the regions at capacity 16 and zero them so pad rows decode to
    // zero in every record form. All consumers stay token-major, so only the
    // capacity changes.
    const size_t bcap = (bnt_mma16 || bnt_mixed16) && b_nt < 16 ? 16 : (size_t) b_nt;
    for (int i = 0; i < nops; ++i) {
        ufuse[i] = !osplit && ufuse_maxb > 0 && m[i]/16 <= ufuse_maxb;
        scr_elems += ((size_t) m[i] + (ufuse[i] ? 0 : (size_t) n))*bcap;
        tcw = tcw && !osplit && mach1_tc_fwht_dim(m[i]) && (!ufuse[i] || mach1_tc_fwht_dim(n));
    }
    // FORK_P16 (gfork): the whole region rides s2, so its scratch must be
    // private - the pool hands a freed block back to main-stream work that
    // is still running concurrently. Only the exact p16 gate/up shape forks;
    // anything else falls back to the ordinary main-stream schedule.
    if (gfork != nullptr && !(bnt_split && glu_out != nullptr && gfork->scrn >= scr_elems)) {
        gfork = nullptr;
    }
    ggml_cuda_pool_alloc<float> scr(ctx.pool());
    cudaStream_t xs = gfork != nullptr ? gfork->s2 : stream;
    if (gfork != nullptr) {
        CUDA_CHECK(cudaStreamWaitEvent(gfork->s2, gfork->ex, 0));
        mach1_stamp(gfork->s2, stream, 3);
        if (mach1_debug_on()) {
            static std::once_flag once;
            std::call_once(once, []() {
                fprintf(stderr, "mach1: p16 shared gate/up FORK ENGAGED (s2 under the expert mega)\n");
            });
        }
    }
    const auto gu_join = [&]() {
        if (gfork != nullptr) {
            mach1_stamp(gfork->s2, stream, 4);
            CUDA_CHECK(cudaEventRecord(gfork->egu, gfork->s2));
            CUDA_CHECK(cudaStreamWaitEvent(stream, gfork->egu, 0));
        }
    };

    char shp[96];
    snprintf(shp, sizeof(shp), " m=%d/%d/%d n=%d nt=%d", m[0], m[1], m[2], n, b_nt);

    mach1_rt_batch_op ops[3] = {};
    float * p = gfork != nullptr ? gfork->scr : scr.alloc(scr_elems);
    if (bcap != (size_t) b_nt) {
        CUDA_CHECK(cudaMemsetAsync(p, 0, scr_elems*sizeof(float), xs));
    }
    int blk = 0, mmax = 0;
    size_t smem = 0;
    for (int i = 0; i < nops; ++i) {
        ops[i].trellis = (const uint16_t *) rt[i]->src[0]->data;
        ops[i].su      = (const float    *) rt[i]->src[1]->data;
        ops[i].sv      = (const float    *) rt[i]->src[2]->data;
        ops[i].scr_v   = p;
        ops[i].dst     = (float *) out[i]->data;
        ops[i].perm    = perm[i];
        ops[i].m       = m[i];
        ops[i].blk0    = blk;
        p    += (size_t) m[i]*bcap;
        blk  += m[i]/16;
        mmax  = std::max(mmax, m[i]);
        if (tcw) {
            smem = std::max(smem, mach1_tc_ws_bytes(m[i]));
            if (ufuse[i]) {
                smem = std::max(smem, mach1_tc_ws_bytes(n));
            }
        }
        if (!ufuse[i]) {
            float * up = p;
            ops[i].scr_u = up;
            p += (size_t) n*bcap;
            if (glu_out != nullptr) {
                // In the stock graph gate dies before up is produced, so the
                // allocator may alias their tensor buffers. They are live
                // together in this batched schedule: after both TT walks,
                // reuse each op's now-dead U scratch as its transformed output
                // rather than extending graph-buffer lifetimes or allocating
                // another region. n >= m for this exact shared-expert shape.
                ops[i].dst = up;
            }
            if (osplit) {
                continue;   // one batched u launch after the loop
            }
            if (tc_htab != nullptr && mach1_tc_fwht_dim(n)) {
                mach1_timed(stream, std::string(key) + "_u_tc" + shp, [&]() {
                mach1_rt_u_tc(tc_htab, ops[i].su, (const float *) rt[i]->src[4]->data, up, n, stream, b_nt);
                });
                continue;
            }
            mach1_timed(stream, std::string(key) + "_u" + shp, [&]() {
            if (mach1_fwht_wg() == 1024) {
                mach1_launch(mach1_rt_u_kernel<1024>,
                    ggml_cuda_kernel_launch_params(dim3(b_nt, 1, 1), dim3(1024, 1, 1), 0, stream),
                    ops[i].su, (const float *) rt[i]->src[4]->data, up, n);
            } else if (mach1_fwht_wg512()) {
                mach1_launch(mach1_rt_u_kernel<512>,
                    ggml_cuda_kernel_launch_params(dim3(b_nt, 1, 1), dim3(512, 1, 1), 0, stream),
                    ops[i].su, (const float *) rt[i]->src[4]->data, up, n);
            } else {
                mach1_launch(mach1_rt_u_kernel<256>,
                    ggml_cuda_kernel_launch_params(dim3(b_nt, 1, 1), dim3(256, 1, 1), 0, stream),
                    ops[i].su, (const float *) rt[i]->src[4]->data, up, n);
            }
            });
        }
    }
    if (!tcw) {
        smem = (size_t) mmax*sizeof(float);
    }

    // walk WG mirrors the unfused dispatch: 512 up to 128 column tiles, 1024
    // above; n is shared across the ops (same x)
    const int wg = n/16 <= 128 ? 512 : 1024;
    const ggml_cuda_kernel_launch_params vp =
        ggml_cuda_kernel_launch_params(dim3(blk, 1, b_nt), dim3(wg, 1, 1), smem, stream);
    const half  * tl_v = (const half  *) rt[0]->src[3]->data;
    const float * x_v  = (const float *) rt[0]->src[4]->data;

    if (osplit) {
        mach1_rt_u_batch ub[3] = {};
        for (int i = 0; i < nops; ++i) {
            ub[i].su    = ops[i].su;
            ub[i].scr_u = (float *) ops[i].scr_u;
        }
        if (zf != nullptr) {
            // the z fork's walk reads u from private scratch only - the pool
            // may hand ops[1].scr_u's bytes to later main-stream work while
            // s2 still reads them
            ub[1].scr_u = zf->zscr + (size_t) m[1]*b_nt;
        }
        mach1_timed(xs, std::string(key) + "_u_tcb" + shp, [&]() {
        if (bnt_mma16 && bnt_imma8) {
            mach1_launch(mach1_rt_u_tc_batch_kernel<512, false, true>,
                ggml_cuda_kernel_launch_params(dim3(nops, 1, b_nt), dim3(512, 1, 1), mach1_tc_ws_bytes(n), xs),
                ub[0].su, ub[0].scr_u, ub[1].su, ub[1].scr_u, ub[2].su, ub[2].scr_u,
                tc_htab, x_v, n, bnt_imma8_zt.zstep);
        } else if (bnt_mma16) {
            mach1_launch(mach1_rt_u_tc_batch_kernel<512, true>,
                ggml_cuda_kernel_launch_params(dim3(nops, 1, b_nt), dim3(512, 1, 1), mach1_tc_ws_bytes(n), xs),
                ub[0].su, ub[0].scr_u, ub[1].su, ub[1].scr_u, ub[2].su, ub[2].scr_u,
                tc_htab, x_v, n, 0.0f);
        } else if (bnt_mixed16 && bnt_imma8) {
            mach1_launch(mach1_rt_u_tc_mixed_p16_kernel<512, true>,
                ggml_cuda_kernel_launch_params(dim3(3, 1, b_nt), dim3(512, 1, 1), mach1_tc_ws_bytes(2048), xs),
                ub[0], ub[1], ub[2], tc_htab, x_v, bnt_imma8_zt.zstep);
        } else if (bnt_mixed16) {
            mach1_launch(mach1_rt_u_tc_mixed_p16_kernel<512>,
                ggml_cuda_kernel_launch_params(dim3(3, 1, b_nt), dim3(512, 1, 1), mach1_tc_ws_bytes(2048), xs),
                ub[0], ub[1], ub[2], tc_htab, x_v, 0.0f);
        } else {
            mach1_launch(mach1_rt_u_tc_batch_kernel<512>,
                ggml_cuda_kernel_launch_params(dim3(nops, 1, b_nt), dim3(512, 1, 1), mach1_tc_ws_bytes(n), xs),
                ub[0].su, ub[0].scr_u, ub[1].su, ub[1].scr_u, ub[2].su, ub[2].scr_u,
                tc_htab, x_v, n, 0.0f);
        }
        });
    }

    if (bnt_mma16) {
        mach1_timed(stream, std::string(key) + (bnt_imma8 ? "_imma8" : "_mma16") + shp, [&]() {
        for (int i = 0; i < nops; ++i) {
            // b_nt == 16: one chunk at zero offset (the certified launch);
            // any other admitted width: the same spine per token chunk.
            if (bnt_imma8) {
                mach1_rt_spine_imma8_p16(ops[i].trellis, bnt_imma8_zt.ztab,
                    ops[i].scr_u, ops[i].scr_v, m[i], n, stream, false, b_nt);
            } else {
                mach1_rt_spine_mma_p16(ops[i].trellis, tl_v,
                    (const half *) ops[i].scr_u, ops[i].scr_v, m[i], n, stream, b_nt);
            }
        }
        });
        if (mach1_rt_stdout_on()) {
            int rsb_m = 0;
            for (int i = 0; i < nops; ++i) {
                rsb_m = std::max(rsb_m, m[i]);
            }
            const unsigned rsb_y = (unsigned) ((rsb_m + MACH1_RT_RSB_CHUNK - 1)/MACH1_RT_RSB_CHUNK);
            mach1_timed(stream, std::string(key) + "_out_rsb_mma16" + shp, [&]() {
            mach1_launch(mach1_rt_out_rs_batch_kernel,
                ggml_cuda_kernel_launch_params(dim3(nops, rsb_y, b_nt), dim3(256, 1, 1), 0, stream),
                ops[0], ops[1], ops[2]);
            });
        } else {
        size_t osmem = 0;
        for (int i = 0; i < nops; ++i) {
            osmem = std::max(osmem, mach1_tc_ws_bytes(m[i]));
        }
        int omax = 0;
        for (int i = 0; i < nops; ++i) {
            omax = std::max(omax, m[i]);
        }
        mach1_timed(stream, std::string(key) + "_out_tcb_mma16" + shp, [&]() {
        mach1_rt_out_tcb_launch(stream, dim3(nops, 1, b_nt), osmem,
                                ops[0], ops[1], ops[2], tc_htab, omax);
        });
        }
        if (mach1_debug_on()) {
            static std::once_flag once;
            std::call_once(once, [&]() {
                fprintf(stderr, "mach1: p16 qkvz sibling transforms + %s ENGAGED\n",
                        bnt_imma8 ? "rt_imma8" : "rt_mma16");
            });
        }
        if (mach1_debug_on() && b_nt < 16) {
            static std::once_flag once_lo;
            std::call_once(once_lo, [&]() {
                fprintf(stderr, "mach1: ntlo sibling pair nt=%d ENGAGED\n", b_nt);
            });
        }
        return skip;
    }

    if (bnt_mixed16) {
        mach1_timed(stream, std::string(key) +
                    (bnt_imma8 ? "_mixed_imma8" : "_mixed_mma16") + shp, [&]() {
        if (bnt_imma8) {
            mach1_rt_spine_imma8_p16(ops[0].trellis, bnt_imma8_zt.ztab,
                ops[0].scr_u, ops[0].scr_v, m[0], n, stream, false, b_nt);
        } else {
            mach1_rt_spine_mma_p16(ops[0].trellis, tl_v,
                (const half *) ops[0].scr_u, ops[0].scr_v, m[0], n, stream, b_nt);
        }
        mach1_rt_batch_op kv[3] = { ops[1], ops[2], {} };
        kv[0].blk0 = 0;
        kv[1].blk0 = m[1]/16;
        mach1_rt_walk_tt_launch(stream, kv, tl_v, (m[1] + m[2])/16, n, b_nt, zt);
        });
        if (mach1_rt_stdout_on()) {
            const unsigned rsb_y = (unsigned) ((m[0] + MACH1_RT_RSB_CHUNK - 1)/MACH1_RT_RSB_CHUNK);
            mach1_timed(stream, std::string(key) + "_out_rsb_mixed16" + shp, [&]() {
            mach1_launch(mach1_rt_out_rs_batch_kernel,
                ggml_cuda_kernel_launch_params(dim3(3, rsb_y, b_nt), dim3(256, 1, 1), 0, stream),
                ops[0], ops[1], ops[2]);
            });
        } else {
        const size_t osmem = mach1_tc_ws_bytes(m[0]);
        mach1_timed(stream, std::string(key) + "_out_tcb_mixed16" + shp, [&]() {
        mach1_rt_out_tcb_launch(stream, dim3(3, 1, b_nt), osmem,
                                ops[0], ops[1], ops[2], tc_htab, m[0]);
        });
        }
        if (mach1_debug_on()) {
            static std::once_flag once;
            std::call_once(once, [&]() {
                fprintf(stderr, "mach1: p16 attention qkv sibling batch + %s ENGAGED\n",
                        bnt_imma8 ? "rt_imma8" : "rt_mma16");
            });
        }
        if (mach1_debug_on() && b_nt < 16) {
            static std::once_flag once_lo;
            std::call_once(once_lo, [&]() {
                fprintf(stderr, "mach1: ntlo sibling trio nt=%d ENGAGED\n", b_nt);
            });
        }
        return skip;
    }

    if (bnt_split && zf != nullptr) {
        // FORK_Z at nt: the z op (ops[1]) walks and folds on s2 under the
        // conv/core chain ahead, which never reads it; main runs the qkv op
        // alone. s2 reads only private scratch (u_tcb wrote z's u there) and
        // immortal tables; the z dst bytes are allocator-reserved until the
        // post-join consumer. Schedule only - values identical.
        CUDA_CHECK(cudaEventRecord(zf->ezx, stream));
        CUDA_CHECK(cudaStreamWaitEvent(zf->s2, zf->ezx, 0));
        mach1_rt_batch_op zop = ops[1];
        zop.scr_u = zf->zscr + (size_t) m[1]*b_nt;
        zop.scr_v = zf->zscr;
        zop.blk0  = 0;
        mach1_rt_batch_op zsolo[3] = { zop, {}, {} };
        mach1_timed(zf->s2, std::string(key) + "_zwalk_tt" + shp, [&]() {
        mach1_rt_walk_tt_launch(zf->s2, zsolo, tl_v, m[1]/16, n, b_nt, zt);
        });
        if (mach1_rt_stdout_on()) {
            const unsigned rsb_y = (unsigned) ((m[1] + MACH1_RT_RSB_CHUNK - 1)/MACH1_RT_RSB_CHUNK);
            mach1_timed(zf->s2, std::string(key) + "_zout_rsb" + shp, [&]() {
            mach1_launch(mach1_rt_out_rs_batch_kernel,
                ggml_cuda_kernel_launch_params(dim3(1, rsb_y, b_nt), dim3(256, 1, 1), 0, zf->s2),
                zsolo[0], zsolo[1], zsolo[2]);
            });
        } else {
        const size_t zsmem = mach1_tc_ws_bytes(m[1]);
        mach1_smem_opt_in((const void *) mach1_rt_out_tc_batch_kernel<512>, zsmem);
        mach1_timed(zf->s2, std::string(key) + "_zout_tcb" + shp, [&]() {
        mach1_launch(mach1_rt_out_tc_batch_kernel<512>,
            ggml_cuda_kernel_launch_params(dim3(1, 1, b_nt), dim3(512, 1, 1), zsmem, zf->s2),
            zsolo[0], zsolo[1], zsolo[2], tc_htab);
        });
        }
        CUDA_CHECK(cudaEventRecord(zf->ez, zf->s2));
        zf->z_armed = true;

        mach1_rt_batch_op qsolo[3] = { ops[0], {}, {} };
        mach1_timed(stream, std::string(key) + "_walk_tt" + shp, [&]() {
        mach1_rt_walk_tt_launch(stream, qsolo, tl_v, m[0]/16, n, b_nt, zt);
        });
        if (mach1_rt_stdout_on()) {
            const unsigned rsb_y = (unsigned) ((m[0] + MACH1_RT_RSB_CHUNK - 1)/MACH1_RT_RSB_CHUNK);
            mach1_timed(stream, std::string(key) + "_out_rsb" + shp, [&]() {
            mach1_launch(mach1_rt_out_rs_batch_kernel,
                ggml_cuda_kernel_launch_params(dim3(1, rsb_y, b_nt), dim3(256, 1, 1), 0, stream),
                qsolo[0], qsolo[1], qsolo[2]);
            });
        } else {
        const size_t qsmem = mach1_tc_ws_bytes(m[0]);
        mach1_smem_opt_in((const void *) mach1_rt_out_tc_batch_kernel<512>, qsmem);
        mach1_timed(stream, std::string(key) + "_out_tcb" + shp, [&]() {
        mach1_launch(mach1_rt_out_tc_batch_kernel<512>,
            ggml_cuda_kernel_launch_params(dim3(1, 1, b_nt), dim3(512, 1, 1), qsmem, stream),
            qsolo[0], qsolo[1], qsolo[2], tc_htab);
        });
        }
        return skip;
    }

    if (bnt_split) {
        // decode-once walks - the sliced batch walk re-decodes per token
        // (the r275 loss) and stays b_nt == 1. WMERGE runs the whole region
        // as ONE pair-select grid; default keeps the raced per-op launches.
        static const bool wmerge = mach1_env_int("GGML_MACH1_QKVB_WMERGE", 0) != 0;
        mach1_timed(xs, std::string(key) + "_walk_tt" + shp, [&]() {
        if (wmerge) {
            mach1_rt_walk_tt_launch(xs, ops, tl_v, blk, n, b_nt, zt);
        } else {
            for (int i = 0; i < nops; ++i) {
                mach1_rt_batch_op solo[3] = { ops[i], {}, {} };
                solo[0].blk0 = 0;
                mach1_rt_walk_tt_launch(xs, solo, tl_v, m[i]/16, n, b_nt, zt);
            }
        }
        });
        if (mach1_rt_stdout_on()) {
            int rsb_m = 0;
            for (int i = 0; i < nops; ++i) {
                rsb_m = std::max(rsb_m, m[i]);
            }
            const unsigned rsb_y = (unsigned) ((rsb_m + MACH1_RT_RSB_CHUNK - 1)/MACH1_RT_RSB_CHUNK);
            mach1_timed(xs, std::string(key) + "_out_rsb" + shp, [&]() {
            mach1_launch(mach1_rt_out_rs_batch_kernel,
                ggml_cuda_kernel_launch_params(dim3(nops, rsb_y, b_nt), dim3(256, 1, 1), 0, xs),
                ops[0], ops[1], ops[2]);
            });
        } else {
        static const bool shexp_oglu = mach1_env_int("GGML_MACH1_SHEXP_OGLU", 0) != 0;
        if (glu_out != nullptr && shexp_oglu && nops == 2 && m[0] == m[1] &&
            tc_htab != nullptr && mach1_tc_fwht_dim(m[0])) {
            const size_t fsmem = 2*(size_t) m[0]*sizeof(float) + mach1_tc_ws_bytes(m[0]);
            mach1_smem_opt_in((const void *) mach1_shexp_out_glu_tc_kernel<512>, fsmem);
            mach1_timed(xs, std::string(key) + "_oglu_tcb" + shp, [&]() {
            mach1_launch(mach1_shexp_out_glu_tc_kernel<512>,
                ggml_cuda_kernel_launch_params(dim3(1, 1, b_nt), dim3(512, 1, 1), fsmem, xs),
                ops[0], ops[1], tc_htab, (float *) glu_out->data);
            });
            if (mach1_debug_on()) {
                static std::once_flag once;
                std::call_once(once, [&]() {
                    fprintf(stderr, "mach1: p16 shared gate/up fused out+glu ENGAGED\n");
                });
            }
            gu_join();
            return skip;
        }
        size_t osmem = 0;
        for (int i = 0; i < nops; ++i) {
            osmem = std::max(osmem, mach1_tc_ws_bytes(m[i]));
        }
        mach1_smem_opt_in((const void *) mach1_rt_out_tc_batch_kernel<512>, osmem);
        mach1_timed(xs, std::string(key) + "_out_tcb" + shp, [&]() {
        mach1_launch(mach1_rt_out_tc_batch_kernel<512>,
            ggml_cuda_kernel_launch_params(dim3(nops, 1, b_nt), dim3(512, 1, 1), osmem, xs),
            ops[0], ops[1], ops[2], tc_htab);
        });
        }
        if (glu_out != nullptr) {
            const int64_t ne = ggml_nelements(glu_out);
            mach1_timed(xs, std::string(key) + "_silu_mul" + shp, [&]() {
            mach1_launch(mach1_silu_mul_p16_kernel,
                ggml_cuda_kernel_launch_params(dim3((unsigned) ((ne + 255)/256), 1, 1),
                                               dim3(256, 1, 1), 0, xs),
                ops[0].dst, ops[1].dst, (float *) glu_out->data, ne);
            });
            if (mach1_debug_on()) {
                static std::once_flag once;
                std::call_once(once, [&]() {
                    fprintf(stderr, "mach1: p16 shared gate/up sibling batch ENGAGED\n");
                });
            }
        }
        gu_join();
        return skip;
    }

    // chunk fold: no dynamic shared (folds through shu); TC fold: vp's smem
    const ggml_cuda_kernel_launch_params vpz =
        ggml_cuda_kernel_launch_params(dim3(blk, 1, b_nt), dim3(wg, 1, 1), 0, stream);
    mach1_timed(stream, std::string(key) + (tcw && !zdp ? "_walk_tc" : "_walk") + shp, [&]() {
    if (zdp && osplit && osplit_lvl >= 2) {
        // level 2: the walk pinned at 2 blocks/SM - twice the waves in flight
        mach1_launch(mach1_rt_qkv_batch_kernel<512, 0, 1, 0, 2>, vpz, ops[0], ops[1], ops[2], tl_v, x_v, cnt, n, (const half *) nullptr, zt.ztab, zt.zstep, 1);
    } else if (zdp && osplit) {
        mach1_launch(mach1_rt_qkv_batch_kernel<512, 0, 1>, vpz, ops[0], ops[1], ops[2], tl_v, x_v, cnt, n, (const half *) nullptr, zt.ztab, zt.zstep, 1);
    } else if (zdp && wg == 512 && qprobe == 1 && b_nt == 1) {
        mach1_launch(mach1_rt_qkv_batch_kernel<512, 0, 1, 1>, vpz, ops[0], ops[1], ops[2], tl_v, x_v, cnt, n, (const half *) nullptr, zt.ztab, zt.zstep, 0);
    } else if (zdp && wg == 512 && qprobe == 2 && b_nt == 1) {
        mach1_launch(mach1_rt_qkv_batch_kernel<512, 0, 1, 2>, vpz, ops[0], ops[1], ops[2], tl_v, x_v, cnt, n, (const half *) nullptr, zt.ztab, zt.zstep, 0);
    } else if (zdp && wg == 512 && qprobe == 3 && b_nt == 1) {
        mach1_launch(mach1_rt_qkv_batch_kernel<512, 0, 1, 3>, vpz, ops[0], ops[1], ops[2], tl_v, x_v, cnt, n, (const half *) nullptr, zt.ztab, zt.zstep, 0);
    } else if (zdp && tcw && !qchunk) {
        mach1_smem_opt_in((const void *) mach1_rt_qkv_batch_kernel<512, 1, 1>, smem);
        mach1_launch(mach1_rt_qkv_batch_kernel<512, 1, 1>, vp, ops[0], ops[1], ops[2], tl_v, x_v, cnt, n, tc_htab, zt.ztab, zt.zstep, 0);
    } else if (zdp && wg == 512) {
        mach1_launch(mach1_rt_qkv_batch_kernel<512, 0, 1>, vpz, ops[0], ops[1], ops[2], tl_v, x_v, cnt, n, (const half *) nullptr, zt.ztab, zt.zstep, 0);
    } else if (zdp) {
        mach1_launch(mach1_rt_qkv_batch_kernel<1024, 0, 1>, vpz, ops[0], ops[1], ops[2], tl_v, x_v, cnt, n, (const half *) nullptr, zt.ztab, zt.zstep, 0);
    } else if (tcw) {
        mach1_smem_opt_in((const void *) mach1_rt_qkv_batch_kernel<512, 1>, smem);
        mach1_launch(mach1_rt_qkv_batch_kernel<512, 1>, vp, ops[0], ops[1], ops[2], tl_v, x_v, cnt, n, tc_htab,
                     (const uint16_t *) nullptr, 0.0f, 0);
    } else if (wg == 512) {
        mach1_launch(mach1_rt_qkv_batch_kernel<512>, vp, ops[0], ops[1], ops[2], tl_v, x_v, cnt, n, (const half *) nullptr,
                     (const uint16_t *) nullptr, 0.0f, 0);
    } else {
        mach1_launch(mach1_rt_qkv_batch_kernel<1024>, vp, ops[0], ops[1], ops[2], tl_v, x_v, cnt, n, (const half *) nullptr,
                     (const uint16_t *) nullptr, 0.0f, 0);
    }
    });

    if (osplit) {
        if (mach1_rt_stdout_on()) {
            int rsb_m = 0;
            for (int i = 0; i < nops; ++i) {
                rsb_m = std::max(rsb_m, m[i]);
            }
            const unsigned rsb_y = (unsigned) ((rsb_m + MACH1_RT_RSB_CHUNK - 1)/MACH1_RT_RSB_CHUNK);
            mach1_timed(stream, std::string(key) + "_out_rsb" + shp, [&]() {
            mach1_launch(mach1_rt_out_rs_batch_kernel,
                ggml_cuda_kernel_launch_params(dim3(nops, rsb_y, 1), dim3(256, 1, 1), 0, stream),
                ops[0], ops[1], ops[2]);
            });
        } else {
        size_t osmem = 0;
        for (int i = 0; i < nops; ++i) {
            osmem = std::max(osmem, mach1_tc_ws_bytes(m[i]));
        }
        mach1_smem_opt_in((const void *) mach1_rt_out_tc_batch_kernel<512>, osmem);
        mach1_timed(stream, std::string(key) + "_out_tcb" + shp, [&]() {
        mach1_launch(mach1_rt_out_tc_batch_kernel<512>,
            ggml_cuda_kernel_launch_params(dim3(nops, 1, 1), dim3(512, 1, 1), osmem, stream),
            ops[0], ops[1], ops[2], tc_htab);
        });
        }
    }

    return skip;
}

static int mach1_shexp_gu_batch_p16(ggml_backend_cuda_context & ctx,
                                    ggml_tensor * gp, ggml_tensor * up,
                                    ggml_tensor * par, int skip,
                                    mach1_fork_state * gfork) {
    const ggml_tensor * const rt[3] = { gp, up, nullptr };
    ggml_tensor       * const outs[3] = { gp, up, nullptr };
    const int         * const perms[3] = { nullptr, nullptr, nullptr };
    return mach1_rt_batch_exec(ctx, "shexp_gub16", rt, outs, perms, 2, skip,
                               nullptr, false, false, par, gfork);
}

static bool mach1_qkv_batch_on() {
    static const bool on = mach1_env_int("GGML_MACH1_QKV_BATCH", 1) != 0;
    return on;
}

// v_tiled fold (GGML_MACH1_VTILED=1; decode only): the GDN layers emit a
// grouped->tiled V-row reorder after the wqkv and wqkv_gate rt ops
// (build_qkvz in src/models/mach1.cpp). Two glue patterns:
//   qkv: [i] MACH1_RT_MM, VIEW qk, CONT, VIEW v, CONT,
//        RESHAPE [hd,r,K,T], PERMUTE (0,2,1,3), CONT, RESHAPE, CONCAT
//   z:   [i] MACH1_RT_MM, RESHAPE [hd,r,K,T], PERMUTE (0,2,1,3), CONT
// Values are only MOVED, never changed, so the whole region collapses to the
// rt walk with the OFUSE epilogue writing dst[perm[i]] instead of dst[i]
// into the region's final tensor - the qk rows keep their index, the V rows
// scatter per mach1_vtiled_perm. The rt_u launch stays when the shape is
// outside the UFUSE window, exactly as in ggml_cuda_op_mach1_rt_mm.

// FORK_Z probe, defined with the gdn_full machinery below
static bool mach1_gdn_full_ahead(const ggml_cgraph * cgraph, int from);

// the reshape/permute/cont triple v_tiled expands to; reads off (hd, K, r).
// Also matches the INVERSE (tiled->grouped) reorder ahead of ssm_out - the
// triple has the same structure, only the middle extents swap roles.
static bool mach1_match_tiled(const ggml_tensor * rsh, const ggml_tensor * prm, const ggml_tensor * c,
                              const ggml_tensor * src, int64_t rows, int * hd, int * K, int * r,
                              int64_t nt = 1) {
    if (rsh->op != GGML_OP_RESHAPE || prm->op != GGML_OP_PERMUTE || c->op != GGML_OP_CONT) {
        return false;
    }
    if (rsh->src[0] != src || prm->src[0] != rsh || c->src[0] != prm) {
        return false;
    }
    // nt > 1: the token axis rides ne[3] through the (0,2,1,3) permute, so
    // the reorder still acts within each token's row block and the per-token
    // scatter map is unchanged
    if (rsh->ne[3] != nt || rsh->ne[0]*rsh->ne[1]*rsh->ne[2] != rows) {
        return false;
    }
    // exactly ggml_permute(t, 0, 2, 1, 3): axes 1 and 2 swap in ne AND nb
    if (prm->ne[0] != rsh->ne[0] || prm->ne[1] != rsh->ne[2] ||
        prm->ne[2] != rsh->ne[1] || prm->ne[3] != rsh->ne[3]) {
        return false;
    }
    if (prm->nb[0] != rsh->nb[0] || prm->nb[1] != rsh->nb[2] ||
        prm->nb[2] != rsh->nb[1] || prm->nb[3] != rsh->nb[3]) {
        return false;
    }
    if (c->type != GGML_TYPE_F32 || !ggml_is_contiguous(c) || !ggml_are_same_shape(c, prm)) {
        return false;
    }
    *hd = (int) rsh->ne[0];
    *r  = (int) rsh->ne[1];
    *K  = (int) rsh->ne[2];
    return true;
}

static int ggml_cuda_mach1_vtiled_exec(ggml_backend_cuda_context & ctx, const ggml_tensor * rt,
                                       ggml_tensor * out, int qk_off, int hd, int K, int r, int skip,
                                       cudaStream_t sover = nullptr, float * scrover = nullptr,
                                       int * cntover = nullptr) {
    const int n = (int) rt->src[1]->ne[0];
    const int m = (int) rt->src[2]->ne[0];

    // FORK_Z: sover/scrover/cntover move the whole op to the fork stream with
    // private scratch and counter; everything else is identical
    cudaStream_t stream = sover != nullptr ? sover : ctx.stream();
    int * cnt = cntover != nullptr ? cntover : mach1_stream_counter(ctx.device, ctx.stream());
    if (cnt == nullptr) {
        return 0;
    }
    const int * perm = mach1_vtiled_perm(ctx.device, stream, m, qk_off, hd, K, r);
    if (perm == nullptr) {
        return 0;
    }

    ggml_cuda_pool_alloc<float> scr(ctx.pool());
    float * scr_u;
    if (scrover != nullptr) {
        scr_u = scrover;
    } else {
        scr_u = scr.alloc((size_t) n + (size_t) m);
    }
    float * scr_v = scr_u + (size_t) n;

    // mirrors the unfused dispatch: rt_u only outside the UFUSE window, walk
    // WG 512 up to 128 column tiles, 1024 above
    static const int ufuse_maxb = mach1_env_int("GGML_MACH1_UFUSE_MAXB", 256);
    const bool ufuse = ufuse_maxb > 0 && m/16 <= ufuse_maxb;
    // wg 512 walks only - the 1024-thread TCF form exceeds the register
    // budget, see the dispatch comment in ggml_cuda_op_mach1_rt_mm
    const half * tc_htab = mach1_tc_fwht_tab(ctx.device, stream);
    const bool tcw = tc_htab != nullptr && n/16 <= 128 && mach1_tc_fwht_dim(m) &&
                     (!ufuse || mach1_tc_fwht_dim(n));

    char shp[64];
    snprintf(shp, sizeof(shp), " m=%d n=%d qk=%d", m, n, qk_off);

    if (!ufuse) {
        if (tc_htab != nullptr && mach1_tc_fwht_dim(n)) {
            mach1_timed(stream, std::string("vtiled_u_tc") + shp, [&]() {
            mach1_rt_u_tc(tc_htab, (const float *) rt->src[1]->data,
                          (const float *) rt->src[4]->data, scr_u, n, stream);
            });
        } else
        mach1_timed(stream, std::string("vtiled_u") + shp, [&]() {
        if (mach1_fwht_wg() == 1024) {
            mach1_launch(mach1_rt_u_kernel<1024>,
                ggml_cuda_kernel_launch_params(dim3(1, 1, 1), dim3(1024, 1, 1), 0, stream),
                (const float *) rt->src[1]->data, (const float *) rt->src[4]->data, scr_u, n);
        } else if (mach1_fwht_wg512()) {
            mach1_launch(mach1_rt_u_kernel<512>,
                ggml_cuda_kernel_launch_params(dim3(1, 1, 1), dim3(512, 1, 1), 0, stream),
                (const float *) rt->src[1]->data, (const float *) rt->src[4]->data, scr_u, n);
        } else {
            mach1_launch(mach1_rt_u_kernel<256>,
                ggml_cuda_kernel_launch_params(dim3(1, 1, 1), dim3(256, 1, 1), 0, stream),
                (const float *) rt->src[1]->data, (const float *) rt->src[4]->data, scr_u, n);
        }
        });
    }

    const int wg = n/16 <= 128 ? 512 : 1024;
    size_t smem = (size_t) m*sizeof(float);
    if (tcw) {
        smem = mach1_tc_ws_bytes(m);
        if (ufuse) {
            smem = std::max(smem, mach1_tc_ws_bytes(n));
        }
    }
    const ggml_cuda_kernel_launch_params vp =
        ggml_cuda_kernel_launch_params(dim3(m/16, 1, 1), dim3(wg, 1, 1), smem, stream);
    const uint16_t * tre_v = (const uint16_t *) rt->src[0]->data;
    const half     * tl_v  = (const half     *) rt->src[3]->data;
    const float    * su_v  = (const float    *) rt->src[1]->data;
    const float    * sv_v  = (const float    *) rt->src[2]->data;
    const float    * x_v   = (const float    *) rt->src[4]->data;
    float          * dst_v = (float *) out->data;
    mach1_timed(stream, std::string(tcw ? "vtiled_walk_tc" : "vtiled_walk") + shp, [&]() {
    if (tcw && ufuse) {
        mach1_smem_opt_in((const void *) mach1_rt_walk_rows_v_kernel<512, 4, 1, 1, 1>, smem);
        mach1_launch(mach1_rt_walk_rows_v_kernel<512, 4, 1, 1, 1>, vp,
            tre_v, tl_v, scr_u, su_v, x_v, scr_v, m, n, sv_v, dst_v, cnt, perm, tc_htab,
            (const uint16_t *) nullptr, 0.0f, (const int *) nullptr);
    } else if (tcw) {
        mach1_smem_opt_in((const void *) mach1_rt_walk_rows_v_kernel<512, 4, 0, 1, 1>, smem);
        mach1_launch(mach1_rt_walk_rows_v_kernel<512, 4, 0, 1, 1>, vp,
            tre_v, tl_v, scr_u, su_v, x_v, scr_v, m, n, sv_v, dst_v, cnt, perm, tc_htab,
            (const uint16_t *) nullptr, 0.0f, (const int *) nullptr);
    } else if (wg == 512 && ufuse) {
        mach1_launch(mach1_rt_walk_rows_v_kernel<512, 4, 1, 1>, vp,
            tre_v, tl_v, scr_u, su_v, x_v, scr_v, m, n, sv_v, dst_v, cnt, perm, (const half *) nullptr,
            (const uint16_t *) nullptr, 0.0f, (const int *) nullptr);
    } else if (wg == 512) {
        mach1_launch(mach1_rt_walk_rows_v_kernel<512, 4, 0, 1>, vp,
            tre_v, tl_v, scr_u, su_v, x_v, scr_v, m, n, sv_v, dst_v, cnt, perm, (const half *) nullptr,
            (const uint16_t *) nullptr, 0.0f, (const int *) nullptr);
    } else if (ufuse) {
        mach1_launch(mach1_rt_walk_rows_v_kernel<1024, 4, 1, 1>, vp,
            tre_v, tl_v, scr_u, su_v, x_v, scr_v, m, n, sv_v, dst_v, cnt, perm, (const half *) nullptr,
            (const uint16_t *) nullptr, 0.0f, (const int *) nullptr);
    } else {
        mach1_launch(mach1_rt_walk_rows_v_kernel<1024, 4, 0, 1>, vp,
            tre_v, tl_v, scr_u, su_v, x_v, scr_v, m, n, sv_v, dst_v, cnt, perm, (const half *) nullptr,
            (const uint16_t *) nullptr, 0.0f, (const int *) nullptr);
    }
    });

    return skip;
}

int ggml_cuda_mach1_vtiled_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    // fold/zbank/bank/fuse_rt/coop serve rt ops from other numeric paths, and
    // the non-default walk knobs leave the rows_v kernel this fold rides
    static const bool on = mach1_env_int("GGML_MACH1_VTILED", 1) != 0 &&
                           mach1_env_int("GGML_MACH1_ABLATE", 0) == 0 &&
                           mach1_env_int("GGML_MACH1_FUSE_RT", 0) == 0 &&
                           mach1_env_int("GGML_MACH1_COOP", 0) == 0 &&
                           mach1_env_int("GGML_MACH1_WALK_VEC", 1) != 0 &&
                           mach1_env_int("GGML_MACH1_WALK_RG", 4) == 4 &&
                           mach1_env_int("GGML_MACH1_WALK_PROBE", 0) == 0 &&
                           mach1_env_int("GGML_MACH1_LUT1K", 0) == 0;
    if (!on || mach1_fold_enabled() || mach1_zbank_enabled()) {
        return 0;
    }

    const ggml_tensor * rt = cgraph->nodes[node_idx];
    const ggml_tensor * x  = rt->src[4];
    const int n = (int) rt->src[1]->ne[0];
    const int m = (int) rt->src[2]->ne[0];
    if (rt->src[0]->ne[0] != 64 || rt->src[3]->ne[0] != 2 || rt->src[3]->ne[1] != 512) {
        return 0;
    }
    // nt <= 4: the established qkvz PAIR path token-slices through the TT
    // batch exec. The default-off exact-nt16 admission keeps both existing
    // compressed MMA spines and batches only their sibling u/out transforms;
    // single-op fallbacks and FORK_Z stay on their original contracts.
    const int shexp_nt = (int) ggml_nrows(x);
    static const int qkvz_nt_max = mach1_env_int("GGML_MACH1_QKVZ_NT", 1);
    static const bool qkvz_mma16 = mach1_env_int("GGML_MACH1_QKVZ_MMA16_BATCH", 0) != 0;
    const bool mma16_nt = qkvz_mma16 && (shexp_nt == 16 || (mach1_nt32_on() && shexp_nt == 32) ||
                                         (mach1_ntlo_on() && shexp_nt >= mach1_ntlo_min() && shexp_nt < 16) ||
                                         mach1_nt_ok(shexp_nt));
    if (x->type != GGML_TYPE_F32 || shexp_nt < 1 ||
        (!mma16_nt && (shexp_nt > qkvz_nt_max || shexp_nt > 4)) ||
        !ggml_is_contiguous(x)) {
        return 0;
    }
    if (n % 16 != 0 || m % 16 != 0 || n > 4096 || m > 8192 || n/16 > 256) {
        return 0;
    }
    // the wg 1024 walk's static shared plus the m-float epilogue must fit the
    // default 48 KB budget (the wg 512 walk at m = 8192 fits, as OFUSE proved)
    if (n/16 > 128 && m > 4096) {
        return 0;
    }

    int hd = 0, K = 0, r = 0;

    // qkv pattern: cont(view qk) + cont(view v) + tiled(v) + concat
    if (node_idx + 10 <= cgraph->n_nodes) {
        const ggml_tensor * vqk = cgraph->nodes[node_idx + 1];
        const ggml_tensor * cqk = cgraph->nodes[node_idx + 2];
        const ggml_tensor * vv  = cgraph->nodes[node_idx + 3];
        const ggml_tensor * cv  = cgraph->nodes[node_idx + 4];
        const ggml_tensor * rsh = cgraph->nodes[node_idx + 5];
        const ggml_tensor * prm = cgraph->nodes[node_idx + 6];
        const ggml_tensor * cvt = cgraph->nodes[node_idx + 7];
        const ggml_tensor * rs2 = cgraph->nodes[node_idx + 8];
        ggml_tensor       * cat = cgraph->nodes[node_idx + 9];
        const int64_t qk_off = vqk->ne[0];
        if (vqk->op == GGML_OP_VIEW && cqk->op == GGML_OP_CONT &&
            vv->op == GGML_OP_VIEW && cv->op == GGML_OP_CONT && cat->op == GGML_OP_CONCAT &&
            vqk->src[0] == rt && vqk->data == rt->data && qk_off > 0 && qk_off < m &&
            vqk->ne[1] == shexp_nt && vqk->nb[1] == rt->nb[1] &&
            cqk->src[0] == vqk && cqk->type == GGML_TYPE_F32 && ggml_is_contiguous(cqk) &&
            vv->src[0] == rt && (const char *) vv->data == (const char *) rt->data + qk_off*sizeof(float) &&
            vv->ne[0] == m - qk_off && vv->ne[1] == shexp_nt && vv->nb[1] == rt->nb[1] &&
            cv->src[0] == vv && cv->type == GGML_TYPE_F32 && ggml_is_contiguous(cv) &&
            mach1_match_tiled(rsh, prm, cvt, cv, m - qk_off, &hd, &K, &r, shexp_nt) &&
            rs2->op == GGML_OP_RESHAPE && rs2->src[0] == cvt &&
            rs2->ne[0] == m - qk_off && rs2->ne[1] == shexp_nt &&
            cat->src[0] == cqk && cat->src[1] == rs2 && ggml_get_op_params_i32(cat, 0) == 0 &&
            cat->type == GGML_TYPE_F32 && ggml_is_contiguous(cat) &&
            cat->ne[0] == m && ggml_nrows(cat) == shexp_nt) {
            const ggml_op ops[10] = { GGML_OP_MACH1_RT_MM, GGML_OP_VIEW, GGML_OP_CONT,
                                      GGML_OP_VIEW, GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_PERMUTE,
                                      GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_CONCAT };
            const int out_idx = node_idx + 9;
            if (ggml_can_fuse_subgraph(cgraph, node_idx, 10, ops, &out_idx, 1)) {
                // FORK_Z: unbundle the pair - qkv walks main alone, and the z
                // op (matched at node_idx + 10 next) rides the fork stream
                // under the conv/core chain that never reads it
                const bool zfork = mach1_fork_z_on() && shexp_nt == 1 &&
                                   mach1_gdn_full_ahead(cgraph, node_idx + 14);
                // GGML_MACH1_QKV_BATCH=1: swallow the adjacent wqkv_gate op
                // (z pattern, same x - build_qkvz expands the pair back to
                // back) into one batched launch with per-op scatter dsts
                if (!zfork && mach1_qkv_batch_on() && node_idx + 14 <= cgraph->n_nodes) {
                    const ggml_tensor * rtz = cgraph->nodes[node_idx + 10];
                    ggml_tensor       * cvz = cgraph->nodes[node_idx + 13];
                    int hdz = 0, Kz = 0, rz = 0;
                    const int mz = rtz->op == GGML_OP_MACH1_RT_MM ? (int) rtz->src[2]->ne[0] : 0;
                    if (mz > 0 && mz % 16 == 0 && mz <= 8192 && (n/16 <= 128 || mz <= 4096) &&
                        rtz->src[4] == x && rtz->src[3] == rt->src[3] &&
                        rtz->src[0]->ne[0] == 64 && (int) rtz->src[1]->ne[0] == n &&
                        mach1_match_tiled(cgraph->nodes[node_idx + 11], cgraph->nodes[node_idx + 12],
                                    cvz, rtz, mz, &hdz, &Kz, &rz, shexp_nt)) {
                        const ggml_op opsz[14] = { GGML_OP_MACH1_RT_MM, GGML_OP_VIEW, GGML_OP_CONT,
                                                   GGML_OP_VIEW, GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_PERMUTE,
                                                   GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_CONCAT,
                                                   GGML_OP_MACH1_RT_MM, GGML_OP_RESHAPE, GGML_OP_PERMUTE,
                                                   GGML_OP_CONT };
                        const int out_idxs[2] = { node_idx + 9, node_idx + 13 };
                        if (ggml_can_fuse_subgraph(cgraph, node_idx, 14, opsz, out_idxs, 2)) {
                            const int * pq = mach1_vtiled_perm(ctx.device, ctx.stream(), m, (int) qk_off, hd, K, r);
                            const int * pz = mach1_vtiled_perm(ctx.device, ctx.stream(), mz, 0, hdz, Kz, rz);
                            if (pq != nullptr && pz != nullptr) {
                                const ggml_tensor * const rts[3]   = { rt, rtz, nullptr };
                                ggml_tensor       * const outs[3]  = { cat, cvz, nullptr };
                                const int         * const perms[3] = { pq, pz, nullptr };
                                mach1_fork_state * zfnt = nullptr;
                                if (shexp_nt >= 2 && shexp_nt <= mach1_fork_z_nt() &&
                                    mach1_gdn_full_ahead(cgraph, node_idx + 14)) {
                                    mach1_fork_state * f = mach1_fork_get(ctx.device, ctx.stream(), 0);
                                    if (f != nullptr && mach1_fork_zres(f, ctx.stream(),
                                            ((size_t) mz + (size_t) n)*(size_t) shexp_nt)) {
                                        zfnt = f;
                                    }
                                }
                                const int skipped = mach1_rt_batch_exec(
                                    ctx, "qkvzb", rts, outs, perms, 2, 13, zfnt, mma16_nt);
                                if (skipped > 0) {
                                    return skipped;
                                }
                            }
                        }
                    }
                }
                if (shexp_nt == 1) {
                    return ggml_cuda_mach1_vtiled_exec(ctx, rt, cat, (int) qk_off, hd, K, r, 9);
                }
                return 0;   // the single-op vtiled exec is nt == 1 only
            }
        }
    }

    // z pattern: tiled(z) directly on the rt output
    if (node_idx + 4 <= cgraph->n_nodes) {
        const ggml_tensor * rsh = cgraph->nodes[node_idx + 1];
        const ggml_tensor * prm = cgraph->nodes[node_idx + 2];
        ggml_tensor       * cvt = cgraph->nodes[node_idx + 3];
        if (mach1_match_tiled(rsh, prm, cvt, rt, m, &hd, &K, &r)) {
            const ggml_op ops[4] = { GGML_OP_MACH1_RT_MM, GGML_OP_RESHAPE, GGML_OP_PERMUTE, GGML_OP_CONT };
            const int out_idx = node_idx + 3;
            if (ggml_can_fuse_subgraph(cgraph, node_idx, 4, ops, &out_idx, 1)) {
                // The legacy nt == 1 FORK_Z lane ran the z walk on s2 reading
                // x with the join to main landing only after the gdn core
                // launch, so main could reuse x's pool bytes mid-walk. It is
                // disabled; the batched FORK_Z path sequences x safely.
                return ggml_cuda_mach1_vtiled_exec(ctx, rt, cvt, 0, hd, K, r, 3);
            }
        }
    }

    return 0;
}

// XPERM fold (GGML_MACH1_XPERM=1; decode only). The GDN layers emit the
// INVERSE (tiled -> grouped) activation reorder ahead of ssm_out
// (build_layer_attn_linear in src/models/mach1.cpp): [CONT(permute(reshape(attn))),
// RESHAPE, MACH1_RT_MM]. The cont is a pure row permutation of the rt op's
// INPUT, so the walk's u prologue reads x through the row map instead of
// materializing it - one glue node and its boundary gone per GDN layer,
// values bit-identical (the map moves load addresses, never values).
int ggml_cuda_mach1_xperm_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    static const bool on = mach1_env_int("GGML_MACH1_XPERM", 0) != 0 &&
                           mach1_env_int("GGML_MACH1_ABLATE", 0) == 0 &&
                           mach1_env_int("GGML_MACH1_FUSE_RT", 0) == 0 &&
                           mach1_env_int("GGML_MACH1_COOP", 0) == 0 &&
                           mach1_env_int("GGML_MACH1_WALK_VEC", 1) != 0 &&
                           mach1_env_int("GGML_MACH1_WALK_RG", 4) == 4 &&
                           mach1_env_int("GGML_MACH1_WALK_PROBE", 0) == 0 &&
                           mach1_env_int("GGML_MACH1_LUT1K", 0) == 0;
    if (!on || mach1_fold_enabled() || mach1_zbank_enabled()) {
        return 0;
    }
    if (node_idx + 3 > cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * c = cgraph->nodes[node_idx];
    if (c->op != GGML_OP_CONT || c->src[0] == nullptr) {
        return 0;
    }
    const ggml_tensor * prm = c->src[0];
    if (prm->op != GGML_OP_PERMUTE || prm->src[0] == nullptr) {
        return 0;
    }
    const ggml_tensor * rsh = prm->src[0];
    if (rsh->op != GGML_OP_RESHAPE || rsh->src[0] == nullptr) {
        return 0;
    }
    const ggml_tensor * xin = rsh->src[0];
    if (xin->type != GGML_TYPE_F32 || !ggml_is_contiguous(xin)) {
        return 0;
    }
    const int64_t rows = rsh->ne[0]*rsh->ne[1]*rsh->ne[2];
    int hd = 0, K = 0, r = 0;
    if (ggml_nelements(xin) != rows || !mach1_match_tiled(rsh, prm, c, xin, rows, &hd, &K, &r)) {
        return 0;
    }
    const ggml_tensor * rs2 = cgraph->nodes[node_idx + 1];
    ggml_tensor       * rt  = cgraph->nodes[node_idx + 2];
    if (rs2->op != GGML_OP_RESHAPE || rs2->src[0] != c || ggml_nrows(rs2) != 1 ||
        rt->op != GGML_OP_MACH1_RT_MM || rt->src[4] != rs2) {
        return 0;
    }
    if (rt->src[0]->ne[0] != 64 || rt->src[3]->ne[0] != 2 || rt->src[3]->ne[1] != 512 ||
        (int64_t) rt->src[1]->ne[0] != rows) {
        return 0;
    }
    const int n = (int) rows;
    const int m = (int) rt->src[2]->ne[0];
    // the fold rides the in-block butterfly u prologue only: wg 1024 walks
    // (n/16 > 128 keeps the TC prologue off) inside the UFUSE window - the
    // same gate ggml_cuda_op_mach1_rt_mm computes and asserts
    static const int ufuse_maxb = mach1_env_int("GGML_MACH1_UFUSE_MAXB", 256);
    if (n % 16 != 0 || n/16 <= 128 || n/16 > 256 || n > 4096 ||
        m % 16 != 0 || m > 8192 || ufuse_maxb <= 0 || m/16 > ufuse_maxb) {
        return 0;
    }
    const ggml_op ops3[3] = { GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_MACH1_RT_MM };
    const int out_idx = node_idx + 2;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, 3, ops3, &out_idx, 1)) {
        return 0;
    }
    // gather map: the v_tiled builder with the middle extents swapped
    // (mach1_match_tiled read the INVERSE triple, so K and r trade places);
    // verified elementwise against the cont semantics in the round-239 sim
    const int * xp = mach1_vtiled_perm(ctx.device, ctx.stream(), n, 0, hd, r, K);
    if (xp == nullptr) {
        return 0;
    }
    ggml_cuda_op_mach1_rt_mm(ctx, rt, xin->data, xp);
    return 2;
}

// qkv trio batching (GGML_MACH1_QKV_BATCH=1; decode only): three adjacent
// independent rt ops on one input - the attention wq/wk/wv, which
// build_layer_attn (src/models/mach1.cpp) expands back to back - run as ONE
// launch with per-op folded out stages. This is batching, not
// region-swallowing: all three dsts stay ordinary tensors the rest of the
// graph consumes (the q views/norm/rope read them as before).
int ggml_cuda_mach1_qkv_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    // fold/zbank/bank/fuse_rt/coop serve rt ops from other numeric paths, and
    // the non-default walk knobs leave the rows_v kernel this batch rides
    static const bool on = mach1_qkv_batch_on() &&
                           mach1_env_int("GGML_MACH1_ABLATE", 0) == 0 &&
                           mach1_env_int("GGML_MACH1_FUSE_RT", 0) == 0 &&
                           mach1_env_int("GGML_MACH1_COOP", 0) == 0 &&
                           mach1_env_int("GGML_MACH1_WALK_VEC", 1) != 0 &&
                           mach1_env_int("GGML_MACH1_WALK_RG", 4) == 4 &&
                           mach1_env_int("GGML_MACH1_WALK_PROBE", 0) == 0 &&
                           mach1_env_int("GGML_MACH1_LUT1K", 0) == 0;
    if (!on || mach1_fold_enabled() || mach1_zbank_enabled()) {
        return 0;
    }
    if (node_idx + 3 > cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * rt[3];
    for (int i = 0; i < 3; ++i) {
        rt[i] = cgraph->nodes[node_idx + i];
        if (rt[i]->op != GGML_OP_MACH1_RT_MM) {
            return 0;
        }
    }

    // one x and one codebook across the three rt ops; decode only
    const ggml_tensor * x    = rt[0]->src[4];
    const ggml_tensor * tlut = rt[0]->src[3];
    // token slices: mach1_rt_batch_exec and the batch kernel are z-slice
    // generalized (scratch [nt, .], counter trios per slice, blockIdx.z)
    const int qkv_nt = (int) ggml_nrows(x);
    static const int qkv_nt_max = mach1_env_int("GGML_MACH1_QKVB_NT", 1);
    static const bool p16_sibling = mach1_env_int("GGML_MACH1_P16_SIBLING_BATCH", 0) != 0;
    const bool mma16_nt = p16_sibling && (qkv_nt == 16 || (mach1_nt32_on() && qkv_nt == 32) ||
                                          (mach1_ntlo_on() && qkv_nt >= mach1_ntlo_min() && qkv_nt < 16) ||
                                          mach1_nt_ok(qkv_nt));
    if (x->type != GGML_TYPE_F32 || qkv_nt < 1 ||
        (!mma16_nt && (qkv_nt > qkv_nt_max || qkv_nt > 4)) ||
        !ggml_is_contiguous(x)) {
        return 0;
    }
    if (tlut->ne[0] != 2 || tlut->ne[1] != 512) {
        return 0;
    }
    const int n = (int) rt[0]->src[1]->ne[0];
    if (n % 16 != 0 || n > 4096 || n/16 > 256) {
        return 0;
    }
    int mmax = 0;
    for (int i = 0; i < 3; ++i) {
        if (rt[i]->src[4] != x || rt[i]->src[3] != tlut || rt[i]->src[0]->ne[0] != 64 ||
            (int) rt[i]->src[1]->ne[0] != n) {
            return 0;
        }
        const int m = (int) rt[i]->src[2]->ne[0];
        if (m % 16 != 0 || m > 8192) {
            return 0;
        }
        mmax = std::max(mmax, m);
    }
    // the wg 1024 walk's static shared plus the m-float epilogue must fit the
    // default 48 KB budget (the wg 512 walk at m = 8192 fits, as OFUSE proved)
    if (n/16 > 128 && mmax > 4096) {
        return 0;
    }

    const ggml_op ops[3] = { GGML_OP_MACH1_RT_MM, GGML_OP_MACH1_RT_MM, GGML_OP_MACH1_RT_MM };
    const int out_idxs[3] = { node_idx, node_idx + 1, node_idx + 2 };
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, 3, ops, out_idxs, 3)) {
        return 0;
    }

    ggml_tensor * const outs[3]  = { cgraph->nodes[node_idx], cgraph->nodes[node_idx + 1],
                                     cgraph->nodes[node_idx + 2] };
    const int   * const perms[3] = { nullptr, nullptr, nullptr };
    return mach1_rt_batch_exec(ctx, "qkvb", rt, outs, perms, 3, 2,
                               nullptr, false, mma16_nt);
}

// GDN-core fusion (GGML_MACH1_GDN_FUSE=1; decode only). Between the fused
// conv+silu and the gated-norm epilogue every GDN layer runs a chain of tiny
// per-head nodes feeding GGML_OP_GATED_DELTA_NET:
//   l2_norm(q_conv), l2_norm(k_conv)                       2 launches
//   sigmoid(beta)                                          1
//   add(alpha, dt), fused softplus*ssm_a                   2
//   gated_delta_net (+ fused state-cache cpy)              1
// One launch replaces all of them. The delta-rule block body is the stock
// gated_delta_net_cuda kernel with the same (head, col-group) grid role and
// warp reduction; the pre-chain values are computed in-block by replicating
// each stock kernel's op order exactly: l2_norm_f32<WARP_SIZE>'s serial
// column stride and rsqrtf(fmaxf(sum, eps*eps)) (the silu itself is not
// replicated - the fused conv+silu dst is read directly), op_sigmoid /
// op_softplus formulas, plain add/mul. All inputs are live at the gdn node
// and the kernel writes exactly what the stock kernel writes (the attention
// scores and the recurrent-cache row), so the hazard profile is identical to
// the unfused path; no completion counter and no extra shared, preserving
// the stock kernel's occupancy.
//
// The post-side chain (fused rms_norm*ssm_norm, fused silu(z)*normed, and
// the tiled->grouped reorder cont) stays on ordinary nodes: a first cut
// folded it into the last block of this grid behind a completion counter and
// LOST 2% end to end - the one-block epilogue serializes ~8K loads plus 32
// head reductions per layer, and its 32 KB staging buffer (allocated by
// every block) halves the delta-rule phase's occupancy. Folding it instead
// into the ssm_out rt walk's UFUSE prologue (a row-permuted gated load) is
// the follow-up rung.
//
// GGML_MACH1_GDN_FULL extends the region backwards over the two nodes ahead of
// it, both of which carry real BYTES rather than a launch boundary only:
//
//   GATHER = 1 (FULL >= 1): the recurrent state arrives as
//   get_rows(ssm_states_all, s_copy) - a 2 MB per layer round trip whose only
//   consumer is this kernel. The kernel instead reads the cache row s_copy[0]
//   directly, on the device, so no host sync is needed to learn the index. A
//   warp owns one (head, column) and reads exactly the S floats it later
//   writes, so the in-place case (gather row == scatter row, the usual one at
//   decode) needs no ordering beyond that warp's own; when the rows differ the
//   read and write sets are disjoint.
//
//   CONV = 1 (FULL = 2): additionally folds the fused conv+silu in. A block
//   recomputes only the channels it consumes - its q and k head, and its own
//   CGY v columns - out of the [4, C] conv input, replicating ssm_conv_f32's
//   n_t == 1 body (same tap order, same absent-bias add, same silu). That
//   trades one launch for 2*S + CGY redundant 4-tap dots per block against an
//   L2-resident input, so CGY doubles to halve the block count.
// GDN_FULL level 3 (PROJ/CONVG): the region extends backwards over the
// beta/alpha [n,Hv] projections and the conv-state round trip. PROJ computes
// the two per-head dots in-kernel (ulp-class vs the mul_mat kernel: KLD
// gate, not smoke). CONVG reads the conv cache row and the qkv row directly
// - the CONCAT becomes an index select, the GET_ROWS a device gather - and
// the 3-tap state shift runs as its own tiny kernel AFTER this one, so no
// block can overtake another block's read of the old state.
struct mach1_gdn_proj_args {
    const float * x;       // [n] the layer's attn-norm input
    const void  * wb;      // ssm_beta  [n, Hv]
    const void  * wa;      // ssm_alpha [n, Hv]
    int           n;
    int           wtype;   // 0 = f32, 1 = f16
};

struct mach1_gdn_convg_args {
    const float * crow_base;   // conv cache base (reshaped rows)
    const int   * cidx;        // cache row to read (device)
    int64_t       crow_stride; // row stride in floats
    const float * qkv;         // the layer's qkv_mixed row [C]
};

// n_seqs > 1 (decode batch): per-seq element strides applied at blockIdx.y.
// All-zero at n_seqs == 1, so the single-seq launches are bit-identical.
// The GATHER fold stays n_seqs == 1 only: s_copy[i] may name another seq's
// scatter row after a slot shuffle, and blocks of different seqs have no
// ordering, so the batched path reads the graph's get_rows output instead.
struct mach1_gdn_seq_args {
    int64_t conv_ss = 0;   // conv+silu output   [C, 1, ns]
    int64_t cx_ss   = 0;   // conv input concat  [DC, C, ns]
    int64_t ab_ss   = 0;   // alpha/beta rows    [Hv, ns]
    int64_t st_ss   = 0;   // gathered state     [S, S, Hv, ns]
    int64_t at_ss   = 0;   // attn scores out    [S, Hv, 1, ns]
    int64_t so_ss   = 0;   // cache scatter rows [S*S*Hv, ns]
};

template <int S, int CGY = 4, int GATHER = 0, int CONV = 0, int DC = 4, int PROJ = 0>
__global__ void __launch_bounds__(CGY*32, 2) mach1_gdn_core_kernel(
        const float * __restrict__ conv,      // CONV = 0: fused conv+silu output row [2*S*Hk + S*Hv]
        const float * __restrict__ alpha,     // [Hv] raw ssm_alpha projection
        const float * __restrict__ dt,        // [Hv] ssm_dt bias
        const float * __restrict__ ga,        // [Hv] ssm_a
        const float * __restrict__ beta,      // [Hv] raw ssm_beta projection
        const float * __restrict__ state_in,  // GATHER = 0: [S, S, Hv]
        float       * __restrict__ attn,      // gdn dst: per-head attention scores [S*Hv]
        float       * __restrict__ state_out, // recurrent-cache row [S*S*Hv]
        const int Hk,
        const float eps_l2, const float scale,
        const float * __restrict__ states_all,// GATHER = 1: recurrent cache base
        const int   * __restrict__ sidx,      // GATHER = 1: cache row to read
        const int64_t srow,                   // GATHER = 1: cache row stride in floats
        const float * __restrict__ conv_x,    // CONV = 1: conv input [DC, C] (null under CONVG)
        const float * __restrict__ conv_w,    // CONV = 1: conv1d weight [DC, C]
        const float * __restrict__ conv_b,    // CONV = 1: conv bias, always null here
        const mach1_gdn_proj_args  pa = {},   // PROJ = 1
        const mach1_gdn_convg_args ca = {},   // PROJ = 1: virtual conv input
        const mach1_gdn_seq_args   sa = {}) { // n_seqs > 1: per-seq strides
    constexpr int warp_size     = 32;
    constexpr int rows_per_lane = S / warp_size;

    const int h    = blockIdx.x;
    const int col  = blockIdx.z*CGY + threadIdx.y;
    const int lane = threadIdx.x;

    // the seq slice: blockIdx.y is 0 and every stride 0 at n_seqs == 1
    const int seq = blockIdx.y;
    conv      += (int64_t) seq*sa.conv_ss;
    conv_x    += (int64_t) seq*sa.cx_ss;
    alpha     += (int64_t) seq*sa.ab_ss;
    beta      += (int64_t) seq*sa.ab_ss;
    state_in  += (int64_t) seq*sa.st_ss;
    attn      += (int64_t) seq*sa.at_ss;
    state_out += (int64_t) seq*sa.so_ss;

    const int kh = h % Hk;

    // ssm_conv_f32<true, 128, DC>'s n_t == 1 body for one channel
    __shared__ float sh_qk[CONV ? 2*S : 1];
    __shared__ float sh_v [CONV ? CGY : 1];
    if (CONV) {
        const int tid = threadIdx.y*warp_size + lane;
        const float * crow = PROJ ? ca.crow_base + (int64_t) ca.cidx[0]*ca.crow_stride : nullptr;
        for (int i = tid; i < 2*S + CGY; i += CGY*warp_size) {
            const int c = i <   S ? kh*S + i
                        : i < 2*S ? Hk*S + kh*S + (i - S)
                                  : 2*Hk*S + h*S + blockIdx.z*CGY + (i - 2*S);
            float sumf = 0.0f;
#pragma unroll
            for (int j = 0; j < DC; ++j) {
                // CONVG: tap j < DC-1 comes from the cache row, the last tap
                // is the fresh qkv value - exactly the CONCAT's layout
                const float xv = PROJ ? (j < DC - 1 ? crow[c*(DC - 1) + j] : ca.qkv[c])
                                      : conv_x[c*DC + j];
                sumf += xv * conv_w[c*DC + j];
            }
            sumf += conv_b != nullptr ? conv_b[c] : 0.0f;
            (i < 2*S ? sh_qk[i] : sh_v[i - 2*S]) = ggml_cuda_op_silu_single(sumf);
        }
        __syncthreads();
    }

    const float * qb;
    const float * kb;
    float vc;
    if (CONV) {
        qb = sh_qk;
        kb = sh_qk + S;
        vc = sh_v[threadIdx.y];
    } else {
        qb = conv + kh*S;
        kb = conv + Hk*S + kh*S;
        vc = conv[2*Hk*S + h*S + col];
    }

    // l2_norm_f32<WARP_SIZE> body per head vector, warp 0 = q, warp 1 = k
    __shared__ float sh_scale[2];
    if (threadIdx.y < 2) {
        const float * xb = threadIdx.y == 0 ? qb : kb;
        float tmp = 0.0f;
        for (int c = lane; c < S; c += warp_size) {
            const float xi = xb[c];
            tmp += xi*xi;
        }
        tmp = warp_reduce_sum(tmp);
        if (lane == 0) {
            sh_scale[threadIdx.y] = rsqrtf(fmaxf(tmp, eps_l2*eps_l2));
        }
    }
    __syncthreads();
    const float qs = sh_scale[0];
    const float ks = sh_scale[1];

    // PROJ: the two [n] dots for this head, block-reduced; every (h, z)
    // block recomputes them - 4K MACs against the launches they replace
    __shared__ float sh_ab[2];
    if (PROJ) {
        const int tid = threadIdx.y*warp_size + lane;
        float pa_ = 0.0f;
        float pb_ = 0.0f;
        if (pa.wtype == 1) {
            const half * wa16 = (const half *) pa.wa + (int64_t) h*pa.n;
            const half * wb16 = (const half *) pa.wb + (int64_t) h*pa.n;
            for (int i = tid; i < pa.n; i += CGY*warp_size) {
                pa_ += pa.x[i]*__half2float(wa16[i]);
                pb_ += pa.x[i]*__half2float(wb16[i]);
            }
        } else {
            const float * wa32 = (const float *) pa.wa + (int64_t) h*pa.n;
            const float * wb32 = (const float *) pa.wb + (int64_t) h*pa.n;
            for (int i = tid; i < pa.n; i += CGY*warp_size) {
                pa_ += pa.x[i]*wa32[i];
                pb_ += pa.x[i]*wb32[i];
            }
        }
        pa_ = warp_reduce_sum(pa_);
        pb_ = warp_reduce_sum(pb_);
        __shared__ float sh_red[2][CGY];
        if (lane == 0) {
            sh_red[0][threadIdx.y] = pa_;
            sh_red[1][threadIdx.y] = pb_;
        }
        __syncthreads();
        if (threadIdx.y == 0 && lane < 2) {
            float acc = 0.0f;
#pragma unroll
            for (int w = 0; w < CGY; ++w) {
                acc += sh_red[lane][w];
            }
            sh_ab[lane] = acc;
        }
        __syncthreads();
    }

    // gate chain: add, op_softplus, mul, then the delta rule's expf; op_sigmoid
    const float ab       = (PROJ ? sh_ab[0] : alpha[h]) + dt[h];
    const float sp       = ab > 20.0f ? ab : logf(1.0f + expf(ab));
    const float g_val    = expf(sp * ga[h]);
    const float beta_val = 1.0f / (1.0f + expf(-(PROJ ? sh_ab[1] : beta[h])));

    // gated_delta_net_cuda body (nt = 1, !KDA, !keep_rs_t), q/k read through
    // the replicated l2 scale instead of the elided l2_norm dsts
    const float * state_base = state_in;
    if (GATHER) {
        state_base = states_all + (int64_t) sidx[0]*srow;
    }
    const float * curr_state = state_base + h*S*S + col*S;
    float s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r*warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r*warp_size + lane;
        k_reg[r] = ks*kb[i];
        q_reg[r] = qs*qb[i];
    }

    float kv_shard = 0.0f;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        kv_shard += s_shard[r] * k_reg[r];
    }
    const float kv_col = warp_reduce_sum<warp_size>(kv_shard);

    const float delta_col = (vc - g_val * kv_col) * beta_val;

    float attn_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        s_shard[r]    = g_val * s_shard[r] + k_reg[r] * delta_col;
        attn_partial += s_shard[r] * q_reg[r];
    }
    const float attn_col = warp_reduce_sum<warp_size>(attn_partial);

    if (lane == 0) {
        attn[h*S + col] = attn_col * scale;
    }

    float * st = state_out + h*S*S;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i    = r*warp_size + lane;
        st[col*S + i] = s_shard[r];
    }
}

// what the fused launch needs out of the matched region
struct mach1_gdn_plan {
    const ggml_tensor * conv;    // fused conv+silu output
    const ggml_tensor * alpha;
    const ggml_tensor * dt;
    const ggml_tensor * ga;
    const ggml_tensor * beta;
    const ggml_tensor * state;   // gathered recurrent state
    const ggml_tensor * gdn;
    const ggml_tensor * vcd;     // cache row the state scatter targets
    int64_t S, Hk, Hv;
    int64_t ns;                  // decode sequences in the ubatch (tokens/seq == 1)
    float   eps_l2;
};

// the 15-node core chain, matched positionally from its q L2_NORM
static bool mach1_gdn_match_core(const ggml_cgraph * cgraph, int node_idx, mach1_gdn_plan & p) {
    const int len = 15;   // l2_norm(q) .. state-cache cpy, matched positionally below
    if (node_idx + len > cgraph->n_nodes) {
        return false;
    }

    const ggml_tensor * l2q = cgraph->nodes[node_idx +  0];
    const ggml_tensor * vk  = cgraph->nodes[node_idx +  1];
    const ggml_tensor * l2k = cgraph->nodes[node_idx +  2];
    const ggml_tensor * vv  = cgraph->nodes[node_idx +  3];
    const ggml_tensor * ra  = cgraph->nodes[node_idx +  4];
    const ggml_tensor * add = cgraph->nodes[node_idx +  5];
    const ggml_tensor * sp  = cgraph->nodes[node_idx +  6];
    const ggml_tensor * gm  = cgraph->nodes[node_idx +  7];
    const ggml_tensor * rg  = cgraph->nodes[node_idx +  8];
    const ggml_tensor * rb  = cgraph->nodes[node_idx +  9];
    const ggml_tensor * sig = cgraph->nodes[node_idx + 10];
    const ggml_tensor * gdn = cgraph->nodes[node_idx + 11];
    const ggml_tensor * vns = cgraph->nodes[node_idx + 12];
    const ggml_tensor * vcd = cgraph->nodes[node_idx + 13];
    const ggml_tensor * cpy = cgraph->nodes[node_idx + 14];

    if (l2q->op != GGML_OP_L2_NORM || vk->op != GGML_OP_VIEW || l2k->op != GGML_OP_L2_NORM ||
        vv->op != GGML_OP_VIEW || ra->op != GGML_OP_RESHAPE || add->op != GGML_OP_ADD ||
        sp->op != GGML_OP_UNARY || gm->op != GGML_OP_MUL || rg->op != GGML_OP_RESHAPE ||
        rb->op != GGML_OP_RESHAPE || sig->op != GGML_OP_UNARY || gdn->op != GGML_OP_GATED_DELTA_NET ||
        vns->op != GGML_OP_VIEW || vcd->op != GGML_OP_VIEW || cpy->op != GGML_OP_CPY) {
        return false;
    }
    if (ggml_get_unary_op(sp) != GGML_UNARY_OP_SOFTPLUS || ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID) {
        return false;
    }

    // conv+silu output and the q/k/v head views on it; decode only (one
    // token per seq; the seq axis rides ne[3] of the views / ne[2] of conv)
    const ggml_tensor * vq   = l2q->src[0];
    const ggml_tensor * conv = vq->op == GGML_OP_VIEW ? vq->src[0] : nullptr;
    if (conv == nullptr || conv->type != GGML_TYPE_F32 || !ggml_is_contiguous(conv) ||
        conv->ne[1] != 1) {
        return false;
    }
    const int64_t S  = vq->ne[0];
    const int64_t Hk = vq->ne[1];
    const int64_t Hv = vv->ne[1];
    const int64_t ns = vq->ne[3];
    if (S != 128 || Hk <= 0 || Hv <= 0 || Hv % Hk != 0) {
        return false;
    }
    if (vq->ne[2] != 1 || ns < 1 || ns > 8) {   // one token per seq
        return false;
    }
    if (conv->ne[0] != 2*S*Hk + S*Hv || conv->ne[2] != ns) {
        return false;
    }
    auto head_view_ok = [&](const ggml_tensor * v, int64_t heads, size_t offs) {
        return v->type == GGML_TYPE_F32 && v->src[0] == conv && v->view_offs == offs &&
               v->ne[0] == S && v->ne[1] == heads && v->ne[2] == 1 && v->ne[3] == ns &&
               v->nb[0] == sizeof(float) && v->nb[1] == S*sizeof(float) && v->nb[2] == conv->nb[1] &&
               (ns == 1 || v->nb[3] == conv->nb[2]);
    };
    if (!head_view_ok(vq, Hk, 0) || !head_view_ok(vk, Hk, S*Hk*sizeof(float)) ||
        !head_view_ok(vv, Hv, 2*S*Hk*sizeof(float))) {
        return false;
    }
    if (l2k->src[0] != vk || l2q->type != GGML_TYPE_F32 || l2k->type != GGML_TYPE_F32) {
        return false;
    }
    const float eps_l2 = ggml_get_op_params_f32(l2q, 0);
    if (ggml_get_op_params_f32(l2k, 0) != eps_l2) {
        return false;
    }

    // gate chain: g = softplus(alpha + dt) * ssm_a, b = sigmoid(beta)
    const ggml_tensor * alpha = ra->src[0];
    if (alpha->type != GGML_TYPE_F32 || !ggml_is_contiguous(alpha) ||
        alpha->ne[0] != Hv || ggml_nrows(alpha) != ns) {
        return false;
    }
    const ggml_tensor * dt = add->src[1];
    if (add->src[0] != ra || add->type != GGML_TYPE_F32 ||
        dt->type != GGML_TYPE_F32 || !ggml_is_contiguous(dt) || ggml_nelements(dt) != Hv) {
        return false;
    }
    const ggml_tensor * ga = gm->src[1];
    if (sp->src[0] != add || gm->src[0] != sp || gm->type != GGML_TYPE_F32 ||
        ga->type != GGML_TYPE_F32 || !ggml_is_contiguous(ga) || ggml_nelements(ga) != Hv) {
        return false;
    }
    if (rg->src[0] != gm || rg->ne[0] != 1 || rg->ne[1] != Hv || rg->ne[3] != ns) {
        return false;
    }
    const ggml_tensor * beta = rb->src[0];
    if (beta->type != GGML_TYPE_F32 || !ggml_is_contiguous(beta) ||
        beta->ne[0] != Hv || ggml_nrows(beta) != ns) {
        return false;
    }
    if (rb->ne[0] != 1 || rb->ne[1] != Hv || rb->ne[3] != ns ||
        sig->src[0] != rb || sig->type != GGML_TYPE_F32) {
        return false;
    }

    // the delta-rule op itself: decode shape, snapshot count 1, state gathered
    const ggml_tensor * state = gdn->src[5];
    if (gdn->src[0] != l2q || gdn->src[1] != l2k || gdn->src[2] != vv ||
        gdn->src[3] != rg || gdn->src[4] != sig || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_get_op_params_i32(gdn, 0) != 1) {
        return false;
    }
    if (state->type != GGML_TYPE_F32 || !ggml_is_contiguous(state) ||
        state->ne[0] != S || state->ne[1] != S || state->ne[2] != Hv || state->ne[3] != ns) {
        return false;
    }

    // state-cache scatter, exactly as ggml_cuda_try_gdn_cache_fusion requires;
    // at ns > 1 the scatter rows are the contiguous slots [kv_head, kv_head+ns)
    const int64_t D = S*S*Hv;
    if (vns->src[0] != gdn || vns->view_offs != (size_t) S*Hv*ns*sizeof(float) || !ggml_is_contiguous(vns)) {
        return false;
    }
    if (cpy->src[0] != vns || cpy->src[1] != vcd || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    if (vcd->type != GGML_TYPE_F32 || vcd->data == nullptr ||
        vcd->ne[0] != D || vcd->ne[1] != ns || vcd->ne[2] != 1 || vcd->ne[3] != 1 ||
        vcd->nb[0] != sizeof(float) || vcd->nb[1] != (size_t) D*sizeof(float)) {
        return false;
    }

    // no intermediate may be a graph output, be consumed outside the region,
    // or carry stale graph-reuse state. The gdn node is exempt from the
    // use-count rule: the kernel fully materializes its attention scores,
    // which the downstream gated norm reads as ordinary nodes (the state
    // tail stays unwritten, exactly like the stock gdn cache fusion).
    // ggml_can_fuse_subgraph cannot be used here: the region's views alias
    // non-constant tensors from outside it (conv/alpha/beta), which its
    // view_src rule rejects, so the same checks run manually.
    for (int j = node_idx; j < node_idx + len; ++j) {
        const ggml_tensor * nn = cgraph->nodes[j];
        if ((nn->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;
        }
        if (nn->flags & GGML_TENSOR_FLAG_OUTPUT) {
            return false;
        }
        if (nn == gdn) {
            continue;
        }
        int uses = 0;
        for (int j2 = node_idx; j2 < node_idx + len; ++j2) {
            for (int si = 0; si < GGML_MAX_SRC; ++si) {
                if (cgraph->nodes[j2]->src[si] == nn) {
                    uses++;
                }
            }
        }
        if (uses != ggml_node_get_use_count(cgraph, j)) {
            return false;
        }
    }

    p = { conv, alpha, dt, ga, beta, state, gdn, vcd, S, Hk, Hv, ns, eps_l2 };
    return true;
}

// GGML_MACH1_GDN_NT: max decode seqs the full-region fusion covers (the
// batched form keeps the graph's state gather - see mach1_gdn_seq_args)
static int64_t mach1_gdn_nt_cap() {
    static const int64_t cap = std::max<int64_t>(1, std::min<int64_t>(8,
        mach1_env_int("GGML_MACH1_GDN_NT", 4)));
    return cap;
}

// GGML_MACH1_GDN_FULL: 0 off, 1 fold the state gather, 2 also fold the conv,
// 3 also fold the beta/alpha projections and the glue window (19 nodes, or
// 17 in form B) into the PROJ core (ggml_cuda_mach1_gdn_proj_fuse).
//
// Level 3 is NOT the default. It long appeared to never fire; the real cause
// was the second glue-window form (see the form A/B note at the matcher), and
// with both forms matched it engages on every GDN layer. It stays opt-in
// until a throughput receipt beats level 2.
void mach1_diag_note(const char * key);   // defined in ggml-cuda.cu

static int mach1_gdn_full_level() {
    static const int lvl = mach1_env_int("GGML_MACH1_GDN_FULL", 2);
    return lvl;
}

// what the dispatch loop skips anyway (mirrors ggml_cuda_is_view_or_noop)
static bool mach1_node_is_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

// mach1 anchor: a mach1 graph always carries mach1 weight ops ahead of any
// GDN region, a stock GATED_DELTA_NET graph never does - requiring one keeps
// the GDN fusions off stock models. One linear scan, cached per (graph, size).
static bool mach1_graph_has_mach1_op(const ggml_cgraph * cgraph) {
    static std::mutex mtx;
    static const ggml_cgraph * cg = nullptr;
    static int  cn  = 0;
    static bool hit = false;

    std::lock_guard<std::mutex> lock(mtx);
    if (cgraph != cg || cgraph->n_nodes != cn) {
        cg  = cgraph;
        cn  = cgraph->n_nodes;
        hit = false;
        for (int i = 0; i < cn && !hit; ++i) {
            switch (cgraph->nodes[i]->op) {
                case GGML_OP_MACH1_NE_MM:
                case GGML_OP_MACH1_EMBED_ROWS:
                case GGML_OP_MACH1_EXP_MM:
                case GGML_OP_MACH1_EXP_BASIS:
                case GGML_OP_MACH1_RT_MM:
                case GGML_OP_MACH1_HEAD_MM:
                case GGML_OP_MACH1_EMBED_GATHER:
                    hit = true;
                    break;
                default:
                    break;
            }
        }
    }
    return hit;
}

// NOTE: unlike the weight-path matchers this region carries no mach1 stage, so
// GGML_MACH1_ABLATE has nothing to skip inside it and does not bail here - that
// is what lets the ABLATE=255 glue floor be measured with the region live.
int ggml_cuda_mach1_gdn_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    static const bool on = mach1_env_int("GGML_MACH1_GDN_FUSE", 1) != 0;
    mach1_gdn_plan p;
    if (!on || !mach1_graph_has_mach1_op(cgraph) || !mach1_gdn_match_core(cgraph, node_idx, p)) {
        return 0;
    }
    if (p.ns != 1) {   // batched decode goes through the full-region path
        return 0;
    }

    cudaStream_t stream = ctx.stream();

    char shp[64];
    snprintf(shp, sizeof(shp), " Hk=%d Hv=%d", (int) p.Hk, (int) p.Hv);
    mach1_timed(stream, std::string("gdn_core") + shp, [&]() {
    mach1_launch(mach1_gdn_core_kernel<128>,
        ggml_cuda_kernel_launch_params(dim3(p.Hv, 1, 128/4), dim3(32, 4, 1), 0, stream),
        (const float *) p.conv->data, (const float *) p.alpha->data, (const float *) p.dt->data,
        (const float *) p.ga->data, (const float *) p.beta->data, (const float *) p.state->data,
        (float *) p.gdn->data, (float *) p.vcd->data,
        (int) p.Hk, p.eps_l2, 1.0f / sqrtf((float) p.S),
        nullptr, nullptr, 0, nullptr, nullptr, nullptr,
        mach1_gdn_proj_args{}, mach1_gdn_convg_args{}, mach1_gdn_seq_args{});
    });

    return 15 - 1;
}

// Batched-decode GDN prep fusion (GGML_MACH1_GDN_PREP=1; ns >= 2). The core
// fusion above cannot run at B > 1 (its GATHER/state form is single-seq), and
// under the indexed-state graph (GGML_MACH1_GDN_SGF) the full-region matcher
// has no get_rows to enter at - so at B16 the six tiny pre-chain nodes
//   l2_norm(q_conv), l2_norm(k_conv), sigmoid(beta),
//   add(alpha, dt), softplus, mul(ssm_a)
// run as individual launches, 30 layers deep, per step. This fuse matches the
// 11-node window [l2q .. sigmoid] positionally (the same layout the core
// matcher proves at ns == 1), writes ONLY the four dsts the downstream
// gated_delta_net reads (l2q, l2k, gate mul, beta sigmoid), each with the
// stock kernel's exact op order - l2_norm_f32<WARP_SIZE>'s serial column
// stride and rsqrtf(fmaxf(sum, eps*eps)), op_sigmoid / op_softplus formulas,
// plain add/mul - and leaves the delta-rule node itself to its own (indexed)
// launch. Bitwise by construction; no state, no aliasing, no hazards beyond
// the stock nodes' own.
struct mach1_gdn_prep_plan {
    const ggml_tensor * conv;    // fused conv(+silu) output feeding the q/k views
    const ggml_tensor * alpha;   // raw alpha projection [Hv, ns]
    const ggml_tensor * dt;      // ssm_dt [Hv]
    const ggml_tensor * ga;      // ssm_a  [Hv]
    const ggml_tensor * beta;    // raw beta projection [Hv, ns]
    ggml_tensor * l2q;           // dsts this launch must materialize
    ggml_tensor * l2k;
    ggml_tensor * gm;
    ggml_tensor * sig;
    int64_t S, Hk, Hv, ns;
    float   eps_l2;
};

static bool mach1_gdn_match_prep(const ggml_cgraph * cgraph, int node_idx, mach1_gdn_prep_plan & p) {
    const int len = 11;   // l2_norm(q) .. sigmoid; the gdn node anchors at +11
    if (node_idx + len + 1 > cgraph->n_nodes) {
        return false;
    }

    ggml_tensor * l2q = cgraph->nodes[node_idx +  0];
    ggml_tensor * vk  = cgraph->nodes[node_idx +  1];
    ggml_tensor * l2k = cgraph->nodes[node_idx +  2];
    ggml_tensor * vv  = cgraph->nodes[node_idx +  3];
    ggml_tensor * ra  = cgraph->nodes[node_idx +  4];
    ggml_tensor * add = cgraph->nodes[node_idx +  5];
    ggml_tensor * sp  = cgraph->nodes[node_idx +  6];
    ggml_tensor * gm  = cgraph->nodes[node_idx +  7];
    ggml_tensor * rg  = cgraph->nodes[node_idx +  8];
    ggml_tensor * rb  = cgraph->nodes[node_idx +  9];
    ggml_tensor * sig = cgraph->nodes[node_idx + 10];
    ggml_tensor * gdn = cgraph->nodes[node_idx + 11];

    if (l2q->op != GGML_OP_L2_NORM || vk->op != GGML_OP_VIEW || l2k->op != GGML_OP_L2_NORM ||
        vv->op != GGML_OP_VIEW || ra->op != GGML_OP_RESHAPE || add->op != GGML_OP_ADD ||
        sp->op != GGML_OP_UNARY || gm->op != GGML_OP_MUL || rg->op != GGML_OP_RESHAPE ||
        rb->op != GGML_OP_RESHAPE || sig->op != GGML_OP_UNARY || gdn->op != GGML_OP_GATED_DELTA_NET) {
        return false;
    }
    if (ggml_get_unary_op(sp) != GGML_UNARY_OP_SOFTPLUS || ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID) {
        return false;
    }

    const ggml_tensor * vq   = l2q->src[0];
    const ggml_tensor * conv = vq->op == GGML_OP_VIEW ? vq->src[0] : nullptr;
    if (conv == nullptr || conv->type != GGML_TYPE_F32 || !ggml_is_contiguous(conv) ||
        conv->ne[1] != 1) {
        return false;
    }
    const int64_t S  = vq->ne[0];
    const int64_t Hk = vq->ne[1];
    const int64_t Hv = vv->ne[1];
    const int64_t ns = vq->ne[3];
    if (S != 128 || Hk <= 0 || Hv <= 0 || Hv % Hk != 0) {
        return false;
    }
    if (vq->ne[2] != 1 || ns < 2 || ns > (mach1_nt32_on() ? 32 : 16)) {   // ns == 1 belongs to the core fuse
        return false;
    }
    if (conv->ne[0] != 2*S*Hk + S*Hv || conv->ne[2] != ns) {
        return false;
    }
    auto head_view_ok = [&](const ggml_tensor * v, int64_t heads, size_t offs) {
        return v->type == GGML_TYPE_F32 && v->src[0] == conv && v->view_offs == offs &&
               v->ne[0] == S && v->ne[1] == heads && v->ne[2] == 1 && v->ne[3] == ns &&
               v->nb[0] == sizeof(float) && v->nb[1] == S*sizeof(float) && v->nb[2] == conv->nb[1] &&
               v->nb[3] == conv->nb[2];
    };
    if (!head_view_ok(vq, Hk, 0) || !head_view_ok(vk, Hk, S*Hk*sizeof(float)) ||
        !head_view_ok(vv, Hv, 2*S*Hk*sizeof(float))) {
        return false;
    }
    if (l2k->src[0] != vk || l2q->type != GGML_TYPE_F32 || l2k->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(l2q) || !ggml_is_contiguous(l2k)) {
        return false;
    }
    const float eps_l2 = ggml_get_op_params_f32(l2q, 0);
    if (ggml_get_op_params_f32(l2k, 0) != eps_l2) {
        return false;
    }

    const ggml_tensor * alpha = ra->src[0];
    if (alpha->type != GGML_TYPE_F32 || !ggml_is_contiguous(alpha) ||
        alpha->ne[0] != Hv || ggml_nrows(alpha) != ns) {
        return false;
    }
    const ggml_tensor * dt = add->src[1];
    if (add->src[0] != ra || add->type != GGML_TYPE_F32 ||
        dt->type != GGML_TYPE_F32 || !ggml_is_contiguous(dt) || ggml_nelements(dt) != Hv) {
        return false;
    }
    const ggml_tensor * ga = gm->src[1];
    if (sp->src[0] != add || gm->src[0] != sp || gm->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(gm) ||
        ga->type != GGML_TYPE_F32 || !ggml_is_contiguous(ga) || ggml_nelements(ga) != Hv) {
        return false;
    }
    if (rg->src[0] != gm || rg->ne[0] != 1 || rg->ne[1] != Hv || rg->ne[3] != ns) {
        return false;
    }
    const ggml_tensor * beta = rb->src[0];
    if (beta->type != GGML_TYPE_F32 || !ggml_is_contiguous(beta) ||
        beta->ne[0] != Hv || ggml_nrows(beta) != ns) {
        return false;
    }
    if (rb->ne[0] != 1 || rb->ne[1] != Hv || rb->ne[3] != ns ||
        sig->src[0] != rb || sig->type != GGML_TYPE_F32 || !ggml_is_contiguous(sig)) {
        return false;
    }

    // the delta-rule node must be the sole consumer of everything we elide
    if (gdn->src[0] != l2q || gdn->src[1] != l2k || gdn->src[2] != vv ||
        gdn->src[3] != rg || gdn->src[4] != sig || gdn->type != GGML_TYPE_F32) {
        return false;
    }

    // no swallowed node may be a graph output or escape the window (+ gdn)
    for (int j = node_idx; j < node_idx + len; ++j) {
        const ggml_tensor * nn = cgraph->nodes[j];
        if ((nn->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;
        }
        if (nn->flags & GGML_TENSOR_FLAG_OUTPUT) {
            return false;
        }
        int uses = 0;
        for (int j2 = node_idx; j2 < node_idx + len + 1; ++j2) {
            for (int si = 0; si < GGML_MAX_SRC; ++si) {
                if (cgraph->nodes[j2]->src[si] == nn) {
                    uses++;
                }
            }
        }
        if (uses != ggml_node_get_use_count(cgraph, j)) {
            return false;
        }
    }

    p = { conv, alpha, dt, ga, beta, l2q, l2k, gm, sig, S, Hk, Hv, ns, eps_l2 };
    return true;
}

// One warp per l2 row (blockIdx.x < 2*Hk: q rows then k rows), plus one
// trailing block per seq for the elementwise gate/beta chain. Each write
// replicates the stock kernel's op order exactly (see the comment above).
template <int S>
__global__ void mach1_gdn_prep_kernel(
        const float * __restrict__ conv,     // [C, 1, ns] fused conv+silu output
        const float * __restrict__ alpha,    // [Hv, ns]
        const float * __restrict__ dt,       // [Hv]
        const float * __restrict__ ga,       // [Hv]
        const float * __restrict__ beta,     // [Hv, ns]
        float       * __restrict__ l2q_dst,  // [S, Hk, 1, ns] contiguous
        float       * __restrict__ l2k_dst,  // [S, Hk, 1, ns] contiguous
        float       * __restrict__ gm_dst,   // [Hv, 1, ns] contiguous
        float       * __restrict__ sig_dst,  // [1, Hv, 1, ns] contiguous
        const int Hk, const int Hv, const float eps_l2) {
    constexpr int warp_size = 32;
    const int seq  = blockIdx.y;
    const int bx   = blockIdx.x;
    const int lane = threadIdx.x;
    const int64_t C = (int64_t) 2*S*Hk + (int64_t) S*Hv;

    if (bx < 2*Hk) {
        // l2_norm_f32<WARP_SIZE> body for one (q|k, head, seq) row
        const bool  is_k = bx >= Hk;
        const int   h    = is_k ? bx - Hk : bx;
        const float * x  = conv + (int64_t) seq*C + (is_k ? (int64_t) Hk*S : 0) + (int64_t) h*S;
        float * dst = (is_k ? l2k_dst : l2q_dst) + ((int64_t) seq*Hk + h)*S;
        float tmp = 0.0f;
        for (int col = lane; col < S; col += warp_size) {
            const float xi = x[col];
            tmp += xi*xi;
        }
        tmp = warp_reduce_sum(tmp);
        const float scale = rsqrtf(fmaxf(tmp, eps_l2*eps_l2));
        for (int col = lane; col < S; col += warp_size) {
            dst[col] = scale*x[col];
        }
        return;
    }

    // elementwise chain for this seq: b = sigmoid(beta), g = softplus(alpha+dt)*ssm_a
    for (int h = lane; h < Hv; h += warp_size) {
        const int64_t i = (int64_t) seq*Hv + h;
        sig_dst[i] = 1.0f / (1.0f + expf(-beta[i]));
        const float ab = alpha[i] + dt[h];
        const float sv = ab > 20.0f ? ab : logf(1.0f + expf(ab));
        gm_dst[i] = sv*ga[h];
    }
}

int ggml_cuda_mach1_gdn_prep_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    static const bool on = mach1_env_int("GGML_MACH1_GDN_PREP", 0) != 0;
    mach1_gdn_prep_plan p;
    if (!on || !mach1_graph_has_mach1_op(cgraph) || !mach1_gdn_match_prep(cgraph, node_idx, p)) {
        return 0;
    }

    cudaStream_t stream = ctx.stream();

    if (mach1_debug_on()) {
        static std::once_flag once;
        std::call_once(once, []() {
            fprintf(stderr, "mach1: p16 GDN prep chain fused ENGAGED (11-node window, 6 launches -> 1)\n");
        });
    }

    char shp[64];
    snprintf(shp, sizeof(shp), " Hk=%d Hv=%d nt=%d", (int) p.Hk, (int) p.Hv, (int) p.ns);
    mach1_timed(stream, std::string("gdn_prep") + shp, [&]() {
    mach1_launch(mach1_gdn_prep_kernel<128>,
        ggml_cuda_kernel_launch_params(dim3(2*p.Hk + 1, p.ns, 1), dim3(32, 1, 1), 0, stream),
        (const float *) p.conv->data, (const float *) p.alpha->data, (const float *) p.dt->data,
        (const float *) p.ga->data, (const float *) p.beta->data,
        (float *) p.l2q->data, (float *) p.l2k->data,
        (float *) p.gm->data, (float *) p.sig->data,
        (int) p.Hk, (int) p.Hv, p.eps_l2);
    });

    return 11 - 1;
}

// GDN_FULL level 3: the conv-state shift the graph's CPY performed - taps
// 1..2 slide down and the fresh qkv value enters last. Launched AFTER the
// core kernel on the stream, so every conv read of the old row has retired.
__global__ void mach1_gdn_convup_kernel(
        float       * __restrict__ dstrow,   // the update view's row [C*3]
        const float * __restrict__ qkv,      // [C]
        const int C) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= C) {
        return;
    }
    float * r = dstrow + (int64_t) c*3;
    const float t1 = r[1];
    const float t2 = r[2];
    r[0] = t1;
    r[1] = t2;
    r[2] = qkv[c];
}

// Full GDN glue fusion (GGML_MACH1_GDN_FULL != 0; decode only). Entered at the
// recurrent-state get_rows, the region runs to the state-cache cpy:
//
//   get_rows(ssm_states_all, s_copy_main)      the 2 MB/layer round trip
//   get_rows(extra) + cpy                      empty at n_rs == n_seqs
//   ssm_conv + silu (stock fused pair)
//   the 15-node GDN core above
//
// The nodes between them are views/reshapes or empty, i.e. exactly what the
// dispatch loop skips anyway, so swallowing them changes nothing. Level 1
// issues the conv pair through its own op (one launch, unchanged bits) and
// only elides the gather; level 2 folds the conv into the kernel as well.
struct mach1_gdn_full_m {
    const ggml_tensor * gr = nullptr;
    ggml_tensor * cv = nullptr;
    ggml_tensor * si = nullptr;
    int core = 0;
    mach1_gdn_plan p;
    bool conv_in_kernel = false;
};

static bool mach1_gdn_full_match(const ggml_cgraph * cgraph, int node_idx, int lvl, mach1_gdn_full_m & mm) {
    const ggml_tensor * gr  = cgraph->nodes[node_idx];
    const ggml_tensor * ids = gr->src[1];
    const ggml_tensor * sta = gr->src[0];
    if (gr->op != GGML_OP_GET_ROWS || gr->type != GGML_TYPE_F32 || !ggml_is_contiguous(gr) ||
        gr->ne[1] < 1 || gr->ne[1] > 8 || gr->ne[2] != 1 || gr->ne[3] != 1) {
        return false;
    }
    if (ids->type != GGML_TYPE_I32 || ggml_nelements(ids) != gr->ne[1] || ids->data == nullptr) {
        return false;
    }
    if (sta->type != GGML_TYPE_F32 || !ggml_is_contiguous(sta) || sta->data == nullptr ||
        sta->ne[0] != gr->ne[0] || sta->nb[1] != (size_t) sta->ne[0]*sizeof(float)) {
        return false;
    }

    // the conv pair, past whatever views/empties the state build_rs left behind
    auto skip_noops = [&](int from) {
        int n = from;
        while (n < cgraph->n_nodes && n < from + 8 && mach1_node_is_noop(cgraph->nodes[n])) {
            ++n;
        }
        return n;
    };

    const int j = skip_noops(node_idx + 1);
    if (j + 2 > cgraph->n_nodes) {
        return false;
    }
    ggml_tensor * cv = cgraph->nodes[j];
    ggml_tensor * si = cgraph->nodes[j + 1];
    if (cv->op != GGML_OP_SSM_CONV || si->op != GGML_OP_UNARY ||
        ggml_get_unary_op(si) != GGML_UNARY_OP_SILU || si->src[0] != cv ||
        cv->type != GGML_TYPE_F32 || si->type != GGML_TYPE_F32) {
        return false;
    }

    const int core = skip_noops(j + 2);   // the q_conv view sits between
    mach1_gdn_plan p;
    if (!mach1_gdn_match_core(cgraph, core, p)) {
        return false;
    }
    if (p.conv != si || p.state->view_src != gr || p.state->data != gr->data) {
        return false;
    }
    if (p.ns != gr->ne[1]) {   // one gathered state row per decode seq
        return false;
    }
    if (p.ns > mach1_gdn_nt_cap()) {
        return false;
    }

    // every non-view node the region swallows must be consumed only inside it
    for (int n = node_idx; n < core; ++n) {
        const ggml_tensor * nn = cgraph->nodes[n];
        if (mach1_node_is_noop(nn)) {
            continue;
        }
        if ((nn->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || (nn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return false;
        }
        if (nn != gr && nn != cv && nn != si) {
            return false;
        }
        int uses = 0;
        for (int j2 = node_idx; j2 < core + 15; ++j2) {
            for (int si2 = 0; si2 < GGML_MAX_SRC; ++si2) {
                if (cgraph->nodes[j2]->src[si2] == nn) {
                    uses++;
                }
            }
        }
        if (uses != ggml_node_get_use_count(cgraph, n)) {
            return false;
        }
    }

    const ggml_tensor * cx = cv->src[0];
    const ggml_tensor * cw = cv->src[1];
    const int64_t C  = p.conv->ne[0];
    const int64_t DC = cw->ne[0];
    mm.gr   = gr;
    mm.cv   = cv;
    mm.si   = si;
    mm.core = core;
    mm.p    = p;
    mm.conv_in_kernel = lvl >= 2 && DC == 4 &&
        ggml_is_contiguous(cx) && ggml_is_contiguous(cw) &&
        cx->type == GGML_TYPE_F32 && cw->type == GGML_TYPE_F32 &&
        cx->ne[0] == DC && cx->ne[1] == C && cx->ne[2] == p.ns && cw->ne[1] == C &&
        cv->ne[0] == C && cv->ne[1] == 1 && cv->ne[2] == p.ns;
    // conv_x dies at the swallowed ssm_conv in the ALLOCATOR's view, so the
    // gdn out tensor may legally overlay its bytes while the fused kernel is
    // still reading them - the r256 aliasing lesson. Degrade to the issued
    // conv pair (which writes si before the kernel reads it) if they touch.
    if (mm.conv_in_kernel) {
        const char * a0 = (const char *) p.gdn->data;
        const char * a1 = a0 + ggml_nbytes(p.gdn);
        const char * b0 = (const char *) cx->data;
        const char * b1 = b0 + ggml_nbytes(cx);
        if (a0 < b1 && b0 < a1) {
            mm.conv_in_kernel = false;
        }
    }
    return true;
}

// FORK_Z probe: does the gdn_full region engage ahead of `from` in this
// cgraph? Deterministic against the same graph the backend will execute, so
// an armed z fork is guaranteed its join at that region's exec.
static bool mach1_gdn_full_ahead(const ggml_cgraph * cgraph, int from) {
    const int lvl = mach1_gdn_full_level();
    if (lvl == 0 || mach1_env_int("GGML_MACH1_GDN_FUSE", 1) == 0) {
        return false;
    }
    for (int k = from; k < cgraph->n_nodes && k < from + 40; ++k) {
        if (cgraph->nodes[k]->op == GGML_OP_GET_ROWS) {
            mach1_gdn_full_m mm;
            if (mach1_gdn_full_match(cgraph, k, lvl, mm)) {
                return true;
            }
        }
    }
    return false;
}

int ggml_cuda_mach1_gdn_full_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    const int lvl = mach1_gdn_full_level();
    if (lvl == 0 || mach1_env_int("GGML_MACH1_GDN_FUSE", 1) == 0 || !mach1_graph_has_mach1_op(cgraph)) {
        return 0;
    }

    mach1_gdn_full_m mm;
    if (!mach1_gdn_full_match(cgraph, node_idx, lvl, mm)) {
        return 0;
    }
    // GGML_MACH1_DEBUG=3: one-shot dump of the nodes AROUND the region entry,
    // to see where the beta/alpha [2048,32] matvec producers sit for the
    // backward-extension matcher (the router fold recon)
    if (mach1_env_int("GGML_MACH1_DEBUG", 0) >= 3) {
        static bool once = false;
        if (!once) {
            once = true;
            for (int k = std::max(0, node_idx - 24); k < std::min(cgraph->n_nodes, node_idx + 3); ++k) {
                const ggml_tensor * t = cgraph->nodes[k];
                fprintf(stderr, "mach1-gdnprev: [%+d] %-12s %-28s ne=%lld,%lld,%lld src0=%s\n",
                        k - node_idx, ggml_op_name(t->op), t->name,
                        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
                        t->src[0] ? t->src[0]->name : "-");
            }
        }
    }
    const ggml_tensor * gr  = mm.gr;
    const ggml_tensor * ids = gr->src[1];
    const ggml_tensor * sta = gr->src[0];
    ggml_tensor * cv = mm.cv;
    ggml_tensor * si = mm.si;
    const int core = mm.core;
    mach1_gdn_plan p = mm.p;
    const ggml_tensor * cx = cv->src[0];
    const ggml_tensor * cw = cv->src[1];
    const bool conv_in_kernel = mm.conv_in_kernel;

    cudaStream_t stream = ctx.stream();

    if (!conv_in_kernel) {
        ggml_cuda_op_ssm_conv(ctx, cv, /*bias_add_node =*/ nullptr, si);
    }

    const int64_t ns = p.ns;
    if (ns > 1) {
        // batched decode: the state arrives through the graph's own gather
        // (folding it is unsafe across seqs - s_copy may name another seq's
        // scatter row after a slot shuffle, and blocks of different seqs
        // have no ordering), run here since the region swallowed its node
        ggml_cuda_op_get_rows(ctx, (ggml_tensor *) gr);
    }

    char shp[80];
    snprintf(shp, sizeof(shp), " Hk=%d Hv=%d nt=%d", (int) p.Hk, (int) p.Hv, (int) ns);
    mach1_timed(stream, std::string(conv_in_kernel ? "gdn_full_conv" : "gdn_full") + shp, [&]() {
    if (ns > 1) {
        mach1_gdn_seq_args sa;
        sa.ab_ss = (int64_t) (p.alpha->nb[1]/sizeof(float));
        sa.st_ss = (int64_t) (gr->nb[1]/sizeof(float));
        sa.at_ss = p.S*p.Hv;
        sa.so_ss = (int64_t) (p.vcd->nb[1]/sizeof(float));
        if (conv_in_kernel) {
            constexpr int CGY = 8;
            sa.cx_ss = (int64_t) (cx->nb[2]/sizeof(float));
            mach1_launch(mach1_gdn_core_kernel<128, CGY, 0, 1>,
                ggml_cuda_kernel_launch_params(dim3(p.Hv, ns, 128/CGY), dim3(32, CGY, 1), 0, stream),
                nullptr, (const float *) p.alpha->data, (const float *) p.dt->data,
                (const float *) p.ga->data, (const float *) p.beta->data,
                (const float *) gr->data,
                (float *) p.gdn->data, (float *) p.vcd->data,
                (int) p.Hk, p.eps_l2, 1.0f / sqrtf((float) p.S),
                nullptr, nullptr, 0,
                (const float *) cx->data, (const float *) cw->data, nullptr,
                mach1_gdn_proj_args{}, mach1_gdn_convg_args{}, sa);
        } else {
            sa.conv_ss = (int64_t) (p.conv->nb[2]/sizeof(float));
            mach1_launch(mach1_gdn_core_kernel<128, 4, 0, 0>,
                ggml_cuda_kernel_launch_params(dim3(p.Hv, ns, 128/4), dim3(32, 4, 1), 0, stream),
                (const float *) p.conv->data, (const float *) p.alpha->data, (const float *) p.dt->data,
                (const float *) p.ga->data, (const float *) p.beta->data,
                (const float *) gr->data,
                (float *) p.gdn->data, (float *) p.vcd->data,
                (int) p.Hk, p.eps_l2, 1.0f / sqrtf((float) p.S),
                nullptr, nullptr, 0,
                nullptr, nullptr, nullptr,
                mach1_gdn_proj_args{}, mach1_gdn_convg_args{}, sa);
        }
    } else if (conv_in_kernel) {
        constexpr int CGY = 8;
        mach1_launch(mach1_gdn_core_kernel<128, CGY, 1, 1>,
            ggml_cuda_kernel_launch_params(dim3(p.Hv, 1, 128/CGY), dim3(32, CGY, 1), 0, stream),
            nullptr, (const float *) p.alpha->data, (const float *) p.dt->data,
            (const float *) p.ga->data, (const float *) p.beta->data, nullptr,
            (float *) p.gdn->data, (float *) p.vcd->data,
            (int) p.Hk, p.eps_l2, 1.0f / sqrtf((float) p.S),
            (const float *) sta->data, (const int *) ids->data, sta->ne[0],
            (const float *) cx->data, (const float *) cw->data, nullptr,
            mach1_gdn_proj_args{}, mach1_gdn_convg_args{}, mach1_gdn_seq_args{});
    } else {
        mach1_launch(mach1_gdn_core_kernel<128, 4, 1, 0>,
            ggml_cuda_kernel_launch_params(dim3(p.Hv, 1, 128/4), dim3(32, 4, 1), 0, stream),
            (const float *) p.conv->data, (const float *) p.alpha->data, (const float *) p.dt->data,
            (const float *) p.ga->data, (const float *) p.beta->data, nullptr,
            (float *) p.gdn->data, (float *) p.vcd->data,
            (int) p.Hk, p.eps_l2, 1.0f / sqrtf((float) p.S),
            (const float *) sta->data, (const int *) ids->data, sta->ne[0],
            nullptr, nullptr, nullptr,
            mach1_gdn_proj_args{}, mach1_gdn_convg_args{}, mach1_gdn_seq_args{});
    }
    });

    // FORK_Z join: the z walk ran on s2 under this region; its consumer (the
    // gated norm) is main's next node class, so the join lands here - after
    // the core launch, keeping the core itself concurrent with the walk
    if (mach1_fork_z_on() || mach1_fork_z_nt() > 0) {
        mach1_fork_state * f = mach1_fork_get(ctx.device, stream, 0);
        if (f != nullptr && f->z_armed) {
            f->z_armed = false;
            CUDA_CHECK(cudaStreamWaitEvent(stream, f->ez, 0));
        }
    }

    return (core + 15) - node_idx - 1;
}

// GDN_FULL level 3 (GGML_MACH1_GDN_FULL=3; decode only): the region extends
// BACKWARDS over the beta/alpha projections and the conv-state round trip.
// Entered at the beta mul_mat; the glue window (19 nodes, or 17 in form B;
// recon round 249) is matched positionally, the wiring verified by POINTER
// identity against the core match, and the whole thing runs as the PROJ
// core kernel plus the 3-tap shift. The projection dots are ulp-class vs
// the mul_mat kernel: certification is the KLD gate, never smoke.
int ggml_cuda_mach1_gdn_proj_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    const int lvl = mach1_gdn_full_level();
    if (lvl < 3 || mach1_env_int("GGML_MACH1_GDN_FUSE", 1) == 0 || !mach1_graph_has_mach1_op(cgraph)) {
        return 0;
    }
    if (node_idx + 19 + 15 > cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * mmb = cgraph->nodes[node_idx];
    const ggml_tensor * mma = cgraph->nodes[node_idx + 1];
    if (mmb->op != GGML_OP_MUL_MAT || mma->op != GGML_OP_MUL_MAT ||
        mmb->src[1] == nullptr || mmb->src[1] != mma->src[1] ||
        mmb->ne[1] != 1 || mma->ne[1] != 1 || mmb->ne[0] != mma->ne[0] ||
        mmb->ne[0] > 64 || mmb->type != GGML_TYPE_F32 || mma->type != GGML_TYPE_F32) {
        return 0;
    }
    const ggml_tensor * wb = mmb->src[0];
    const ggml_tensor * wa = mma->src[0];
    const ggml_tensor * x  = mmb->src[1];
    if (wb->type != wa->type || (wb->type != GGML_TYPE_F32 && wb->type != GGML_TYPE_F16) ||
        !ggml_is_contiguous(wb) || !ggml_is_contiguous(wa) ||
        x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || ggml_nrows(x) != 1) {
        return 0;
    }

    // the glue window, positionally (round-249 recon); the two SCALEs must be
    // in-place no-ops (steady decode) - a reset eval falls back unfused
    // TWO glue windows exist and only form A was ever matched, so this fused
    // exactly ONE GDN layer per token and missed the other 29 (round 318-319,
    // counted: form A 123 calls, form B 3567). Form B is form A with the VIEW
    // nodes at positions 3 and 5 absent - everything after shifts left by two,
    // so it runs 17 nodes and the core begins two earlier.
    static const ggml_op win_a[19] = {
        GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_SCALE, GGML_OP_VIEW, GGML_OP_GET_ROWS,
        GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_VIEW, GGML_OP_CPY, GGML_OP_RESHAPE,
        GGML_OP_RESHAPE, GGML_OP_TRANSPOSE, GGML_OP_CONCAT, GGML_OP_VIEW, GGML_OP_VIEW,
        GGML_OP_CPY, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_SCALE };
    static const ggml_op win_b[17] = {
        GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_SCALE, GGML_OP_GET_ROWS, GGML_OP_GET_ROWS,
        GGML_OP_VIEW, GGML_OP_CPY, GGML_OP_RESHAPE, GGML_OP_RESHAPE, GGML_OP_TRANSPOSE,
        GGML_OP_CONCAT, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_CPY, GGML_OP_RESHAPE,
        GGML_OP_VIEW, GGML_OP_SCALE };
    struct wcand { const ggml_op * ops; int len; };
    const wcand cands[2] = { { win_a, 19 }, { win_b, 17 } };
    int wlen = 0;
    for (int c = 0; c < 2 && wlen == 0; ++c) {
        if (node_idx + 2 + cands[c].len + 15 > cgraph->n_nodes) {
            continue;
        }
        bool hit = true;
        for (int k = 0; k < cands[c].len && hit; ++k) {
            hit = cgraph->nodes[node_idx + 2 + k]->op == cands[c].ops[k];
        }
        if (hit) {
            wlen = cands[c].len;
        }
    }
    if (wlen == 0) {
        return 0;
    }
    // every positional read below, expressed against whichever window matched
    const bool wa19  = wlen == 19;
    const int  o_sc1 = 4;
    const int  o_gr2 = wa19 ?  6 :  5;
    const int  o_gre = wa19 ?  8 :  6;
    const int  o_cpe = wa19 ? 10 :  8;
    const int  o_cat = wa19 ? 14 : 12;
    const int  o_upd = wa19 ? 17 : 15;
    const int  o_sc2 = wa19 ? 20 : 18;
    const int  o_core = 2 + wlen;
    const ggml_tensor * sc1 = cgraph->nodes[node_idx + o_sc1];
    const ggml_tensor * gr2 = cgraph->nodes[node_idx + o_gr2];
    const ggml_tensor * gre = cgraph->nodes[node_idx + o_gre];
    const ggml_tensor * cpe = cgraph->nodes[node_idx + o_cpe];
    const ggml_tensor * cat = cgraph->nodes[node_idx + o_cat];
    const ggml_tensor * upd = cgraph->nodes[node_idx + o_upd];
    const ggml_tensor * sc2 = cgraph->nodes[node_idx + o_sc2];
    float s1v = 0.0f, s2v = 0.0f;
    memcpy(&s1v, sc1->op_params, sizeof(float));
    memcpy(&s2v, sc2->op_params, sizeof(float));
    if (s1v != 1.0f || s2v != 1.0f ||
        sc1->data != sc1->src[0]->data || sc2->data != sc2->src[0]->data) {
        mach1_diag_note("g3bail_scale");
        return 0;   // a reset eval: run unfused
    }
    if (gre->ne[1] != 0 || cpe->ne[1] != 0) {
        mach1_diag_note("g3bail_nrs");
        return 0;   // the n_rs > n_seqs shape: run unfused
    }

    // the core region past the window - reuse the level-2 matcher, then pin
    // every pointer this fold substitutes
    mach1_gdn_full_m mm;
    if (!mach1_gdn_full_match(cgraph, node_idx + o_core, lvl, mm)) {
        mach1_diag_note("g3bail_corematch");
        return 0;
    }
    if (!mm.conv_in_kernel) {
        mach1_diag_note("g3bail_convinkernel");
        return 0;
    }
    if (mm.p.beta != mmb || mm.p.alpha != mma) {
        mach1_diag_note("g3bail_betaalpha");
        return 0;
    }
    const ggml_tensor * cx = mm.cv->src[0];
    const ggml_tensor * cw = mm.cv->src[1];
    if (cx != cat || cat->op != GGML_OP_CONCAT ||
        cat->src[0]->op != GGML_OP_RESHAPE || cat->src[0]->src[0] != gr2 ||
        cat->src[1]->op != GGML_OP_TRANSPOSE ||
        gr2->src[0]->type != GGML_TYPE_F32 || gr2->src[1]->type != GGML_TYPE_I32) {
        mach1_diag_note("g3bail_catchain");
        return 0;
    }
    // the update CPY: dst aliases the conv cache row, source is the tap-1..3
    // view of the concat - the shift kernel reproduces it exactly
    if (upd->src[0] == nullptr || upd->src[0]->op != GGML_OP_VIEW ||
        upd->src[0]->src[0] != cat || upd->data == nullptr) {
        mach1_diag_note("g3bail_upd");
        return 0;
    }
    const int64_t C = cat->ne[1];
    if (cat->ne[0] != 4 || upd->ne[0] != 3*C || 2*mm.p.S*mm.p.Hk + mm.p.S*mm.p.Hv != C) {
        mach1_diag_note("g3bail_shape");
        return 0;
    }

    // swallowed real nodes must be consumed only inside the span
    const int span_end = node_idx + o_core + 15;
    for (int nn = node_idx; nn < node_idx + o_core; ++nn) {
        const ggml_tensor * t = cgraph->nodes[nn];
        if (mach1_node_is_noop(t) || t == sc1 || t == sc2 || t == gre || t == cpe) {
            continue;
        }
        if ((t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            mach1_diag_note("g3bail_flags");
            return 0;
        }
        int uses = 0;
        for (int j2 = node_idx; j2 < span_end; ++j2) {
            for (int si2 = 0; si2 < GGML_MAX_SRC; ++si2) {
                if (cgraph->nodes[j2]->src[si2] == t) {
                    uses++;
                }
            }
        }
        if (uses != ggml_node_get_use_count(cgraph, nn)) {
            mach1_diag_note("g3bail_usecount");
            return 0;
        }
    }

    const ggml_tensor * gr  = mm.gr;
    const ggml_tensor * ids = gr->src[1];
    const ggml_tensor * sta = gr->src[0];
    mach1_gdn_plan p = mm.p;

    cudaStream_t stream = ctx.stream();

    mach1_gdn_proj_args pa = {};
    pa.x     = (const float *) x->data;
    pa.wb    = wb->data;
    pa.wa    = wa->data;
    pa.n     = (int) wb->ne[0];
    pa.wtype = wb->type == GGML_TYPE_F16 ? 1 : 0;
    mach1_gdn_convg_args ca = {};
    ca.crow_base   = (const float *) gr2->src[0]->data;
    ca.cidx        = (const int *) gr2->src[1]->data;
    ca.crow_stride = (int64_t)(gr2->src[0]->nb[1]/sizeof(float));
    ca.qkv         = (const float *) cat->src[1]->data;

    char shp[64];
    snprintf(shp, sizeof(shp), " Hk=%d Hv=%d", (int) p.Hk, (int) p.Hv);
    mach1_timed(stream, std::string("gdn_full3") + shp, [&]() {
    constexpr int CGY = 8;
    mach1_launch(mach1_gdn_core_kernel<128, CGY, 1, 1, 4, 1>,
        ggml_cuda_kernel_launch_params(dim3(p.Hv, 1, 128/CGY), dim3(32, CGY, 1), 0, stream),
        nullptr, nullptr, (const float *) p.dt->data,
        (const float *) p.ga->data, nullptr, nullptr,
        (float *) p.gdn->data, (float *) p.vcd->data,
        (int) p.Hk, p.eps_l2, 1.0f / sqrtf((float) p.S),
        (const float *) sta->data, (const int *) ids->data, sta->ne[0],
        nullptr, (const float *) cw->data, nullptr, pa, ca, mach1_gdn_seq_args{});
    mach1_launch(mach1_gdn_convup_kernel,
        ggml_cuda_kernel_launch_params(dim3((unsigned)((C + 255)/256), 1, 1), dim3(256, 1, 1), 0, stream),
        (float *) upd->data, ca.qkv, (int) C);
    });

    // FORK_Z safety: this exec replaces gdn_full's, so the armed join (if the
    // rejected lever is ever re-enabled) must land here too
    if (mach1_fork_z_on() || mach1_fork_z_nt() > 0) {
        mach1_fork_state * f = mach1_fork_get(ctx.device, stream, 0);
        if (f != nullptr && f->z_armed) {
            f->z_armed = false;
            CUDA_CHECK(cudaStreamWaitEvent(stream, f->ez, 0));
        }
    }

    return span_end - node_idx - 1;
}

//
// supports_op — mirrors the Vulkan predicate (ggml-vulkan.cpp)
//

bool ggml_cuda_mach1_supported(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_MACH1_EXP_MM:
            // walk/out kernels stage u and H(v) in fixed shared arrays of 2048
            // floats; the group kernel keeps per-group tables in shared arrays
            // of 512; V=8 (payload v3) requires wave_gamma and vice versa. The
            // two walk variants also fix the tlut storage type: V=8 reads the
            // pre-rounded F16 table, V=2 the F32 one.
            return op->src[2]->ne[0] <= 2048 && op->src[3]->ne[0] <= 2048 &&
                   op->src[0]->ne[2] + (op->src[1] ? op->src[1]->ne[2] : 0) <= 512 &&
                   (op->src[4]->ne[0] == 2) == (op->src[8] == nullptr) &&
                   op->src[4]->type == (op->src[4]->ne[0] == 2 ? GGML_TYPE_F32
                                                               : GGML_TYPE_F16);
        case GGML_OP_MACH1_RT_MM:
            // rt_u stages n floats (<= 4096) and rt_out m floats (<= 8192 =
            // 32 KB static shared) per block
            return op->src[1]->ne[0] <= 4096 && op->src[2]->ne[0] <= 8192 &&
                   op->src[3]->type == GGML_TYPE_F16;
        case GGML_OP_MACH1_NE_MM:
        case GGML_OP_MACH1_EMBED_ROWS:
        case GGML_OP_MACH1_EXP_BASIS:
        case GGML_OP_MACH1_HEAD_MM:
        case GGML_OP_MACH1_EMBED_GATHER:
            // shapes/types are enforced by the ggml builders
            return true;
        default:
            return false;
    }
}
