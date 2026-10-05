
#pragma once

static __device__ __forceinline__ void mach1_pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

static __device__ __forceinline__ void mach1_pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

template <typename T>
static __device__ __forceinline__ T * mach1_pdl_dep(T * p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("" : "+l"(p));
#endif
    return p;
}

#if !defined(MACH1_RT_EMU) && !defined(MACH1_D4_EMU)
static bool mach1_pdl_on() {
    static const bool on = [] {
        const char * s = getenv("GGML_MACH1_PDL");
        if (s != nullptr) {
            return atoi(s) != 0;
        }
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_HOPPER;
    }();
    return on;
}

static bool mach1_skip(int bit) {
    static const int mask = getenv("GGML_MACH1_SKIP") == nullptr ? 0 : atoi(getenv("GGML_MACH1_SKIP"));
    return (mask & bit) != 0;
}

template <typename Kernel, typename... Args>
static void mach1_launch_pdl_raw(Kernel kernel, const dim3 grid, const dim3 block, const size_t shmem,
                                 cudaStream_t stream, Args &&... args) {
#if defined(GGML_CUDA_USE_PDL)
    if (mach1_pdl_on()) {
        cudaLaunchAttribute attr = {};
        attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim          = grid;
        cfg.blockDim         = block;
        cfg.dynamicSmemBytes = shmem;
        cfg.stream           = stream;
        cfg.attrs            = &attr;
        cfg.numAttrs         = 1;
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...));
        return;
    }
#endif
    kernel<<<grid, block, shmem, stream>>>(std::forward<Args>(args)...);
    CUDA_CHECK(cudaGetLastError());
}
#endif
