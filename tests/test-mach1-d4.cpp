
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <d4_fixtures.gguf> [n_threads]\n", argv[0]);
        return 2;
    }
    const int n_threads = argc > 2 ? atoi(argv[2]) : 4;
    const char * ex = getenv("GGML_MACH1_D4_EXACT");
    const bool exact = ex && atoi(ex) != 0;
    const double tol_rms = exact ? 1e-5 : 1.5e-2;
    const double tol_max = exact ? 1e-4 : 1.5e-1;
    printf("walk: %s\n", exact ? "fp32 (exact)" : "default");

    ggml_context * ctx_data = NULL;
    gguf_init_params gp = {  false,  &ctx_data };
    gguf_context * gctx = gguf_init_from_file(argv[1], gp);
    if (!gctx) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 2;
    }
    auto get = [&](const std::string & name) {
        ggml_tensor * t = ggml_get_tensor(ctx_data, name.c_str());
        if (!t) {
            fprintf(stderr, "missing tensor %s\n", name.c_str());
            exit(2);
        }
        return t;
    };

    const int64_t kid = gguf_find_key(gctx, "cases");
    const size_t n_cases = gguf_get_arr_n(gctx, kid);
    int fails = 0;
    for (size_t ci = 0; ci < n_cases; ++ci) {
        const std::string c = gguf_get_arr_str(gctx, kid, ci);

        ggml_init_params ip = {  64u << 20,  NULL,  false };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * y = ggml_mach1_d4_mm(ctx, get(c + ".trellis"), get(c + ".offs"), get(c + ".su"), get(c + ".sv"),
                                           get(c + ".gw"), get("zt"), get("units"), get(c + ".ids"), get(c + ".x"), nullptr);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        if (ggml_graph_compute_with_ctx(ctx, gf, n_threads) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: compute failed\n", c.c_str());
            return 1;
        }
        const ggml_tensor * ref = get(c + ".y");
        if (!ggml_are_same_shape(y, ref)) {
            fprintf(stderr, "%s: shape mismatch\n", c.c_str());
            return 1;
        }
        const float * a = (const float *) y->data;
        const float * b = (const float *) ref->data;
        double err2 = 0.0, ref2 = 0.0, maxrel = 0.0;
        const int64_t m = ref->ne[0];
        for (int64_t r = 0; r < ggml_nrows(ref); ++r) {
            double rn = 0.0;
            for (int64_t i = 0; i < m; ++i) {
                rn += (double) b[r*m + i]*b[r*m + i];
            }
            rn = sqrt(rn/m);
            for (int64_t i = 0; i < m; ++i) {
                const double d = (double) a[r*m + i] - b[r*m + i];
                err2 += d*d;
                ref2 += (double) b[r*m + i]*b[r*m + i];
                maxrel = fmax(maxrel, fabs(d)/(rn + 1e-30));
            }
        }
        const double rel = sqrt(err2/(ref2 + 1e-30));
        const bool ok = rel < tol_rms && maxrel < tol_max;
        printf("%-10s rows %4lld  rel_rms %.3e  max_abs/row_rms %.3e  %s\n", c.c_str(), (long long) ggml_nrows(ref), rel, maxrel, ok ? "OK" : "FAIL");
        fails += !ok;
        ggml_free(ctx);
    }
    gguf_free(gctx);
    ggml_free(ctx_data);
    return fails ? 1 : 0;
}
