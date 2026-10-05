// Bit-exactness gate for the Mach-1 codec CPU kernels.
//
// Reads a fixture GGUF containing real sliced tensors from the released
// checkpoint plus reference goldens. The NE/head kernels are driven with an
// IDENTITY input so every output element is a single fp32 product (no
// accumulation) - the comparison is bitwise, not tolerance.
//
// usage: test-mach1-codec <fixtures.gguf>

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>

static int compare_bits(const char * name, const ggml_tensor * y, const ggml_tensor * golden) {
    if (!ggml_are_same_shape(y, golden)) {
        fprintf(stderr, "%s: shape mismatch [%lld,%lld] vs [%lld,%lld]\n", name,
                (long long) y->ne[0], (long long) y->ne[1],
                (long long) golden->ne[0], (long long) golden->ne[1]);
        return 1;
    }
    const size_t n = ggml_nelements(y);
    const uint32_t * a = (const uint32_t *) y->data;
    const uint32_t * b = (const uint32_t *) golden->data;
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            if (bad < 4) {
                float fa, fb;
                memcpy(&fa, &a[i], 4);
                memcpy(&fb, &b[i], 4);
                fprintf(stderr, "%s: mismatch at %zu: %.9g (0x%08x) vs %.9g (0x%08x)\n",
                        name, i, fa, a[i], fb, b[i]);
            }
            bad++;
        }
    }
    printf("%-14s %8zu elements  %s (%zu mismatched)\n", name, n, bad ? "FAIL" : "BIT-EXACT", bad);
    return bad ? 1 : 0;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <mach1_fixtures.gguf>\n", argv[0]);
        return 2;
    }

    ggml_context * ctx_data = NULL;
    gguf_init_params gp = { /*.no_alloc =*/ false, /*.ctx =*/ &ctx_data };
    gguf_context * gctx = gguf_init_from_file(argv[1], gp);
    if (!gctx) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 2;
    }

    auto get = [&](const char * name) {
        ggml_tensor * t = ggml_get_tensor(ctx_data, name);
        if (!t) {
            fprintf(stderr, "fixture tensor missing: %s\n", name);
            exit(2);
        }
        return t;
    };

    ggml_init_params ip = {  (size_t) 1536*1024*1024,  NULL,  false };
    ggml_context * ctx = ggml_init(ip);

    const int fixkey = gguf_find_key(gctx, "mach1.fixtures.version");

    if (fixkey >= 0 && gguf_get_val_u32(gctx, fixkey) >= 4) {
        auto make_eye = [&](int64_t d) {
            ggml_tensor * e = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, d);
            memset(e->data, 0, ggml_nbytes(e));
            for (int64_t i = 0; i < d; ++i) {
                ((float *) e->data)[i*d + i] = 1.0f;
            }
            return e;
        };
        ggml_tensor * tlut = get("da4.tlut");

        struct da_case { const char * set; int mode; int split; bool optional; ggml_tensor * y; };
        da_case dcs[] = {
            { "da4.k15",     0, 0, false, NULL },
            { "da4.k175s1",  1, 0, false, NULL },
            { "da4.k1875s1", 1, 0, false, NULL },
            { "da4.eout",    0, 2, false, NULL },
            { "da4.ein",     0, 1, false, NULL },
            { "da4.h20",     0, 0, true,  NULL },
        };
        char nb[64];
        for (auto & c : dcs) {
            snprintf(nb, sizeof(nb), "%s.trellis", c.set);
            if (c.optional && !ggml_get_tensor(ctx_data, nb)) {
                continue;
            }
            auto gete = [&](const char * f) {
                snprintf(nb, sizeof(nb), "%s.%s", c.set, f);
                return get(nb);
            };
            ggml_tensor * golden = gete("golden");
            c.y = ggml_mach1_da_mm(ctx, gete("trellis"), gete("su"), gete("sv"),
                                   gete("wgamma"), tlut, make_eye(golden->ne[1]),
                                   NULL, NULL, c.mode, c.split, 0);
        }

        ggml_tensor * y_exc = ggml_mach1_da_mm(ctx, get("exc4.trellis"), get("exc4.su"),
                                               get("exc4.sv"), get("exc4.wgamma"), tlut,
                                               make_eye(get("exc4.golden")->ne[1]),
                                               get("exc4.exc_idx"), get("exc4.exc_rows"), 0, 0, 0);

        struct int_case { const char * set; ggml_tensor * y; };
        int_case ics[] = { { "int4.q3", NULL }, { "int4.q4", NULL }, { "int4.head", NULL } };
        for (auto & c : ics) {
            auto gete = [&](const char * f) {
                snprintf(nb, sizeof(nb), "%s.%s", c.set, f);
                return get(nb);
            };
            ggml_tensor * q = gete("q");
            const int64_t n = gete("golden")->ne[1];
            const int bits  = (int)(q->ne[0]*8/n);
            c.y = ggml_mach1_int_mm(ctx, q, gete("mn"), gete("mx"), make_eye(n), bits);
        }

        ggml_tensor * y_emb = ggml_mach1_da_embed(ctx, get("emb4.trellis"), get("emb4.su"),
                                                  get("emb4.sv"), get("emb4.wgamma"), tlut,
                                                  get("emb4.ids"), 1);

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
        for (auto & c : dcs) {
            if (c.y) {
                ggml_build_forward_expand(gf, c.y);
            }
        }
        ggml_build_forward_expand(gf, y_exc);
        for (auto & c : ics) {
            ggml_build_forward_expand(gf, c.y);
        }
        ggml_build_forward_expand(gf, y_emb);
        if (ggml_graph_compute_with_ctx(ctx, gf, 4) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed\n");
            return 2;
        }

        int rc = 0;
        for (auto & c : dcs) {
            if (!c.y) {
                continue;
            }
            snprintf(nb, sizeof(nb), "%s.golden", c.set);
            rc |= compare_bits(c.set, c.y, get(nb));
        }
        rc |= compare_bits("exc4", y_exc, get("exc4.golden"));
        for (auto & c : ics) {
            snprintf(nb, sizeof(nb), "%s.golden", c.set);
            rc |= compare_bits(c.set, c.y, get(nb));
        }
        rc |= compare_bits("emb4", y_emb, get("emb4.golden"));

        ggml_free(ctx);
        ggml_free(ctx_data);
        gguf_free(gctx);
        printf(rc == 0 ? "PASS: all mach1 v4 codec kernels bit-exact vs reference goldens\n"
                       : "FAIL (v4)\n");
        return rc;
    }

    // v3 (additive payload) fixture set: V8+wave_gamma experts, rotated NE,
    // int5-g64 head, nibble-LUT embed
    if (fixkey >= 0 && gguf_get_val_u32(gctx, fixkey) >= 3) {
        auto make_eye = [&](int64_t d) {
            ggml_tensor * e = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, d);
            memset(e->data, 0, ggml_nbytes(e));
            for (int64_t i = 0; i < d; ++i) {
                ((float *) e->data)[i*d + i] = 1.0f;
            }
            return e;
        };

        // rotated NE (shexp gate [512, 2048])
        ggml_tensor * ne_tl = get("ne3.tlut");
        ggml_tensor * y_rt = ggml_mach1_rt_mm(ctx, get("ne3.trellis"), get("ne3.su"),
                                              get("ne3.sv"), ne_tl, make_eye(get("ne3.su")->ne[0]));

        // int5-g64 head slice
        ggml_tensor * y_head = ggml_mach1_head_mm(ctx, get("head3.qp"), get("head3.gscale"),
                                                  make_eye(2048));

        // nibble-LUT embed slice
        ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, get("emb3.golden")->ne[1]);
        for (int64_t i = 0; i < ids->ne[0]; ++i) {
            ((int32_t *) ids->data)[i] = (int32_t) i;
        }
        ggml_tensor * y_emb = ggml_mach1_embed_gather(ctx, get("emb3.codes"), get("emb3.lut"), ids);

        // V8 experts with wave_gamma; identity input -> W bitwise
        struct v3_case { const char * p; ggml_tensor * y; } cases[2] = { { "gate", 0 }, { "down", 0 } };
        ggml_tensor * tlut = get("exp3.tlut");
        for (auto & c : cases) {
            char nb[64];
            auto gete = [&](const char * f) {
                snprintf(nb, sizeof(nb), "exp3.%s.%s", c.p, f);
                return get(nb);
            };
            ggml_tensor * su = gete("su");
            const int64_t n = su->ne[0];
            ggml_tensor * eyex = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n, 1, n);
            memset(eyex->data, 0, ggml_nbytes(eyex));
            for (int64_t i = 0; i < n; ++i) {
                ((float *) eyex->data)[i*n + i] = 1.0f;
            }
            ggml_tensor * ids_e = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n);
            for (int64_t i = 0; i < n; ++i) {
                ((int32_t *) ids_e->data)[i] = 0;   // orig id 0 -> slot 1 (fixture remap)
            }
            c.y = ggml_mach1_exp_mm(ctx, gete("trellis"), NULL, su, gete("sv"), tlut,
                                    gete("remap"), ids_e, eyex, gete("gamma"));
        }

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(gf, y_rt);
        ggml_build_forward_expand(gf, y_head);
        ggml_build_forward_expand(gf, y_emb);
        for (auto & c : cases) {
            ggml_build_forward_expand(gf, c.y);
        }
        if (ggml_graph_compute_with_ctx(ctx, gf, 4) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed\n");
            return 2;
        }

        int rc = 0;
        rc |= compare_bits("ne3.rt",  y_rt,   get("ne3.golden"));
        rc |= compare_bits("head3",   y_head, get("head3.golden"));
        rc |= compare_bits("emb3",    y_emb,  get("emb3.golden"));
        for (auto & c : cases) {
            char nb[64], gname[64];
            snprintf(nb, sizeof(nb), "exp3.%s", c.p);
            snprintf(gname, sizeof(gname), "exp3.%s.golden", c.p);
            rc |= compare_bits(nb, c.y, get(gname));
        }

        ggml_free(ctx);
        ggml_free(ctx_data);
        gguf_free(gctx);
        printf(rc == 0 ? "PASS: all mach1 v3 codec kernels bit-exact vs reference goldens\n"
                       : "FAIL (v3)\n");
        return rc;
    }

    // identity input: y[:, j] = W[:, j] exactly — single product per element
    const int64_t T = 2048;
    ggml_tensor * eye = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, T);
    memset(eye->data, 0, ggml_nbytes(eye));
    for (int64_t i = 0; i < T; ++i) {
        ((float *) eye->data)[i*T + i] = 1.0f;
    }

    ggml_tensor * y_ne   = ggml_mach1_ne_mm(ctx, get("ne.packed"),   get("ne.gscale"),   get("ne.lut"),   eye);
    ggml_tensor * y_head = ggml_mach1_ne_mm(ctx, get("head.packed"), get("head.gscale"), get("head.lut"), eye);

    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, get("embed.golden")->ne[1]);
    for (int64_t i = 0; i < ids->ne[0]; ++i) {
        ((int32_t *) ids->data)[i] = (int32_t) i;
    }
    ggml_tensor * y_emb = ggml_mach1_embed_rows(ctx, get("embed.q"), get("embed.mn"), get("embed.mx"), ids);

    // expert tier: kept (K=2) and demoted (K=1) experts on both geometries;
    // identity input -> every output column is the materialized W bitwise,
    // and the basis op's residual matrix bitwise (kept slots must be zero)
    struct exp_case {
        const char * p;
        ggml_tensor * y_kept, * y_dem, * y_basis, * y_basis0;
    } exp_cases[2] = { { "gate", 0, 0, 0, 0 }, { "down", 0, 0, 0, 0 } };

    for (auto & ec : exp_cases) {
        char nb[64];
        auto gete = [&](const char * f) {
            snprintf(nb, sizeof(nb), "exp.%s.%s", ec.p, f);
            return get(nb);
        };
        ggml_tensor * kt = gete("kept"), * dt = gete("dem");
        ggml_tensor * su = gete("su"),   * sv = gete("sv"), * rm = gete("remap");

        const int64_t n = su->ne[0];
        ggml_tensor * eyex = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n, 1, n);
        memset(eyex->data, 0, ggml_nbytes(eyex));
        for (int64_t i = 0; i < n; ++i) {
            ((float *) eyex->data)[i*n + i] = 1.0f;
        }
        ggml_tensor * ids_k = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n);
        ggml_tensor * ids_d = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n);
        for (int64_t i = 0; i < n; ++i) {
            ((int32_t *) ids_k->data)[i] = 0;   // orig id 0 -> kept slot 1
            ((int32_t *) ids_d->data)[i] = 1;   // orig id 1 -> demoted slot 1
        }
        ggml_tensor * tlut = get("exp.tlut");
        ec.y_kept   = ggml_mach1_exp_mm(ctx, kt, dt, su, sv, tlut, rm, ids_k, eyex, NULL);
        ec.y_dem    = ggml_mach1_exp_mm(ctx, kt, dt, su, sv, tlut, rm, ids_d, eyex, NULL);
        ec.y_basis  = ggml_mach1_exp_basis(ctx, gete("basis_a"), gete("basis_b"), gete("basis_c"), rm, ids_d, eyex, NULL);
        ec.y_basis0 = ggml_mach1_exp_basis(ctx, gete("basis_a"), gete("basis_b"), gete("basis_c"), rm, ids_k, eyex, NULL);
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(gf, y_ne);
    ggml_build_forward_expand(gf, y_head);
    ggml_build_forward_expand(gf, y_emb);
    for (auto & ec : exp_cases) {
        ggml_build_forward_expand(gf, ec.y_kept);
        ggml_build_forward_expand(gf, ec.y_dem);
        ggml_build_forward_expand(gf, ec.y_basis);
        ggml_build_forward_expand(gf, ec.y_basis0);
    }

    if (ggml_graph_compute_with_ctx(ctx, gf, 4) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        return 2;
    }

    int rc = 0;
    rc |= compare_bits("ne",    y_ne,   get("ne.golden"));
    rc |= compare_bits("head",  y_head, get("head.golden"));
    rc |= compare_bits("embed", y_emb,  get("embed.golden"));
    for (auto & ec : exp_cases) {
        char nb[64], gname[64];
        auto cmp = [&](const char * f, ggml_tensor * y, const char * gsuf) {
            snprintf(nb, sizeof(nb), "exp.%s.%s", ec.p, f);
            snprintf(gname, sizeof(gname), "exp.%s.%s", ec.p, gsuf);
            rc |= compare_bits(nb, y, get(gname));
        };
        cmp("kept",  ec.y_kept,  "golden_kept");
        cmp("dem",   ec.y_dem,   "golden_dem");
        cmp("basis", ec.y_basis, "golden_basis");

        // kept-slot basis output must be exactly zero
        const float * z = (const float *) ec.y_basis0->data;
        size_t nz = 0;
        for (size_t i = 0; i < (size_t) ggml_nelements(ec.y_basis0); ++i) {
            nz += z[i] != 0.0f;
        }
        snprintf(nb, sizeof(nb), "exp.%s.b0", ec.p);
        printf("%-14s %8zu elements  %s (%zu nonzero)\n", nb,
               (size_t) ggml_nelements(ec.y_basis0), nz ? "FAIL" : "ZERO", nz);
        rc |= nz != 0;
    }

    ggml_free(ctx);
    ggml_free(ctx_data);
    gguf_free(gctx);

    printf(rc == 0 ? "PASS: all mach1 codec kernels bit-exact vs reference goldens\n"
                   : "FAIL\n");
    return rc;
}
