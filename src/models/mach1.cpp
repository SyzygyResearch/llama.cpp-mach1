#include "models.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

// Mach-1-Additive.
//
// The checkpoint is a qwen35moe topology whose quantized matrices ship as
// trellis code streams instead of dense weights (the exporter's schema
// constants are the frozen contract for the tensor names used here).
// Payload v2 tensor set:
//   * NE tier      - <base>.m1_packed / .m1_gscale / .m1_lut on the stock
//                    attention/GDN/shared-expert stems, plus lm_head under
//                    "output" (K=3, 8 LUT chunks). Decoded in-kernel by
//                    GGML_OP_MACH1_NE_MM.
//   * embeddings   - token_embd.m1_q / .m1_mn / .m1_mx (int3 asymmetric),
//                    gathered by GGML_OP_MACH1_EMBED_ROWS.
//   * experts      - blk.N.ffn_{gate,up,down}_exps.m1_* (bitshift trellis, decoded
//                    in-kernel by GGML_OP_MACH1_EXP_MM). Hybrid bring-up GGUFs
//                    may instead carry stock dequantized ffn_*_exps tensors,
//                    which run the stock path.
//   * everything else (norms, router, GDN scalars) is dense F32 on the stock
//     names and loads exactly like qwen35moe.
// Payload v3 (format_version 3) swaps the NE tier for <base>.m1_rt_trellis /
// .m1_rt_su / .m1_rt_sv (shared mach1.ne_tlut codebook), the embeddings for
// token_embd.m1_codes / .m1_lut, the lm_head for output.m1_qp / .m1_gscale
// (int5-g64), and in the experts swaps the demoted tier (m1_dem_trellis +
// basis) for m1_wave_gamma.
//
// hparams come verbatim from llama_model_qwen35moe::load_arch_hparams (the
// exporter re-prefixes qwen35moe.* KVs to mach1.*).

void llama_model_mach1::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    GGML_ASSERT(hparams.n_layer_nextn == 0 && "mach1 checkpoints do not ship the MTP head");

    // packed sidecar with shape taken from the GGUF metadata (kept/demoted
    // expert counts vary per layer, so shapes cannot be derived from hparams)
    auto create_m1 = [&](const LLM_TN_IMPL & tnv, bool required) -> ggml_tensor * {
        const std::string name = tnv.str();
        ggml_tensor * meta = ml.get_tensor_meta(name.c_str());
        if (meta == nullptr) {
            if (!required) {
                return nullptr;
            }
            throw std::runtime_error("mach1: missing required tensor '" + name + "'");
        }
        switch (ggml_n_dims(meta)) {
            case 1: return create_tensor(tnv, { meta->ne[0] }, 0);
            case 2: return create_tensor(tnv, { meta->ne[0], meta->ne[1] }, 0);
            case 3: return create_tensor(tnv, { meta->ne[0], meta->ne[1], meta->ne[2] }, 0);
            default:
                throw std::runtime_error("mach1: unexpected rank for tensor '" + name + "'");
        }
    };
    // codec streams echo their GGUF shape above, so every dim the graph builds
    // fixed-shape views on must be pinned to its hparams-derived value here
    auto check_m1_dim = [&](const ggml_tensor * t, int dim, int64_t expected) {
        if (t->ne[dim] != expected) {
            throw std::runtime_error(format("mach1: tensor '%s' has wrong shape: ne[%d] = %lld, expected %lld",
                t->name, dim, (long long) t->ne[dim], (long long) expected));
        }
    };
    // payload version: 2 = trellis V2 tiers, 3 = additive (V8 experts, rotated
    // NE spine, int5-g64 head, nibble-LUT embed)
    ml.get_key("mach1.format_version", m1_version, false);

    // The codec tensors are RAW code streams stored under stock ggml types
    // (I8/I16/I32/F16), so a payload this build predates would not trip any
    // type check - it would be silently misread as v2/v3 bytes. Reject it
    // here instead, with the version in the message.
    constexpr uint32_t M1_FORMAT_VERSION_MAX = 3;
    if (m1_version > M1_FORMAT_VERSION_MAX) {
        throw std::runtime_error(format(
            "mach1 checkpoint declares format_version %u; this build supports up to %u "
            "- use a newer fork build", m1_version, M1_FORMAT_VERSION_MAX));
    }

    // n_in/n_out are the dense widths of the projection the stream decodes to
    auto create_m1_ne = [&](llm_tensor base, int il, int64_t n_in, int64_t n_out) -> m1_ne {
        m1_ne w;
        if (m1_version >= 3) {
            w.rt_trellis = create_m1(tn(base, "m1_rt_trellis", il), true);
            w.rt_su      = create_m1(tn(base, "m1_rt_su",      il), true);
            w.rt_sv      = create_m1(tn(base, "m1_rt_sv",      il), true);
            check_m1_dim(w.rt_su, 0, n_in);
            check_m1_dim(w.rt_sv, 0, n_out);
            return w;
        }
        w.packed = create_m1(tn(base, "m1_packed", il), true);
        w.gscale = create_m1(tn(base, "m1_gscale", il), true);
        w.lut    = create_m1(tn(base, "m1_lut",    il), true);
        check_m1_dim(w.packed, 1, n_out);
        check_m1_dim(w.gscale, 1, n_out);
        return w;
    };

    m1_layers.resize(n_layer);

    // global: shared expert trellis codebook + packed embedding + packed lm_head
    m1_tlut = create_m1(tn(LLM_TENSOR_MACH1_TLUT), true);

    if (m1_version >= 3) {
        m1_ne_tlut    = create_m1(tn(LLM_TENSOR_MACH1_NE_TLUT), true);
        m1_embed_codes = create_m1(tn(LLM_TENSOR_TOKEN_EMBD, "m1_codes"), true);
        m1_embed_lut   = create_m1(tn(LLM_TENSOR_TOKEN_EMBD, "m1_lut"),   true);
    } else {
        m1_embed_q  = create_m1(tn(LLM_TENSOR_TOKEN_EMBD, "m1_q"),  true);
        m1_embed_mn = create_m1(tn(LLM_TENSOR_TOKEN_EMBD, "m1_mn"), true);
        m1_embed_mx = create_m1(tn(LLM_TENSOR_TOKEN_EMBD, "m1_mx"), true);
    }

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    if (m1_version >= 3) {
        m1_head_qp     = create_m1(tn(LLM_TENSOR_OUTPUT, "m1_qp"),     true);
        m1_head_gscale = create_m1(tn(LLM_TENSOR_OUTPUT, "m1_gscale"), true);
        check_m1_dim(m1_head_qp,     0, n_embd/8*5);
        check_m1_dim(m1_head_qp,     1, n_vocab);
        check_m1_dim(m1_head_gscale, 0, n_embd/64);
        check_m1_dim(m1_head_gscale, 1, n_vocab);
    } else {
        m1_output = create_m1_ne(LLM_TENSOR_OUTPUT, -1, n_embd, n_vocab);
    }

    // hybrid bring-up artifacts carry stock dequantized expert tensors
    m1_stock_experts = ml.get_tensor_meta(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", 0).str().c_str()) != nullptr;

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];
        auto & m1l   = m1_layers[il];

        const int64_t n_ff_exp   = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t key_dim    = hparams.ssm_d_state * hparams.ssm_n_group;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        // dense F32 keeps — identical to qwen35moe
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, 0);

        if (!hparams.is_recr(il)) {
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, 0);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, 0);

            m1l.wq = create_m1_ne(LLM_TENSOR_ATTN_Q,   il, n_embd, n_embd_head_k * n_head * 2);
            m1l.wk = create_m1_ne(LLM_TENSOR_ATTN_K,   il, n_embd, n_embd_k_gqa);
            m1l.wv = create_m1_ne(LLM_TENSOR_ATTN_V,   il, n_embd, n_embd_v_gqa);
            m1l.wo = create_m1_ne(LLM_TENSOR_ATTN_OUT, il, n_embd_head_k * n_head, n_embd);
        } else {
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, 0);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, 0);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, 0);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, 0);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, n_v_heads }, 0);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, 0);

            m1l.wqkv      = create_m1_ne(LLM_TENSOR_ATTN_QKV,  il, n_embd, conv_dim);
            m1l.wqkv_gate = create_m1_ne(LLM_TENSOR_ATTN_GATE, il, n_embd, value_dim);
            m1l.ssm_out   = create_m1_ne(LLM_TENSOR_SSM_OUT,   il, value_dim, n_embd);
        }

        // router (dense)
        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", il), { n_embd, n_expert }, 0);

        if (m1_stock_experts) {
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", il), { n_embd, n_ff_exp, n_expert }, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", il), { n_embd, n_ff_exp, n_expert }, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, 0);
        } else {
            m1l.remap = create_m1(tn(LLM_TENSOR_FFN_GATE_INP, "m1_remap", il), true);
            check_m1_dim(m1l.remap, 0, n_expert);
            const llm_tensor exps[3] = { LLM_TENSOR_FFN_GATE_EXPS, LLM_TENSOR_FFN_UP_EXPS, LLM_TENSOR_FFN_DOWN_EXPS };
            for (int p = 0; p < 3; ++p) {
                auto & e = m1l.exps[p];
                e.kept_trellis = create_m1(tn(exps[p], "m1_kept_trellis", il), true);
                e.dem_trellis  = create_m1(tn(exps[p], "m1_dem_trellis",  il), m1_version < 3);
                e.su           = create_m1(tn(exps[p], "m1_su", il), true);
                e.sv           = create_m1(tn(exps[p], "m1_sv", il), true);
                e.basis_a      = create_m1(tn(exps[p], "m1_basis_a", il), false);
                e.basis_b      = create_m1(tn(exps[p], "m1_basis_b", il), false);
                e.basis_c      = create_m1(tn(exps[p], "m1_basis_c", il), false);
                e.wave_gamma   = create_m1(tn(exps[p], "m1_wave_gamma", il), m1_version >= 3);

                // the triple is consumed as a unit by ggml_mach1_exp_basis
                if ((e.basis_b != nullptr) != (e.basis_a != nullptr) || (e.basis_c != nullptr) != (e.basis_a != nullptr)) {
                    throw std::runtime_error("mach1: incomplete basis triple '" + tn(exps[p], "m1_basis_a", il).str() + "' - m1_basis_a/b/c must all be present or all absent");
                }

                // decoded expert widths the ffn graph views assume (down feeds
                // the [n_embd, n_tokens] expert-slot views)
                const int64_t w_in  = p == 2 ? n_ff_exp : n_embd;
                const int64_t w_out = p == 2 ? n_embd   : n_ff_exp;
                check_m1_dim(e.su, 0, w_in);
                check_m1_dim(e.sv, 0, w_out);
                if (e.basis_a) {
                    check_m1_dim(e.basis_a, 0, w_in);
                    check_m1_dim(e.basis_b, 1, w_out);
                    if (e.dem_trellis) {
                        check_m1_dim(e.basis_c, 1, e.dem_trellis->ne[2]);
                    }
                }
            }
        }

        // shared expert (dense gate + packed projections)
        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, 0);
        m1l.gate_shexp = create_m1_ne(LLM_TENSOR_FFN_GATE_SHEXP, il, n_embd, n_ff_shexp);
        m1l.up_shexp   = create_m1_ne(LLM_TENSOR_FFN_UP_SHEXP,   il, n_embd, n_ff_shexp);
        m1l.down_shexp = create_m1_ne(LLM_TENSOR_FFN_DOWN_SHEXP, il, n_ff_shexp, n_embd);
    }
}

std::unique_ptr<llm_graph_context> llama_model_mach1::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        throw std::runtime_error("mach1: no MTP head in this checkpoint");
    }
    return std::make_unique<graph>(*this, params);
}

// ---------------------------------------------------------------------------
// graph — qwen35moe topology with every dense projection routed through the
// mach1 codec ops. Structure mirrors llama_model_qwen35moe::graph so the two
// stay diffable.
// ---------------------------------------------------------------------------

ggml_tensor * llama_model_mach1::graph::ne_mm(const m1_ne & w, ggml_tensor * x) {
    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx0, x);
    }
    if (w.rt_trellis) {   // payload v3: rotated int-lattice spine
        return ggml_mach1_rt_mm(ctx0, w.rt_trellis, w.rt_su, w.rt_sv, model.m1_ne_tlut, x);
    }
    return ggml_mach1_ne_mm(ctx0, w.packed, w.gscale, w.lut, x);
}

// grouped -> tiled V-head row reorder on an activation vector segment
// [head_v_dim * n_v_heads, T]: new[(v*K + k)*hd + d] = old[(k*r + v)*hd + d].
// The v2 exporter bakes this into the packed NE streams; the v3 rotated codec
// cannot take row permutations (the Hadamard mixes rows), so it moves here.
ggml_tensor * llama_model_mach1::graph::v_tiled(ggml_tensor * y) {
    const int64_t hd = hparams.ssm_d_state;                    // head_v_dim
    const int64_t K  = hparams.ssm_n_group;                    // num_k_heads
    const int64_t r  = hparams.ssm_dt_rank / K;                // v heads per k head
    const int64_t T  = y->ne[1];
    GGML_ASSERT(y->ne[0] == hd*K*r);
    ggml_tensor * t = ggml_reshape_4d(ctx0, y, hd, r, K, T);
    t = ggml_cont(ctx0, ggml_permute(ctx0, t, 0, 2, 1, 3));    // [hd, K, r, T]
    return ggml_reshape_2d(ctx0, t, hd*K*r, T);
}

// build_inp_embd with the int3 embedding gather in place of get_rows
// (mirrors llm_graph_context::build_inp_embd; no lora on packed embeddings)
ggml_tensor * llama_model_mach1::graph::build_inp_embd_mach1() {
    const int64_t n_embd_inp = hparams.n_embd_inp();
    const int64_t n_embd_    = hparams.n_embd;
    GGML_ASSERT(n_embd_inp == n_embd_ && "mach1: no deepstack inputs");

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
    cb(inp->tokens, "inp_tokens", -1);
    ggml_set_input(inp->tokens);
    res->t_inp_tokens = inp->tokens;

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
    cb(inp->embd, "inp_embd", -1);
    ggml_set_input(inp->embd);

    std::array<ggml_tensor *, 2> inps;
    inps[0] = model.m1_embed_codes
        ? ggml_mach1_embed_gather(ctx0, model.m1_embed_codes, model.m1_embed_lut, inp->tokens)
        : ggml_mach1_embed_rows(ctx0, model.m1_embed_q, model.m1_embed_mn, model.m1_embed_mx, inp->tokens);
    inps[1] = inp->embd;

    GGML_ASSERT(ggml_are_same_shape(inps[0], inps[1]));

    ggml_tensor * cur = ggml_build_forward_select(gf, inps.data(), inps.size(), ubatch.token ? 0 : 1);

    res->t_inp_embd = cur;
    res->add_input(std::move(inp));

    return cur;
}

llama_model_mach1::graph::graph(const llama_model_mach1 & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd_mach1();

    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        ggml_tensor * ffn_residual = cur;

        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_moe", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head: v2 = K=3 NE tier (8 LUT chunks); v3 = int5-g64 codec
    if (model.m1_head_qp) {
        if (!ggml_is_contiguous(cur)) {
            cur = ggml_cont(ctx0, cur);
        }
        cur = ggml_mach1_head_mm(ctx0, model.m1_head_qp, model.m1_head_gscale, cur);
    } else {
        cur = ne_mm(model.m1_output, cur);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_mach1::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = ne_mm(model.m1_layers[il].wqkv, input);
    if (model.m1_layers[il].wqkv.rt_trellis) {
        // v3: the grouped->tiled V-row reorder is not baked into the rotated
        // codec, so permute the V segment of the output activations here
        const int64_t key_dim = hparams.ssm_d_state * hparams.ssm_n_group;
        const int64_t val_dim = hparams.ssm_d_state * hparams.ssm_dt_rank;
        const int64_t T = qkv_mixed->ne[1];
        ggml_tensor * qk = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv_mixed, 2*key_dim, T,
                                                        qkv_mixed->nb[1], 0));
        ggml_tensor * v  = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv_mixed, val_dim, T,
                                                        qkv_mixed->nb[1], 2*key_dim*sizeof(float)));
        qkv_mixed = ggml_concat(ctx0, qk, v_tiled(v), 0);
    }

    ggml_tensor * z = ne_mm(model.m1_layers[il].wqkv_gate, input);
    if (model.m1_layers[il].wqkv_gate.rt_trellis) {
        z = v_tiled(z);   // v3: same reorder on the z gate rows
    }
    cb(z, "z", il);

    // expanded back to back so the two same-input rt ops and their reorder
    // glue sit adjacent in the cgraph for the CUDA qkv batch matcher
    // (ggml_cuda_mach1_vtiled_fuse) - node order only, semantics unchanged
    ggml_build_forward_expand(gf, qkv_mixed);
    ggml_build_forward_expand(gf, z);

    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_mach1::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llama_model_mach1::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * Qcur_full = ne_mm(model.m1_layers[il].wq, cur); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Kcur = ne_mm(model.m1_layers[il].wk, cur);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = ne_mm(model.m1_layers[il].wv, cur);
    cb(Vcur, "Vcur", il);

    // expanded back to back so the three same-input rt ops sit adjacent in
    // the cgraph for the CUDA qkv batch matcher (ggml_cuda_mach1_qkv_fuse) -
    // node order only, semantics unchanged
    ggml_build_forward_expand(gf, Qcur_full);
    ggml_build_forward_expand(gf, Kcur);
    ggml_build_forward_expand(gf, Vcur);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = ne_mm(model.m1_layers[il].wo, cur);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_mach1::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    // the beta/alpha projections, gathered state and conv output are expanded
    // ahead of the deferred norm/gate chain so that chain sits contiguous in
    // the cgraph for the CUDA GDN-core matcher (ggml_cuda_mach1_gdn_fuse) -
    // node order only, semantics unchanged
    ggml_build_forward_expand(gf, beta);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    ggml_build_forward_expand(gf, alpha);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    static const int gdn_sgf = getenv("GGML_MACH1_GDN_SGF") != nullptr ?
                               atoi(getenv("GGML_MACH1_GDN_SGF")) : 0;
    // GGML_MACH1_NT32: widen the exact-B16 admission to the (16, 32] decode
    // widths. GGML_MACH1_NTLO: widen it to [2, 16) (the indexed kernels are
    // seq-general; B1 keeps the copying path - its draws were unstable under
    // the indexed form). Same steady-state aliasing contract; B16 behavior
    // unchanged.
    static const bool nt32_on = getenv("GGML_MACH1_NT32") != nullptr &&
                                atoi(getenv("GGML_MACH1_NT32")) != 0;
    static const bool ntlo_on = getenv("GGML_MACH1_NTLO") != nullptr &&
                                atoi(getenv("GGML_MACH1_NTLO")) != 0;
    // GGML_MACH1_NTLOW (the main-line name) engages under either env, as in
    // mach1.cu; NTLOW_MIN (floor 2) narrows only the NTLOW term, while NTLO
    // alone admits all of [2, 16).
    static const bool ntlow_on = ntlo_on || (getenv("GGML_MACH1_NTLOW") != nullptr &&
                                 atoi(getenv("GGML_MACH1_NTLOW")) != 0);
    static const int ntlow_min = getenv("GGML_MACH1_NTLOW_MIN") != nullptr ?
                                 std::max(2, atoi(getenv("GGML_MACH1_NTLOW_MIN"))) : 2;
    const bool sgf_seqs = n_seqs == 16 || (nt32_on && n_seqs > 16 && n_seqs <= 32) ||
                          (ntlo_on && n_seqs >= 2 && n_seqs < 16) ||
                          (ntlow_on && n_seqs >= ntlow_min && n_seqs < 16);

    // level 2: the conv chain (gather + concat + conv + silu + state write)
    // collapses into one indexed kernel under the same aliasing contract
    const bool sgf_conv = gdn_sgf >= 2 && n_seq_tokens == 1 && sgf_seqs &&
                          cparams.fused_gdn_ar && conv_kernel_size == 4;
    ggml_tensor * conv_input = nullptr;
    ggml_tensor * conv_idx_out = nullptr;
    if (sgf_conv) {
        const auto  * kv_state   = inp->mctx;
        const int32_t rrow       = hparams.n_embd_r();
        const int32_t rs_z       = kv_state->get_rs_z();
        const uint32_t n_rs      = kv_state->get_n_rs();
        const uint32_t rs_head   = kv_state->get_head();
        ggml_tensor * rrows = ggml_reshape_2d(ctx0, conv_states_all, rrow, conv_states_all->ne[1]);
        ggml_tensor * rzero = ggml_view_1d(ctx0, rrows,
            (int64_t) rrow*(rs_z >= 0), rs_z*rrows->nb[1]*(rs_z >= 0));
        ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, rzero, 0));
        ggml_tensor * rextra = ggml_get_rows(ctx0, rrows, inp->s_copy_extra);
        ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, rextra,
                ggml_view_2d(ctx0, conv_states_all, rrow, (int64_t) n_rs - n_seqs,
                    conv_states_all->nb[1], (rs_head + n_seqs)*conv_states_all->nb[1])));
        ggml_tensor * x_new = qkv_mixed;
        if (!ggml_is_contiguous(x_new)) {
            x_new = ggml_cont(ctx0, x_new);
        }
        conv_idx_out = ggml_ssm_conv_idx(ctx0, rrows, conv_kernel,
                                         inp->s_copy_main, x_new, (int32_t) rs_head, /*silu=*/true);
        cb(conv_idx_out, "conv_output_silu", il);
        ggml_build_forward_expand(gf, conv_idx_out);
    } else {
        conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);
    }

    // GDN_SGF: at fused single-token decode the delta-net op gathers its
    // state rows through s_copy at load, deleting the materialized get_rows
    // copy of the full state per layer per step. The zero and extra-state
    // maintenance nodes build_rs would emit are kept identical. Aliasing
    // contract: reads are rows s_copy[i], writes rows head+i; the fused form
    // requires s_copy[i] == head+j only when i == j (true at steady-state
    // decode; a forked/shuffled mapping must use the copying path).
    // exact-B16 gate like the rest of the p16 stack: the B1 draws were
    // unstable under the indexed form and B1 keeps the copying path
    const bool sgf_on = gdn_sgf >= 1 && n_seq_tokens == 1 && sgf_seqs && cparams.fused_gdn_ar;
    ggml_tensor * state;
    if (sgf_on) {
        const auto  * kv_state   = inp->mctx;
        const int32_t state_size = hparams.n_embd_s();
        const int32_t rs_z       = kv_state->get_rs_z();
        const uint32_t n_rs      = kv_state->get_n_rs();
        const uint32_t rs_head   = kv_state->get_head();
        ggml_tensor * states = ggml_reshape_2d(ctx0, ssm_states_all, state_size, ssm_states_all->ne[1]);
        ggml_tensor * state_zero = ggml_view_1d(ctx0, states,
            (int64_t) state_size*(rs_z >= 0), rs_z*states->nb[1]*(rs_z >= 0));
        ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));
        ggml_tensor * states_extra = ggml_get_rows(ctx0, states, inp->s_copy_extra);
        ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, states_extra,
                ggml_view_2d(ctx0, ssm_states_all, state_size, (int64_t) n_rs - n_seqs,
                    ssm_states_all->nb[1], (rs_head + n_seqs)*ssm_states_all->nb[1])));
        gdn_state_rows = states;
        gdn_state_idx  = inp->s_copy_main;
        // shape-only stand-in for the shared asserts; the indexed op never reads it
        state = ggml_reshape_4d(ctx0,
            ggml_view_2d(ctx0, states, state_size, n_seqs, states->nb[1], rs_head*states->nb[1]),
            head_v_dim, head_v_dim, num_v_heads, n_seqs);
    } else {
        gdn_state_rows = nullptr;
        gdn_state_idx  = nullptr;
        state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
        state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
        ggml_build_forward_expand(gf, state);
    }
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_silu;
    if (sgf_conv) {
        conv_output_silu = conv_idx_out;
    } else {
        ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
        cb(conv_output_proper, "conv_output_raw", il);

        conv_output_silu = ggml_silu(ctx0, conv_output_proper);
        cb(conv_output_silu, "conv_output_silu", il);
        ggml_build_forward_expand(gf, conv_output_silu);
    }

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);

    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // The packed ssm_out stream keeps the HF column order, where V heads are
    // GROUPED by K head; the graph runs in ggml's TILED order (the exporter
    // row-reorders every other v-indexed tensor). A trellis stream's columns
    // cannot be permuted at export, so convert the activations tiled->grouped
    // here: heads [hd, r*K (k fastest)] -> [hd, K*r (v fastest)].
    ggml_tensor * final_output;
    if (num_k_heads != num_v_heads) {
        const int64_t r = num_v_heads / num_k_heads;
        ggml_tensor * t = ggml_reshape_4d(ctx0, attn_out_norm, head_v_dim, num_k_heads, r, n_seq_tokens*n_seqs);
        t = ggml_cont(ctx0, ggml_permute(ctx0, t, 0, 2, 1, 3));   // [hd, r, K, T]
        final_output = ggml_reshape_3d(ctx0, t, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    } else {
        final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    }
    cb(final_output, "final_output", il);

    cur = ne_mm(model.m1_layers[il].ssm_out, final_output);
    cb(cur, "linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_mach1::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * moe_out;
    if (model.m1_stock_experts) {
        moe_out =
            build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
                nullptr, model.layers[il].ffn_gate_up_exps,
                model.layers[il].ffn_up_exps_s,
                model.layers[il].ffn_gate_exps_s,
                model.layers[il].ffn_down_exps_s);
    } else {
        // routed experts from packed trellis streams. Routing mirrors
        // build_moe_ffn(SOFTMAX, norm_w=true, scale=expert_weights_scale) with
        // the three mul_mat_id calls replaced by the mach1 expert ops.
        const auto & m1l = model.m1_layers[il];

        // like build_moe_ffn, take the token count from the tensor: the
        // last-layer inp_out_ids gather may have shrunk cur below ubatch
        const int64_t n_tokens = cur->ne[1];

        ggml_tensor * logits = build_lora_mm(model.layers[il].ffn_gate_inp, cur); // [n_expert, n_tokens]
        cb(logits, "ffn_moe_logits", il);

        ggml_tensor * probs = ggml_soft_max(ctx0, logits);
        cb(probs, "ffn_moe_probs", il);

        ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, probs, n_expert_used); // [n_expert_used, n_tokens]
        cb(selected_experts, "ffn_moe_topk", il);

        ggml_tensor * weights = ggml_get_rows(ctx0,
            ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens), selected_experts); // [1, n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights", il);

        // norm_w: normalize the selected probabilities to sum to 1
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        weights = ggml_div(ctx0, weights, weights_sum);
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_norm", il);

        const float w_scale = hparams.expert_weights_scale;
        if (w_scale != 0.0f && w_scale != 1.0f) {
            weights = ggml_scale(ctx0, weights, w_scale);
            cb(weights, "ffn_moe_weights_scaled", il);
        }

        ggml_build_forward_expand(gf, weights);

        ggml_tensor * xexp = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

        auto exp_mm = [&](const m1_exp & e, ggml_tensor * xin) {
            if (!ggml_is_contiguous(xin)) {
                xin = ggml_cont(ctx0, xin);
            }
            ggml_tensor * out = ggml_mach1_exp_mm(ctx0,
                e.kept_trellis, e.dem_trellis, e.su, e.sv, model.m1_tlut,
                m1l.remap, selected_experts, xin, e.wave_gamma);
            if (e.basis_a) {
                // fused acc form: dst = out + basis (no separate ggml_add node)
                out = ggml_mach1_exp_basis(ctx0, e.basis_a, e.basis_b, e.basis_c,
                                           m1l.remap, selected_experts, xin, out);
            }
            return out;
        };

        ggml_tensor * gate = exp_mm(m1l.exps[0], xexp); // [n_ff_exp, n_expert_used, n_tokens]
        cb(gate, "ffn_moe_gate", il);
        ggml_tensor * up   = exp_mm(m1l.exps[1], xexp);
        cb(up, "ffn_moe_up", il);

        ggml_tensor * par;
        const float swiglu_limit = il >= 0 ? hparams.swiglu_clamp_exp[il] : 0.0f;
        if (swiglu_limit > 1e-6f) {
            up = ggml_clamp(ctx0, up, -swiglu_limit, swiglu_limit);
            ggml_tensor * gate_act = ggml_silu(ctx0, gate);
            gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, swiglu_limit);
            par = ggml_mul(ctx0, gate_act, up);
            cb(par, "ffn_moe_swiglu_limited", il);
        } else {
            par = ggml_swiglu_split(ctx0, gate, up);
            cb(par, "ffn_moe_swiglu", il);
        }

        ggml_tensor * experts = exp_mm(m1l.exps[2], par); // [n_embd, n_expert_used, n_tokens]
        cb(experts, "ffn_moe_down", il);

        experts = ggml_mul(ctx0, experts, weights);
        cb(experts, "ffn_moe_weighted", il);

        ggml_build_forward_expand(gf, experts);

        // aggregate the expert slots (same view+add pattern as build_moe_ffn)
        ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };
        for (uint32_t i = 0; i < hparams.n_expert_used; ++i) {
            cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);
            ggml_build_forward_expand(gf, cur_experts[i]);
        }
        moe_out = cur_experts[0];
        for (uint32_t i = 1; i < hparams.n_expert_used; ++i) {
            moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
            ggml_build_forward_expand(gf, moe_out);
        }
        if (hparams.n_expert_used == 1) {
            moe_out = ggml_cont(ctx0, moe_out);
        }
    }
    cb(moe_out, "ffn_moe_out", il);

    // shared expert: silu(gate(x)) * up(x) -> down, then sigmoid-gated
    ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
    cb(shared_gate, "shared_expert_gate", il);

    shared_gate = ggml_sigmoid(ctx0, shared_gate);
    cb(shared_gate, "shared_expert_gate_sigmoid", il);

    // expanded ahead of the rt chain so the gate dst is computed before the
    // CUDA fused region that reads it (ggml_cuda_mach1_shexp_fuse)
    ggml_build_forward_expand(gf, shared_gate);

    ggml_tensor * g = ne_mm(model.m1_layers[il].gate_shexp, cur);
    ggml_tensor * u = ne_mm(model.m1_layers[il].up_shexp,   cur);
    ggml_tensor * ffn_shexp = ne_mm(model.m1_layers[il].down_shexp,
                                    ggml_mul(ctx0, ggml_silu(ctx0, g), u));
    cb(ffn_shexp, "ffn_shexp", il);

    ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
    cb(ffn_shexp, "ffn_shexp_gated", il);

    cur = ggml_add(ctx0, moe_out, ffn_shexp);
    cb(cur, "ffn_out", il);

    return cur;
}
