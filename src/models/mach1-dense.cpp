#include "models.h"
#include "llama-memory-recurrent.h"

#include <stdexcept>

enum {
    M1D_MODE_SYM = 0,
    M1D_MODE_S1  = 1,
};
enum {
    M1D_SPLIT_NONE = 0,
    M1D_SPLIT_IN   = 1,
    M1D_SPLIT_OUT  = 2,
};

void llama_model_mach1_dense::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    uint32_t m1_version = 0;
    ml.get_key("mach1.format_version", m1_version, false);
    if (m1_version != 4) {
        throw std::runtime_error(format(
            "mach1 dense checkpoint declares format_version %u; this loader takes exactly 4", m1_version));
    }
    GGML_ASSERT(hparams.n_layer_nextn == 0 && "mach1 v4 checkpoints do not ship the MTP head");

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
    auto check_m1_dim = [&](const ggml_tensor * t, int dim, int64_t expected) {
        if (t->ne[dim] != expected) {
            throw std::runtime_error(format("mach1: tensor '%s' has wrong shape: ne[%d] = %lld, expected %lld",
                t->name, dim, (long long) t->ne[dim], (long long) expected));
        }
    };

    auto create_m1w = [&](llm_tensor base, int il, int64_t n_in, int64_t n_out, int split) -> m1w {
        m1w w;
        if (ml.get_tensor_meta(tn(base, "m1_da_trellis", il).str().c_str()) != nullptr) {
            w.trellis = create_m1(tn(base, "m1_da_trellis", il), true);
            w.su      = create_m1(tn(base, "m1_da_su",      il), true);
            w.sv      = create_m1(tn(base, "m1_da_sv",      il), true);
            w.wgamma  = create_m1(tn(base, "m1_da_wgamma",  il), true);
            const int64_t E = w.trellis->ne[2];
            GGML_ASSERT(split != M1D_SPLIT_NONE || E == 1);
            check_m1_dim(w.su, 0, split == M1D_SPLIT_IN  ? n_in/E  : n_in);
            check_m1_dim(w.sv, 0, split == M1D_SPLIT_OUT ? n_out/E : n_out);
            return w;
        }
        w.q  = create_m1(tn(base, "m1_ne_q",  il), true);
        w.mn = create_m1(tn(base, "m1_ne_mn", il), true);
        w.mx = create_m1(tn(base, "m1_ne_mx", il), true);
        check_m1_dim(w.q,  1, n_out);
        check_m1_dim(w.mn, 1, n_out);
        return w;
    };

    m1_layers.resize(n_layer);

    m1_tlut = create_m1(tn(LLM_TENSOR_MACH1_TLUT), true);
    check_m1_dim(m1_tlut, 0, 8);
    check_m1_dim(m1_tlut, 1, 32768);

    m1_embed.trellis = create_m1(tn(LLM_TENSOR_TOKEN_EMBD, "m1_da_trellis"), true);
    m1_embed.su      = create_m1(tn(LLM_TENSOR_TOKEN_EMBD, "m1_da_su"),      true);
    m1_embed.sv      = create_m1(tn(LLM_TENSOR_TOKEN_EMBD, "m1_da_sv"),      true);
    m1_embed.wgamma  = create_m1(tn(LLM_TENSOR_TOKEN_EMBD, "m1_da_wgamma"),  true);
    check_m1_dim(m1_embed.su, 0, n_embd);
    GGML_ASSERT(m1_embed.sv->ne[0]*m1_embed.trellis->ne[2] == n_vocab);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);

    m1_head_hot.q  = create_m1(tn(LLM_TENSOR_OUTPUT, "m1_b4_hot_q"),  true);
    m1_head_hot.mn = create_m1(tn(LLM_TENSOR_OUTPUT, "m1_b4_hot_mn"), true);
    m1_head_hot.mx = create_m1(tn(LLM_TENSOR_OUTPUT, "m1_b4_hot_mx"), true);
    m1_head_cold.trellis = create_m1(tn(LLM_TENSOR_OUTPUT, "m1_b4_trellis"), true);
    m1_head_cold.su      = create_m1(tn(LLM_TENSOR_OUTPUT, "m1_b4_su"),      true);
    m1_head_cold.sv      = create_m1(tn(LLM_TENSOR_OUTPUT, "m1_b4_sv"),      true);
    m1_head_cold.wgamma  = create_m1(tn(LLM_TENSOR_OUTPUT, "m1_b4_wgamma"),  true);
    m1_head_exc_idx      = create_m1(tn(LLM_TENSOR_OUTPUT, "m1_b4_exc_idx"),  true);
    m1_head_exc_rows     = create_m1(tn(LLM_TENSOR_OUTPUT, "m1_b4_exc_rows"), true);
    check_m1_dim(m1_head_cold.su, 0, n_embd);
    check_m1_dim(m1_head_exc_rows, 0, n_embd);
    GGML_ASSERT(m1_head_hot.q->ne[1] + m1_head_cold.sv->ne[0]*m1_head_cold.trellis->ne[2] == n_vocab);

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];
        auto & m1l   = m1_layers[il];

        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, 0);

        if (!hparams.is_recr(il)) {
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, 0);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, 0);

            m1l.wq = create_m1w(LLM_TENSOR_ATTN_Q,   il, n_embd, n_embd_head_k * n_head * 2, M1D_SPLIT_NONE);
            m1l.wk = create_m1w(LLM_TENSOR_ATTN_K,   il, n_embd, n_embd_k_gqa, M1D_SPLIT_NONE);
            m1l.wv = create_m1w(LLM_TENSOR_ATTN_V,   il, n_embd, n_embd_v_gqa, M1D_SPLIT_NONE);
            m1l.wo = create_m1w(LLM_TENSOR_ATTN_OUT, il, n_embd_head_k * n_head, n_embd, M1D_SPLIT_NONE);
        } else {
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, 0);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, 0);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, 0);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, 0);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, n_v_heads }, 0);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, 0);

            m1l.wqkv      = create_m1w(LLM_TENSOR_ATTN_QKV,  il, n_embd, conv_dim, M1D_SPLIT_NONE);
            m1l.wqkv_gate = create_m1w(LLM_TENSOR_ATTN_GATE, il, n_embd, value_dim, M1D_SPLIT_NONE);
            m1l.ssm_out   = create_m1w(LLM_TENSOR_SSM_OUT,   il, value_dim, n_embd, M1D_SPLIT_NONE);
        }

        m1l.gate = create_m1w(LLM_TENSOR_FFN_GATE, il, n_embd, n_ff, M1D_SPLIT_OUT);
        m1l.up   = create_m1w(LLM_TENSOR_FFN_UP,   il, n_embd, n_ff, M1D_SPLIT_OUT);
        m1l.down = create_m1w(LLM_TENSOR_FFN_DOWN, il, n_ff, n_embd, M1D_SPLIT_IN);
    }
}

std::unique_ptr<llm_graph_context> llama_model_mach1_dense::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        throw std::runtime_error("mach1 v4: no MTP head in this checkpoint");
    }
    return std::make_unique<graph>(*this, params);
}

ggml_tensor * llama_model_mach1_dense::graph::da_mm(const m1w & w, ggml_tensor * x, int mode, int split) {
    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx0, x);
    }
    if (w.trellis) {
        return ggml_mach1_da_mm(ctx0, w.trellis, w.su, w.sv, w.wgamma, model.m1_tlut, x,
                                NULL, NULL, mode, split, 0);
    }
    const int bits = (int)(w.q->ne[0]*8/x->ne[0]);
    return ggml_mach1_int_mm(ctx0, w.q, w.mn, w.mx, x, bits);
}

ggml_tensor * llama_model_mach1_dense::graph::v_tiled(ggml_tensor * y) {
    const int64_t hd = hparams.ssm_d_state;
    const int64_t K  = hparams.ssm_n_group;
    const int64_t r  = hparams.ssm_dt_rank / K;
    const int64_t T  = y->ne[1];
    GGML_ASSERT(y->ne[0] == hd*K*r);
    ggml_tensor * t = ggml_reshape_4d(ctx0, y, hd, r, K, T);
    t = ggml_cont(ctx0, ggml_permute(ctx0, t, 0, 2, 1, 3));
    return ggml_reshape_2d(ctx0, t, hd*K*r, T);
}

ggml_tensor * llama_model_mach1_dense::graph::build_inp_embd_mach1() {
    const int64_t n_embd_inp = hparams.n_embd_inp();
    const int64_t n_embd_    = hparams.n_embd;
    GGML_ASSERT(n_embd_inp == n_embd_ && "mach1 v4: no deepstack inputs");

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
    cb(inp->tokens, "inp_tokens", -1);
    ggml_set_input(inp->tokens);
    res->t_inp_tokens = inp->tokens;

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
    cb(inp->embd, "inp_embd", -1);
    ggml_set_input(inp->embd);

    std::array<ggml_tensor *, 2> inps;
    inps[0] = ggml_mach1_da_embed(ctx0, model.m1_embed.trellis, model.m1_embed.su,
                                  model.m1_embed.sv, model.m1_embed.wgamma, model.m1_tlut,
                                  inp->tokens, M1D_MODE_S1);
    inps[1] = inp->embd;

    GGML_ASSERT(ggml_are_same_shape(inps[0], inps[1]));

    ggml_tensor * cur = ggml_build_forward_select(gf, inps.data(), inps.size(), ubatch.token ? 0 : 1);

    res->t_inp_embd = cur;
    res->add_input(std::move(inp));

    return cur;
}

llama_model_mach1_dense::graph::graph(const llama_model_mach1_dense & model, const llm_graph_params & params) :
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
        cb(cur, "post_ffn", il);

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

    if (!ggml_is_contiguous(cur)) {
        cur = ggml_cont(ctx0, cur);
    }
    const int bits = (int)(model.m1_head_hot.q->ne[0]*8/cur->ne[0]);
    ggml_tensor * hot = ggml_mach1_int_mm(ctx0, model.m1_head_hot.q, model.m1_head_hot.mn,
                                          model.m1_head_hot.mx, cur, bits);
    cb(hot, "result_head_hot", -1);

    ggml_tensor * cold = ggml_mach1_da_mm(ctx0, model.m1_head_cold.trellis, model.m1_head_cold.su,
                                          model.m1_head_cold.sv, model.m1_head_cold.wgamma,
                                          model.m1_tlut, cur,
                                          model.m1_head_exc_idx, model.m1_head_exc_rows,
                                          M1D_MODE_S1, M1D_SPLIT_OUT, (int) model.m1_head_hot.q->ne[1]);
    cb(cold, "result_head_cold", -1);

    cur = ggml_concat(ctx0, hot, cold, 0);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_mach1_dense::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = da_mm(model.m1_layers[il].wqkv, input, M1D_MODE_S1, M1D_SPLIT_NONE);
    {
        const int64_t key_dim = hparams.ssm_d_state * hparams.ssm_n_group;
        const int64_t val_dim = hparams.ssm_d_state * hparams.ssm_dt_rank;
        const int64_t T = qkv_mixed->ne[1];
        ggml_tensor * qk = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv_mixed, 2*key_dim, T,
                                                        qkv_mixed->nb[1], 0));
        ggml_tensor * v  = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv_mixed, val_dim, T,
                                                        qkv_mixed->nb[1], 2*key_dim*sizeof(float)));
        qkv_mixed = ggml_concat(ctx0, qk, v_tiled(v), 0);
    }

    ggml_tensor * z = da_mm(model.m1_layers[il].wqkv_gate, input, M1D_MODE_S1, M1D_SPLIT_NONE);
    z = v_tiled(z);
    cb(z, "z", il);

    ggml_build_forward_expand(gf, qkv_mixed);
    ggml_build_forward_expand(gf, z);

    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_mach1_dense::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llama_model_mach1_dense::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * Qcur_full = da_mm(model.m1_layers[il].wq, cur, M1D_MODE_S1, M1D_SPLIT_NONE);
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Kcur = da_mm(model.m1_layers[il].wk, cur, M1D_MODE_S1, M1D_SPLIT_NONE);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = da_mm(model.m1_layers[il].wv, cur, M1D_MODE_S1, M1D_SPLIT_NONE);
    cb(Vcur, "Vcur", il);

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

    cur = da_mm(model.m1_layers[il].wo, cur, M1D_MODE_S1, M1D_SPLIT_NONE);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_mach1_dense::graph::build_layer_attn_linear(
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
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

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

    ggml_tensor * final_output;
    if (num_k_heads != num_v_heads) {
        const int64_t r = num_v_heads / num_k_heads;
        ggml_tensor * t = ggml_reshape_4d(ctx0, attn_out_norm, head_v_dim, num_k_heads, r, n_seq_tokens*n_seqs);
        t = ggml_cont(ctx0, ggml_permute(ctx0, t, 0, 2, 1, 3));
        final_output = ggml_reshape_3d(ctx0, t, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    } else {
        final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    }
    cb(final_output, "final_output", il);

    cur = da_mm(model.m1_layers[il].ssm_out, final_output, M1D_MODE_S1, M1D_SPLIT_NONE);
    cb(cur, "linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_mach1_dense::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    const auto & m1l = model.m1_layers[il];

    ggml_tensor * g = da_mm(m1l.gate, cur, M1D_MODE_SYM, M1D_SPLIT_OUT);
    cb(g, "ffn_gate", il);
    ggml_tensor * u = da_mm(m1l.up, cur, M1D_MODE_SYM, M1D_SPLIT_OUT);
    cb(u, "ffn_up", il);

    ggml_tensor * par = ggml_swiglu_split(ctx0, g, u);
    cb(par, "ffn_swiglu", il);

    cur = da_mm(m1l.down, par, M1D_MODE_SYM, M1D_SPLIT_IN);
    cb(cur, "ffn_out", il);

    return cur;
}
