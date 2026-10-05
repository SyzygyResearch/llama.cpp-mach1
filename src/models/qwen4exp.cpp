#include "models.h"
#include "llama-memory-recurrent.h"
#include "llama-kv-cache.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

static uint64_t q4x_splitmix64(uint64_t v) {
    v = v + 0x9E3779B97F4A7C15ull;
    v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ull;
    v = (v ^ (v >> 27)) * 0x94D049BB133111EBull;
    return v ^ (v >> 31);
}

static bool q4x_is_prime(uint64_t v) {
    if (v < 2) {
        return false;
    }
    if (v % 2 == 0) {
        return v == 2;
    }
    for (uint64_t d = 3; d * d <= v; d += 2) {
        if (v % d == 0) {
            return false;
        }
    }
    return true;
}

static void q4x_ple_ngram_ids_op(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto * p    = (const llama_model_qwen4exp::ple_hash *) userdata;
    const ggml_tensor * hist = dst->src[0];

    GGML_ASSERT(hist->type == GGML_TYPE_F32 && ggml_is_contiguous(hist));
    GGML_ASSERT(dst->type == GGML_TYPE_I32 && ggml_is_contiguous(dst));

    const int64_t n_heads = dst->ne[0];
    const int64_t n_tok   = dst->ne[1];
    const int64_t n_seqs  = dst->ne[2];
    const int64_t L       = hist->ne[0];
    const int64_t ctx     = L - n_tok;
    const int64_t N       = p->ngram_size;
    const int64_t hpn     = p->heads_per_ngram;

    GGML_ASSERT(n_heads == (N - 1) * hpn);
    GGML_ASSERT(ctx == N - 1);

    std::vector<int64_t> tok(L);
    std::vector<int64_t> last_eos(L);

    for (int64_t s = 0; s < n_seqs; ++s) {
        if (s % nth != ith) {
            continue;
        }
        const float * h = (const float *) ((const char *) hist->data + s*hist->nb[1]);
        int64_t le = -1;
        for (int64_t i = 0; i < L; ++i) {
            const int64_t v = (int64_t) h[i];
            tok[i]      = v <= 0 ? (int64_t) p->eos : v - 1;
            last_eos[i] = le;
            if (tok[i] == p->eos) {
                le = i;
            }
        }
        for (int64_t t = 0; t < n_tok; ++t) {
            const int64_t pos = ctx + t;
            int32_t * out = (int32_t *) ((char *) dst->data + t*dst->nb[1] + s*dst->nb[2]);
            uint64_t mixed = 0;
            for (int64_t k = 0; k < N; ++k) {
                const int64_t src = pos - k;
                const bool valid  = src >= 0 && src > last_eos[pos];
                const uint64_t tk = (uint64_t) (valid ? tok[src] : (int64_t) p->eos);
                mixed = k == 0 ? tk * p->mult[0] : (mixed ^ (tk * p->mult[k]));
                if (k >= 1) {
                    const int64_t h0 = (k - 1) * hpn;
                    for (int64_t j = 0; j < hpn; ++j) {
                        const uint64_t m = p->head_size[h0 + j];
                        out[h0 + j] = (int32_t) (mixed % m + p->head_offset[h0 + j]);
                    }
                }
            }
        }
    }
}

static void q4x_qsa_block_mask_op(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    const int topk = *(const int32_t *) userdata;
    const ggml_tensor * sc = dst->src[0];

    GGML_ASSERT(sc->type == GGML_TYPE_F32 && ggml_is_contiguous(sc));
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));

    const int64_t nblk = sc->ne[0];
    const int64_t nt   = sc->ne[1];

    std::vector<int> ord(nblk);

    for (int64_t t = ith; t < nt; t += nth) {
        const float * s = (const float *) sc->data  + t*nblk;
        float       * o = (float       *) dst->data + t*nblk;

        int n_valid = 0;
        for (int64_t b = 0; b < nblk; ++b) {
            if (std::isfinite(s[b])) {
                ord[n_valid++] = (int) b;
            }
        }
        if (n_valid <= topk) {
            for (int64_t b = 0; b < nblk; ++b) {
                o[b] = 0.0f;
            }
            continue;
        }
        std::partial_sort(ord.begin(), ord.begin() + topk, ord.begin() + n_valid,
                [&](int a, int b) { return s[a] > s[b] || (s[a] == s[b] && a < b); });
        for (int64_t b = 0; b < nblk; ++b) {
            o[b] = std::isfinite(s[b]) ? -INFINITY : 0.0f;
        }
        for (int i = 0; i < topk; ++i) {
            o[ord[i]] = 0.0f;
        }
    }
}

class llm_graph_input_q4x : public llm_graph_input_i {
public:
    llm_graph_input_q4x(int64_t n_blk, int64_t blk) : n_blk(n_blk), blk(blk) {}

    void set_input(const llama_ubatch * ubatch) override {
        if (tok) {
            GGML_ASSERT(ubatch->token && "qwen4exp PLE needs token ids (embedding input is not supported)");
            float * d = (float *) tok->data;
            for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
                d[i] = (float) ubatch->token[i] + 1.0f;
            }
        }
        if (blk_pos && blk_pos->data) {
            int32_t * d = (int32_t *) blk_pos->data;
            for (int64_t b = 0; b < n_blk; ++b) {
                d[b] = (int32_t) (b*blk);
            }
        }
        if (blk_valid && blk_valid->data) {
            float * d = (float *) blk_valid->data;
            for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
                const int64_t pos = ubatch->pos[i];
                for (int64_t b = 0; b < n_blk; ++b) {
                    d[i*n_blk + b] = (b + 1)*blk - 1 <= pos ? 0.0f : -INFINITY;
                }
            }
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        bool res = true;
        res &= !tok       || tok->ne[0]       == params.ubatch.n_tokens;
        res &= !blk_valid || blk_valid->ne[1] == params.ubatch.n_tokens;
        return res;
    }

    ggml_tensor * tok       = nullptr;
    ggml_tensor * blk_pos   = nullptr;
    ggml_tensor * blk_valid = nullptr;

    const int64_t n_blk;
    const int64_t blk;
};

void llama_model_qwen4exp::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    {
        std::string gate = "silu";
        ml.get_key(LLM_KV_SSM_OUTPUT_GATE, gate, false);
        GGML_ASSERT((gate == "silu" || gate == "sigmoid") && "qwen4exp: ssm.output_gate must be silu or sigmoid");
        hparams.ssm_gate_sigmoid = gate == "sigmoid";
    }

    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i + 1) % full_attn_interval != 0;
        }
    }

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE, hparams.indexer_block_size);
    hparams.indexer_kv         = true;
    hparams.n_layer_dense_lead = 0;
    qsa_topk_blocks = (int32_t) hparams.indexer_top_k;

    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT, hparams.q4x_hc_count);
    ml.get_key(LLM_KV_HYPER_CONNECTION_RANK,  hparams.q4x_hc_rank);

    std::fill(hparams.ple_layer_arr.begin(), hparams.ple_layer_arr.end(), 0u);
    ml.get_key_or_arr(LLM_KV_PLE_LAYERS, hparams.ple_layer_arr, hparams.n_layer(), false);
    bool any_ple = false;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (hparams.ple_layer_arr[il]) {
            GGML_ASSERT(hparams.is_recr(il) && "qwen4exp: PLE only on linear-attention layers");
            any_ple = true;
        }
    }
    if (any_ple) {
        ml.get_key(LLM_KV_PLE_EMBEDDING_LENGTH, hparams.ple_n_embd);
        ml.get_key(LLM_KV_PLE_CONV_KERNEL,      hparams.ple_conv_kernel);
        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,       hparams.ple_ngram_size);
        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM,  hparams.ple_heads_per_ngram);
        ml.get_key(LLM_KV_PLE_VOCAB_SIZE_BASE,  hparams.ple_vocab_size_base);
        ml.get_key(LLM_KV_PLE_VOCAB_PAD,        hparams.ple_vocab_pad);
        ml.get_key(LLM_KV_PLE_SEED,             hparams.ple_seed);
        ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID,     hparams.ple_eos_token_id);
        GGML_ASSERT(hparams.ple_ngram_size >= 2);

        hparams.n_embd_r_extra = (hparams.ple_conv_kernel - 1) * hparams.ple_ngram_size * hparams.n_embd * hparams.q4x_hc_count
                               + (hparams.ple_ngram_size - 1);
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc      = hparams.q4x_hc_count;
    const int64_t n_hc    = n_embd * hc;
    const int64_t hc_rank = hparams.q4x_hc_rank;

    m1_version = 0;
    ml.get_key("mach1.format_version", m1_version, false);
    if (m1_version != 0 && m1_version != 5) {
        throw std::runtime_error(format("qwen4exp: mach1 format_version %u is not supported (5 only)", m1_version));
    }
    const bool m1 = m1_version != 0;
    auto has = [&](const LLM_TN_IMPL & t) { return m1 && ml.get_tensor_meta(t.str().c_str()) != nullptr; };
    auto create_m1 = [&](const LLM_TN_IMPL & t) -> ggml_tensor * {
        ggml_tensor * meta = ml.get_tensor_meta(t.str().c_str());
        if (meta == nullptr) {
            throw std::runtime_error("qwen4exp: missing codec tensor '" + t.str() + "'");
        }
        std::vector<int64_t> ne(meta->ne, meta->ne + ggml_n_dims(meta));
        switch (ne.size()) {
            case 1: return create_tensor(t, { ne[0] }, 0);
            case 2: return create_tensor(t, { ne[0], ne[1] }, 0);
            default: return create_tensor(t, { ne[0], ne[1], ne[2] }, 0);
        }
    };
    std::vector<ggml_tensor *> ne_tlut_l(n_layer + 1, nullptr);
    auto ne_tlut_at = [&](int il) -> ggml_tensor * {
        ggml_tensor *& t = ne_tlut_l[il + 1];
        if (t == nullptr) {
            if (m1_ne_tlut == nullptr) {
                throw std::runtime_error("qwen4exp: RT spine streams without the mach1.ne_tlut codebook");
            }
            t = create_tensor_for_layer(tn(LLM_TENSOR_MACH1_NE_TLUT), { 2, 512 }, il);
        }
        return t;
    };
    auto lin = [&](llm_tensor t, int il, int64_t n_in, int64_t n_out, ggml_tensor ** w, m1_rt & c, int flags = 0) {
        if (!has(tn(t, "m1_rt_trellis", il))) {
            *w = create_tensor(tn(t, "weight", il), { n_in, n_out }, flags);
            return;
        }
        c.trellis = create_m1(tn(t, "m1_rt_trellis", il));
        c.su      = create_m1(tn(t, "m1_rt_su",      il));
        c.sv      = create_m1(tn(t, "m1_rt_sv",      il));
        c.tlut    = ne_tlut_at(il);
        c.n_out   = n_out;
        const int64_t n = c.su->ne[0];
        const int64_t m = c.sv->ne[0];
        if (n < n_in || m < n_out || c.trellis->ne[0] != 64 || c.trellis->ne[1] != (m/16)*(n/16)) {
            throw std::runtime_error(format("qwen4exp: codec slot '%s' has stream %lld x %lld for a %lld x %lld weight",
                tn(t, "m1_rt_trellis", il).str().c_str(), (long long) m, (long long) n, (long long) n_out, (long long) n_in));
        }
    };

    if (has(tn(LLM_TENSOR_TOKEN_EMBD, "m1_codes"))) {
        m1_embed_codes = create_m1(tn(LLM_TENSOR_TOKEN_EMBD, "m1_codes"));
        m1_embed_lut   = create_m1(tn(LLM_TENSOR_TOKEN_EMBD, "m1_lut"));
        GGML_ASSERT(m1_embed_codes->ne[0] == n_embd/2 && m1_embed_codes->ne[1] == n_vocab);
    } else {
        tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
    }
    if (has(tn(LLM_TENSOR_OUTPUT, "m1_qp"))) {
        m1_head_qp     = create_m1(tn(LLM_TENSOR_OUTPUT, "m1_qp"));
        m1_head_gscale = create_m1(tn(LLM_TENSOR_OUTPUT, "m1_gscale"));
        GGML_ASSERT(m1_head_qp->ne[0] == n_embd/8*5 && m1_head_qp->ne[1] == n_vocab);
        GGML_ASSERT(m1_head_gscale->ne[0] == n_embd/64 && m1_head_gscale->ne[1] == n_vocab);
    } else {
        output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
        if (output == nullptr) {
            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
        }
    }
    if (m1) {
        if (has(tn(LLM_TENSOR_MACH1_NE_TLUT))) {
            m1_ne_tlut = create_tensor(tn(LLM_TENSOR_MACH1_NE_TLUT), { 2, 512 }, 0);
        }
        if (has(tn(LLM_TENSOR_MACH1_TLUT, "m1_d4_zt"))) {
            m1_d4_zt    = create_m1(tn(LLM_TENSOR_MACH1_TLUT, "m1_d4_zt"));
            m1_d4_units = create_m1(tn(LLM_TENSOR_MACH1_TLUT, "m1_d4_units"));
            GGML_ASSERT(m1_d4_zt->ne[0] == 65536 && m1_d4_zt->ne[1] == 5 && m1_d4_units->ne[0] == 5);
            if (ml.get_arr("mach1.expert.hash", m1_d4_hash, false) && m1_d4_hash.size() != 15) {
                throw std::runtime_error("qwen4exp: mach1.expert.hash must be 15 int32");
            }
        }
    }

    hc_out.norm = create_tensor(tn(LLM_TENSOR_QHC_OUT_NORM, "weight"), { n_hc }, 0);
    lin(LLM_TENSOR_QHC_OUT_DOWN, -1, n_hc, hc_rank, &hc_out.down, hc_out.down_rt);
    lin(LLM_TENSOR_QHC_OUT_UP,   -1, hc_rank, n_hc, &hc_out.up,   hc_out.up_rt);

    ple_hashes.assign(n_layer, {});
    int64_t ple_rows = 0;
    {
        const uint64_t n_heads = (uint64_t) (hparams.ple_ngram_size > 0 ? hparams.ple_ngram_size - 1 : 0) * hparams.ple_heads_per_ngram;
        uint64_t prime = hparams.ple_vocab_size_base > 0 ? hparams.ple_vocab_size_base - 1 : 0;
        int ple_index = 0;
        for (int il = 0; il < n_layer; ++il) {
            if (!hparams.ple_layer_arr[il]) {
                continue;
            }
            auto & ph = ple_hashes[il];
            ph.ngram_size      = hparams.ple_ngram_size;
            ph.heads_per_ngram = hparams.ple_heads_per_ngram;
            ph.eos             = (int32_t) hparams.ple_eos_token_id;

            const uint64_t max_long  = (uint64_t) INT64_MAX;
            const uint64_t mult_max  = max_long / std::max<uint64_t>((uint64_t) n_vocab, 1);
            const uint64_t half      = std::max<uint64_t>(1, mult_max / 2);
            const uint64_t base_seed = (uint64_t) hparams.ple_seed + 10007ull * (uint64_t) ple_index;
            for (uint32_t k = 0; k < ph.ngram_size; ++k) {
                const uint64_t v = base_seed + 0x9E3779B97F4A7C15ull * (uint64_t) (k + 1);
                ph.mult.push_back(2 * (q4x_splitmix64(v) % half) + 1);
            }

            uint64_t total = 0;
            for (uint64_t h = 0; h < n_heads; ++h) {
                do { ++prime; } while (!q4x_is_prime(prime));
                ph.head_size.push_back(prime);
                ph.head_offset.push_back((uint64_t) ple_rows + total);
                total += prime;
            }
            const uint64_t pad = std::max<uint32_t>(hparams.ple_vocab_pad, 1);
            ple_rows += (int64_t) ((total + pad - 1) / pad * pad);
            ple_index++;
        }
        if (ple_index > 0) {
            const int64_t hd = hparams.ple_n_embd / (int64_t) n_heads;
            ple_ngram_embd = create_tensor(tn(LLM_TENSOR_PLE_NGRAM_EMBD, "weight"), { hd, ple_rows }, 0);
            GGML_ASSERT(ple_rows < INT32_MAX && "qwen4exp: PLE row ids must fit int32 for get_rows");
        }
    }

    q4x_layers.resize(n_layer);

    const int64_t n_ff_exp   = hparams.n_ff_exp;
    const int64_t n_ff_shexp = hparams.n_ff_shexp;

    const int64_t head_k_dim = hparams.ssm_d_state;
    const int64_t head_v_dim = hparams.ssm_d_inner / hparams.ssm_dt_rank;
    const int64_t n_k_heads  = hparams.ssm_n_group;
    const int64_t n_v_heads  = hparams.ssm_dt_rank;
    const int64_t key_dim    = head_k_dim * n_k_heads;
    const int64_t value_dim  = head_v_dim * n_v_heads;
    const int64_t conv_dim   = key_dim * 2 + value_dim;

    const int64_t n_idx_head = hparams.indexer_n_head;
    const int64_t d_idx      = hparams.indexer_head_size;

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];
        auto & ql    = q4x_layers[il];

        ql.hc_attn.norm   = create_tensor(tn(LLM_TENSOR_QHC_ATTN_NORM,   "weight", il), { n_hc }, 0);
        lin(LLM_TENSOR_QHC_ATTN_DOWN, il, n_hc, hc_rank, &ql.hc_attn.down, ql.hc_attn.down_rt);
        lin(LLM_TENSOR_QHC_ATTN_UP,   il, hc_rank, n_hc, &ql.hc_attn.up,   ql.hc_attn.up_rt);
        ql.hc_attn.inject = create_tensor(tn(LLM_TENSOR_QHC_ATTN_INJECT, "weight", il), { n_hc, hc }, 0);
        ql.hc_ffn.norm    = create_tensor(tn(LLM_TENSOR_QHC_FFN_NORM,    "weight", il), { n_hc }, 0);
        lin(LLM_TENSOR_QHC_FFN_DOWN, il, n_hc, hc_rank, &ql.hc_ffn.down, ql.hc_ffn.down_rt);
        lin(LLM_TENSOR_QHC_FFN_UP,   il, hc_rank, n_hc, &ql.hc_ffn.up,   ql.hc_ffn.up_rt);
        ql.hc_ffn.inject  = create_tensor(tn(LLM_TENSOR_QHC_FFN_INJECT,  "weight", il), { n_hc, hc }, 0);

        if (!hparams.is_recr(il)) {
            lin(LLM_TENSOR_ATTN_Q,   il, n_embd, n_embd_head_k * n_head * 2, &layer.wq, ql.wq);
            lin(LLM_TENSOR_ATTN_K,   il, n_embd, n_embd_k_gqa,               &layer.wk, ql.wk);
            lin(LLM_TENSOR_ATTN_V,   il, n_embd, n_embd_v_gqa,               &layer.wv, ql.wv);
            lin(LLM_TENSOR_ATTN_OUT, il, n_embd_head_k * n_head, n_embd,     &layer.wo, ql.wo);
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, 0);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, 0);

            if (has(tn(LLM_TENSOR_INDEXER_Q_PROJ, "m1_rt_trellis", il))) {
                lin(LLM_TENSOR_INDEXER_Q_PROJ, il, n_embd, n_idx_head * d_idx + d_idx, &layer.index_q_proj, ql.idx_qk);
            } else {
                layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, n_idx_head * d_idx }, 0);
                layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, d_idx }, 0);
            }
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { d_idx }, 0);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { d_idx }, 0);
        } else {
            lin(LLM_TENSOR_ATTN_QKV,  il, n_embd, conv_dim,  &layer.wqkv,      ql.wqkv);
            lin(LLM_TENSOR_ATTN_GATE, il, n_embd, value_dim, &layer.wqkv_gate, ql.wqkv_gate);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, 0);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { n_v_heads }, 0);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { n_v_heads }, 0);
            lin(LLM_TENSOR_SSM_BETA,  il, n_embd, n_v_heads, &layer.ssm_beta,  ql.ssm_beta);
            lin(LLM_TENSOR_SSM_ALPHA, il, n_embd, n_v_heads, &layer.ssm_alpha, ql.ssm_alpha);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, 0);
            lin(LLM_TENSOR_SSM_OUT,   il, value_dim, n_embd, &layer.ssm_out, ql.ssm_out);

            if (hparams.ple_layer_arr[il]) {
                const int64_t pe = hparams.ple_n_embd;
                lin(LLM_TENSOR_PLE_KEY_PROJ,   il, pe, n_hc,   &ql.ple.key_proj,   ql.ple.key_rt);
                lin(LLM_TENSOR_PLE_VALUE_PROJ, il, pe, n_embd, &ql.ple.value_proj, ql.ple.value_rt);
                ql.ple.key_norm   = create_tensor(tn(LLM_TENSOR_PLE_KEY_NORM,   "weight", il), { n_hc }, 0);
                ql.ple.query_norm = create_tensor(tn(LLM_TENSOR_PLE_QUERY_NORM, "weight", il), { n_hc }, 0);
                ql.ple.conv_norm  = create_tensor(tn(LLM_TENSOR_PLE_CONV_NORM,  "weight", il), { n_hc }, 0);
                ql.ple.conv1d     = create_tensor(tn(LLM_TENSOR_PLE_CONV1D,     "weight", il), { n_hc, hparams.ple_conv_kernel }, 0);
            }
        }

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, 0);
        if (has(tn(LLM_TENSOR_FFN_GATE_EXPS, "m1_d4_trellis", il))) {
            GGML_ASSERT(m1_d4_zt && "qwen4exp: D4 experts without the mach1.tlut.m1_d4_zt codebook");
            const llm_tensor exps[3] = { LLM_TENSOR_FFN_GATE_EXPS, LLM_TENSOR_FFN_UP_EXPS, LLM_TENSOR_FFN_DOWN_EXPS };
            for (int p = 0; p < 3; ++p) {
                auto & e = ql.exps[p];
                e.trellis = create_m1(tn(exps[p], "m1_d4_trellis", il));
                e.offs    = create_m1(tn(exps[p], "m1_d4_offs",    il));
                e.su      = create_m1(tn(exps[p], "m1_su",         il));
                e.sv      = create_m1(tn(exps[p], "m1_sv",         il));
                e.gw      = create_m1(tn(exps[p], "m1_d4_gw",      il));
                e.zt      = p == 0 ? create_tensor_for_layer(tn(LLM_TENSOR_MACH1_TLUT, "m1_d4_zt"),    { 65536, 5 }, il) : ql.exps[0].zt;
                e.units   = p == 0 ? create_tensor_for_layer(tn(LLM_TENSOR_MACH1_TLUT, "m1_d4_units"), { 5 },        il) : ql.exps[0].units;
                const int64_t w_in  = p == 2 ? n_ff_exp : n_embd;
                const int64_t w_out = p == 2 ? n_embd   : n_ff_exp;
                GGML_ASSERT(e.su->ne[0] == w_in && e.sv->ne[0] == w_out && e.offs->ne[1] == n_expert);
                GGML_ASSERT(e.gw->ne[0] == w_in/16 + w_out/16);
            }
        } else {
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, 0);
            create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, 0);
        }

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, 0);
        lin(LLM_TENSOR_FFN_GATE_SHEXP, il, n_embd, n_ff_shexp, &layer.ffn_gate_shexp, ql.gate_shexp);
        lin(LLM_TENSOR_FFN_UP_SHEXP,   il, n_embd, n_ff_shexp, &layer.ffn_up_shexp,   ql.up_shexp);
        lin(LLM_TENSOR_FFN_DOWN_SHEXP, il, n_ff_shexp, n_embd, &layer.ffn_down_shexp, ql.down_shexp);
    }

    bool any_rt = false;
    for (const auto & ql : q4x_layers) {
        any_rt |= ql.wq.trellis || ql.wk.trellis || ql.wv.trellis || ql.wo.trellis || ql.idx_qk.trellis ||
                  ql.wqkv.trellis || ql.wqkv_gate.trellis || ql.ssm_alpha.trellis || ql.ssm_beta.trellis ||
                  ql.ssm_out.trellis || ql.gate_shexp.trellis || ql.up_shexp.trellis || ql.down_shexp.trellis ||
                  ql.hc_attn.down_rt.trellis || ql.hc_attn.up_rt.trellis || ql.hc_ffn.down_rt.trellis ||
                  ql.hc_ffn.up_rt.trellis || ql.ple.key_rt.trellis || ql.ple.value_rt.trellis;
    }
    any_rt |= hc_out.down_rt.trellis || hc_out.up_rt.trellis;
    if (any_rt && m1_ne_tlut == nullptr) {
        throw std::runtime_error("qwen4exp: RT spine streams without the mach1.ne_tlut codebook");
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen4exp::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen4exp::graph::graph(const llama_model_qwen4exp & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t hc = hparams.q4x_hc_count;

    GGML_ASSERT(cparams.n_rs_seq == 0 && "qwen4exp: recurrent-state rollback is not supported");

    ggml_tensor * inpL = model.m1_embed_codes ? build_inp_embd_q4x() : build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const auto * kq_mask = inp->get_attn()->get_kq_mask();
    const int64_t n_kv   = kq_mask->ne[0];
    const int64_t blk    = hparams.indexer_block_size;
    const int64_t n_blk  = n_kv / blk;

    auto q4x_inp = std::make_unique<llm_graph_input_q4x>(n_blk, blk);
    bool any_ple = false;
    for (int il = 0; il < n_layer; ++il) {
        any_ple |= hparams.ple_layer_arr[il] != 0;
    }
    if (any_ple) {
        q4x_inp->tok = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_tokens);
        ggml_set_input(q4x_inp->tok);
    }
    if (n_blk > 0) {
        q4x_inp->blk_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_blk);
        ggml_set_input(q4x_inp->blk_pos);
        q4x_inp->blk_valid = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_blk, n_tokens);
        ggml_set_input(q4x_inp->blk_valid);
    }
    auto * q4x = (llm_graph_input_q4x *) res->add_input(std::move(q4x_inp));

    ggml_tensor * H = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens), n_embd, hc, n_tokens, 1);
    H = ggml_reshape_2d(ctx0, H, n_embd*hc, n_tokens);
    cb(H, "hc_streams", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & ql = model.q4x_layers[il];

        ggml_tensor * rs = nullptr;
        if (hparams.is_recr(il)) {
            const auto * mctx_cur = inp->get_recr()->mctx;
            rs = build_rs(inp->get_recr(), mctx_cur->get_r_l(il), hparams.n_embd_r(), ubatch.n_seqs);
            cb(rs, "conv_states", il);
        }

        if (hparams.ple_layer_arr[il]) {
            H = build_ple(H, rs, inp->get_recr(), q4x->tok, il);
            cb(H, "ple_out", il);
        }

        {
            ggml_tensor * inj = nullptr;
            ggml_tensor * x = build_hc_pre(H, ql.hc_attn, &inj, "hc_attn", il);
            if (hparams.is_recr(il)) {
                x = build_layer_gdn(inp->get_recr(), rs, x, il);
            } else {
                x = build_layer_qsa(inp->get_attn(), x, inp_pos, q4x->blk_pos, q4x->blk_valid, il);
            }
            cb(x, "mixer_out", il);
            H = build_hc_post(H, x, inj, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            H = ggml_get_rows(ctx0, H, inp_out_ids);
        }

        {
            ggml_tensor * inj = nullptr;
            ggml_tensor * x = build_hc_pre(H, ql.hc_ffn, &inj, "hc_ffn", il);
            x = build_layer_ffn(x, il);
            H = build_hc_post(H, x, inj, il);
        }

        cb(H, "l_out", il);
    }

    ggml_tensor * cur = build_hc_pre(H, model.hc_out, nullptr, "hc_out", -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if (model.m1_head_qp) {
        cur = ggml_mach1_head_mm(ctx0, model.m1_head_qp, model.m1_head_gscale, ggml_is_contiguous(cur) ? cur : ggml_cont(ctx0, cur));
    } else {
        cur = build_lora_mm(model.output, cur);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llama_model_qwen4exp::graph::lin(ggml_tensor * w, const m1_rt & c, ggml_tensor * x) {
    if (c.trellis == nullptr) {
        return build_lora_mm(w, x);
    }
    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx0, x);
    }
    if (x->ne[0] < c.su->ne[0]) {
        x = ggml_pad(ctx0, x, (int) (c.su->ne[0] - x->ne[0]), 0, 0, 0);
    }
    ggml_tensor * y = ggml_mach1_rt_mm(ctx0, c.trellis, c.su, c.sv, c.tlut, x);
    if (c.n_out < y->ne[0]) {
        y = ggml_cont(ctx0, ggml_view_4d(ctx0, y, c.n_out, y->ne[1], y->ne[2], y->ne[3], y->nb[1], y->nb[2], y->nb[3], 0));
    }
    return y;
}

static bool m1_skip_vheads() {
    static const bool on = getenv("GGML_MACH1_SKIP") != nullptr && (atoi(getenv("GGML_MACH1_SKIP")) & 32) != 0;
    return on;
}

ggml_tensor * llama_model_qwen4exp::graph::v_heads(ggml_tensor * y, int64_t hd, bool to_tiled) {
    const int64_t K = hparams.ssm_n_group;
    const int64_t r = hparams.ssm_dt_rank / K;
    if (r == 1 || m1_skip_vheads()) {
        return y;
    }
    const int64_t T = ggml_nelements(y) / (hd*K*r);
    GGML_ASSERT(y->ne[0] == hd*K*r);
    ggml_tensor * t = ggml_reshape_4d(ctx0, ggml_is_contiguous(y) ? y : ggml_cont(ctx0, y), hd, to_tiled ? r : K, to_tiled ? K : r, T);
    t = ggml_cont(ctx0, ggml_permute(ctx0, t, 0, 2, 1, 3));
    return ggml_reshape(ctx0, t, y);
}

ggml_tensor * llama_model_qwen4exp::graph::build_inp_embd_q4x() {
    const int64_t n_embd_inp = hparams.n_embd_inp();
    GGML_ASSERT(n_embd_inp == hparams.n_embd);

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
    cb(inp->tokens, "inp_tokens", -1);
    ggml_set_input(inp->tokens);
    res->t_inp_tokens = inp->tokens;

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
    cb(inp->embd, "inp_embd", -1);
    ggml_set_input(inp->embd);

    std::array<ggml_tensor *, 2> inps;
    inps[0] = ggml_mach1_embed_gather(ctx0, model.m1_embed_codes, model.m1_embed_lut, inp->tokens);
    inps[1] = inp->embd;
    GGML_ASSERT(ggml_are_same_shape(inps[0], inps[1]));

    ggml_tensor * cur = ggml_build_forward_select(gf, inps.data(), inps.size(), ubatch.token ? 0 : 1);
    res->t_inp_embd = cur;
    res->add_input(std::move(inp));
    ggml_build_forward_expand(gf, cur);
    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_pre(ggml_tensor * h, const hc_block & hcb, ggml_tensor ** inj, const char * tag, int il) {
    const int64_t hc = hparams.q4x_hc_count;
    const int64_t nt = h->ne[1];

    ggml_tensor * h3 = ggml_reshape_3d(ctx0, h, n_embd, hc, nt);
    ggml_tensor * hn = ggml_rms_norm(ctx0, h3, hparams.f_norm_rms_eps);
    hn = ggml_mul(ctx0, hn, ggml_reshape_2d(ctx0, hcb.norm, n_embd, hc));
    ggml_tensor * hn2 = ggml_reshape_2d(ctx0, hn, n_embd*hc, nt);

    if (inj) {
        ggml_tensor * w = build_lora_mm(hcb.inject, hn2);
        w = ggml_scale(ctx0, ggml_sigmoid(ctx0, ggml_scale(ctx0, w, 1.0f/(float) hc)), 2.0f);
        ggml_build_forward_expand(gf, w);
        *inj = w;
    }

    ggml_tensor * m = lin(hcb.down, hcb.down_rt, hn2);
    m = ggml_silu(ctx0, ggml_scale(ctx0, m, 1.0f/(float) hc));
    m = ggml_sigmoid(ctx0, lin(hcb.up, hcb.up_rt, m));
    m = ggml_mul(ctx0, ggml_reshape_3d(ctx0, m, n_embd, hc, nt), hn);

    ggml_build_forward_expand(gf, m);
    std::array<ggml_tensor *, 16> sv = {};
    GGML_ASSERT(hc <= (int64_t) sv.size());
    for (int64_t s = 0; s < hc; ++s) {
        sv[s] = ggml_view_2d(ctx0, m, n_embd, nt, m->nb[2], s*m->nb[1]);
        ggml_build_forward_expand(gf, sv[s]);
    }
    ggml_tensor * x = hc > 1 ? sv[0] : ggml_cont(ctx0, sv[0]);
    for (int64_t s = 1; s < hc; ++s) {
        x = ggml_add(ctx0, x, sv[s]);
        ggml_build_forward_expand(gf, x);
    }
    x = ggml_scale(ctx0, x, 1.0f/(float) hc);
    cb(x, tag, il);
    ggml_build_forward_expand(gf, x);
    return x;
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_post(ggml_tensor * h, ggml_tensor * out, ggml_tensor * inj, int il) {
    const int64_t hc = hparams.q4x_hc_count;
    const int64_t nt = h->ne[1];

    ggml_tensor * o = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, out, n_embd, 1, nt), n_embd, hc, nt, 1);
    o = ggml_mul(ctx0, o, ggml_reshape_3d(ctx0, inj, 1, hc, nt));
    o = ggml_reshape_2d(ctx0, o, n_embd*hc, nt);
    GGML_UNUSED(il);
    return ggml_add(ctx0, h, o);
}

ggml_tensor * llama_model_qwen4exp::graph::build_ple(ggml_tensor * h, ggml_tensor * rs, llm_graph_input_rs * inp, ggml_tensor * tok, int il) {
    const auto & pl  = model.q4x_layers[il].ple;
    const auto & ph  = model.ple_hashes[il];
    const int64_t hc = hparams.q4x_hc_count;
    const int64_t C2 = n_embd*hc;

    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    GGML_ASSERT(ubatch.equal_seqs());

    const int64_t N       = hparams.ple_ngram_size;
    const int64_t ctx_len = N - 1;
    const int64_t kp      = hparams.ple_conv_kernel;
    const int64_t dil     = N;
    const int64_t s_len   = (kp - 1)*dil;
    const int64_t n_heads = (N - 1)*hparams.ple_heads_per_ngram;

    const auto * mctx_cur = inp->mctx;
    ggml_tensor * r_all   = mctx_cur->get_r_l(il);
    const auto kv_head    = mctx_cur->get_head();
    const size_t esz      = ggml_element_size(r_all);

    const int64_t off_conv = (hparams.ssm_d_conv - 1) * (hparams.ssm_d_inner + 2*hparams.ssm_n_group*hparams.ssm_d_state);
    const int64_t off_tok  = off_conv + s_len*C2;
    GGML_ASSERT((int64_t) hparams.n_embd_r() == off_tok + ctx_len);

    ggml_tensor * prev_tok = ggml_view_2d(ctx0, rs, ctx_len, n_seqs, rs->nb[1], off_tok*esz);
    ggml_tensor * hist = ggml_concat(ctx0, ggml_cont(ctx0, prev_tok), ggml_reshape_2d(ctx0, tok, n_seq_tokens, n_seqs), 0);
    cb(hist, "ple_tok_hist", il);
    {
        ggml_tensor * last = ggml_view_2d(ctx0, hist, ctx_len, n_seqs, hist->nb[1], n_seq_tokens*esz);
        ggml_tensor * dst  = ggml_view_2d(ctx0, r_all, ctx_len, n_seqs, r_all->nb[1], kv_head*r_all->nb[1] + off_tok*esz);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, last, dst));
    }

    ggml_tensor * srcs[1] = { hist };
    ggml_tensor * ids = ggml_custom_4d(ctx0, GGML_TYPE_I32, n_heads, n_seq_tokens, n_seqs, 1, srcs, 1,
            q4x_ple_ngram_ids_op, GGML_N_TASKS_MAX, const_cast<ple_hash *>(&ph));
    cb(ids, "ple_ngram_ids", il);

    ggml_tensor * e = ggml_get_rows(ctx0, model.ple_ngram_embd, ggml_reshape_1d(ctx0, ids, n_heads*n_tokens));
    e = ggml_reshape_2d(ctx0, e, hparams.ple_n_embd, n_tokens);
    cb(e, "ple_embd", il);

    ggml_tensor * key = lin(pl.key_proj, pl.key_rt, e);
    key = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, key, n_embd, hc, n_tokens), hparams.f_norm_rms_eps);
    key = ggml_mul(ctx0, key, ggml_reshape_2d(ctx0, pl.key_norm, n_embd, hc));

    ggml_tensor * value = lin(pl.value_proj, pl.value_rt, e);

    ggml_tensor * q = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, h, n_embd, hc, n_tokens), hparams.f_norm_rms_eps);
    q = ggml_mul(ctx0, q, ggml_reshape_2d(ctx0, pl.query_norm, n_embd, hc));

    ggml_tensor * g = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, q));
    g = ggml_scale(ctx0, g, 1.0f/sqrtf((float) n_embd));
    g = ggml_mul(ctx0, ggml_sgn(ctx0, g), ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, g), 1e-6f, INFINITY)));
    cb(g, "ple_gate", il);

    ggml_tensor * gv = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens), n_embd, hc, n_tokens, 1);
    gv = ggml_mul(ctx0, gv, ggml_sigmoid(ctx0, g));

    ggml_tensor * gvn = ggml_rms_norm(ctx0, gv, hparams.f_norm_rms_eps);
    gvn = ggml_mul(ctx0, gvn, ggml_reshape_2d(ctx0, pl.conv_norm, n_embd, hc));

    ggml_tensor * st   = ggml_view_3d(ctx0, rs, C2, s_len, n_seqs, C2*esz, rs->nb[1], off_conv*esz);
    ggml_tensor * xpad = ggml_concat(ctx0, ggml_cont(ctx0, st), ggml_reshape_3d(ctx0, gvn, C2, n_seq_tokens, n_seqs), 1);
    cb(xpad, "ple_conv_in", il);
    {
        ggml_tensor * last = ggml_view_3d(ctx0, xpad, C2, s_len, n_seqs, xpad->nb[1], xpad->nb[2], n_seq_tokens*xpad->nb[1]);
        ggml_tensor * dst  = ggml_view_2d(ctx0, r_all, s_len*C2, n_seqs, r_all->nb[1], kv_head*r_all->nb[1] + off_conv*esz);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, last, dst));
    }
    ggml_tensor * conv = nullptr;
    for (int64_t k = 0; k < kp; ++k) {
        ggml_tensor * xk = ggml_view_3d(ctx0, xpad, C2, n_seq_tokens, n_seqs, xpad->nb[1], xpad->nb[2], k*dil*xpad->nb[1]);
        ggml_tensor * wk = ggml_view_1d(ctx0, pl.conv1d, C2, k*pl.conv1d->nb[1]);
        ggml_tensor * t  = ggml_mul(ctx0, xk, wk);
        conv = conv ? ggml_add(ctx0, conv, t) : t;
    }
    conv = ggml_silu(ctx0, conv);
    cb(conv, "ple_conv", il);

    ggml_tensor * out = ggml_add(ctx0, ggml_reshape_2d(ctx0, gv, C2, n_tokens), ggml_reshape_2d(ctx0, conv, C2, n_tokens));
    return ggml_add(ctx0, h, out);
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_qsa(
        llm_graph_input_attn_kv * inp_attn,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        ggml_tensor *             blk_pos,
        ggml_tensor *             blk_valid,
        int                       il) {
    const auto & layer = model.layers[il];
    const auto & ql    = model.q4x_layers[il];

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t d_idx       = hparams.indexer_head_size;
    const int64_t n_idx_head  = hparams.indexer_n_head;
    const int64_t blk         = hparams.indexer_block_size;

    GGML_ASSERT(!inp_attn->self_k_rot && !inp_attn->self_v_rot && "qwen4exp: attn-rot not supported");

    ggml_tensor * qg = lin(layer.wq, ql.wq, cur);
    ggml_tensor * Qcur = ggml_view_3d(ctx0, qg, n_embd_head, n_head, n_tokens,
            ggml_element_size(qg)*n_embd_head*2, ggml_element_size(qg)*n_embd_head*2*n_head, 0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * gate = ggml_view_3d(ctx0, qg, n_embd_head, n_head, n_tokens,
            ggml_element_size(qg)*n_embd_head*2, ggml_element_size(qg)*n_embd_head*2*n_head, ggml_element_size(qg)*n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head*n_head, n_tokens);

    ggml_tensor * Kcur = lin(layer.wk, ql.wk, cur);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * Vcur = lin(layer.wv, ql.wv, cur);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    ggml_tensor * iq;
    ggml_tensor * ik;
    if (ql.idx_qk.trellis) {
        ggml_tensor * qk = lin(nullptr, ql.idx_qk, cur);
        iq = ggml_cont(ctx0, ggml_view_2d(ctx0, qk, n_idx_head*d_idx, n_tokens, qk->nb[1], 0));
        ik = ggml_cont(ctx0, ggml_view_2d(ctx0, qk, d_idx, n_tokens, qk->nb[1], n_idx_head*d_idx*ggml_element_size(qk)));
    } else {
        iq = build_lora_mm(layer.index_q_proj, cur);
        ik = build_lora_mm(layer.index_k_proj, cur);
    }
    iq = ggml_reshape_3d(ctx0, iq, d_idx, n_idx_head, n_tokens);
    iq = build_norm(iq, layer.index_q_norm, nullptr, LLM_NORM_RMS, il);
    iq = ggml_rope_ext(ctx0, iq, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    ik = ggml_reshape_3d(ctx0, ik, d_idx, 1, n_tokens);

    const auto * mctx_cur = inp_attn->mctx;

    ggml_build_forward_expand(gf, Qcur);
    ggml_build_forward_expand(gf, Vcur);
    ggml_build_forward_expand(gf, Kcur);
    ggml_build_forward_expand(gf, mctx_cur->cpy_k    (ctx0, Kcur, inp_attn->get_k_idxs(), il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v    (ctx0, Vcur, inp_attn->get_v_idxs(), il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_k_idx(ctx0, ik,   inp_attn->get_k_idxs(), il));

    ggml_tensor * kq_mask = inp_attn->get_kq_mask();
    const int64_t n_kv = kq_mask->ne[0];
    GGML_ASSERT(kq_mask->ne[3] == 1 && "qwen4exp QSA: one KV stream per ubatch");
    GGML_ASSERT(kq_mask->ne[1] == n_tokens);

    ggml_tensor * mask = kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);

    const int64_t n_blk = n_kv / blk;
    static const bool qsa_cpu = getenv("GGML_Q4X_QSA_CPU") != nullptr && atoi(getenv("GGML_Q4X_QSA_CPU")) != 0;
    const int64_t topk = model.qsa_topk_blocks;
    if (n_blk > 0 && (qsa_cpu || (topk > 0 && n_blk > topk))) {
        ggml_tensor * kc = mctx_cur->get_k_idx(ctx0, il);
        ggml_tensor * kb = ggml_view_3d(ctx0, kc, d_idx, blk, n_blk, kc->nb[2], blk*kc->nb[2], 0);
        kb = ggml_cont(ctx0, ggml_permute(ctx0, kb, 1, 0, 2, 3));
        kb = ggml_scale(ctx0, ggml_sum_rows(ctx0, kb), 1.0f/(float) blk);
        kb = ggml_reshape_3d(ctx0, kb, d_idx, 1, n_blk);
        kb = build_norm(kb, layer.index_k_norm, nullptr, LLM_NORM_RMS, il);
        kb = ggml_rope_ext(ctx0, kb, blk_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        kb = ggml_reshape_2d(ctx0, kb, d_idx, n_blk);
        cb(kb, "qsa_block_keys", il);

        ggml_tensor * sc = ggml_mul_mat(ctx0, kb, ggml_reshape_2d(ctx0, iq, d_idx, n_idx_head*n_tokens));
        ggml_mul_mat_set_prec(sc, GGML_PREC_F32);
        sc = ggml_relu(ctx0, sc);
        sc = ggml_reshape_3d(ctx0, sc, n_blk, n_idx_head, n_tokens);
        sc = ggml_cont(ctx0, ggml_permute(ctx0, sc, 1, 0, 2, 3));
        sc = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, sc), n_blk, n_tokens);
        sc = ggml_scale(ctx0, sc, 1.0f/sqrtf((float) d_idx));
        sc = ggml_add(ctx0, sc, blk_valid);
        cb(sc, "qsa_block_scores", il);

        ggml_tensor * bm;
        if (qsa_cpu) {
            ggml_tensor * srcs[1] = { sc };
            bm = ggml_custom_4d(ctx0, GGML_TYPE_F32, n_blk, n_tokens, 1, 1, srcs, 1,
                    q4x_qsa_block_mask_op, GGML_N_TASKS_MAX, const_cast<int32_t *>(&model.qsa_topk_blocks));
        } else {
            ggml_tensor * scc = ggml_clamp(ctx0, sc, -1e30f, 1e30f);
            ggml_tensor * ord = ggml_argsort(ctx0, scc, GGML_SORT_ORDER_DESC);
            ggml_tensor * ik  = ggml_cont(ctx0, ggml_view_2d(ctx0, ord, 1, n_tokens, ord->nb[1],
                    (size_t) (topk - 1)*ggml_element_size(ord)));
            ggml_tensor * thr = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, scc, 1, n_blk, n_tokens), ik);
            thr = ggml_reshape_2d(ctx0, thr, 1, n_tokens);
            ggml_tensor * drop = ggml_step(ctx0, ggml_neg(ctx0, ggml_sub(ctx0, scc, thr)));
            ggml_tensor * vis  = ggml_step(ctx0, ggml_scale_bias(ctx0, scc, 1.0f, 1e29f));
            bm = ggml_scale(ctx0, ggml_mul(ctx0, drop, vis), -1e30f);
        }
        cb(bm, "qsa_block_mask", il);

        ggml_tensor * tm = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, bm, 1, n_blk, n_tokens), blk, n_blk, n_tokens, 1);
        tm = ggml_reshape_2d(ctx0, tm, blk*n_blk, n_tokens);
        if (blk*n_blk < n_kv) {
            tm = ggml_pad(ctx0, tm, (int) (n_kv - blk*n_blk), 0, 0, 0);
        }
        mask = ggml_add(ctx0, ggml_reshape_4d(ctx0, tm, n_kv, n_tokens, 1, 1), mask);
    }
    if (mask->type != kq_mask->type) {
        mask = ggml_cast(ctx0, mask, kq_mask->type);
    }
    cb(mask, "qsa_mask", il);

    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    const float kq_scale = 1.0f/sqrtf((float) n_embd_head);
    cur = build_attn_mha(Qcur, k, v, nullptr, mask, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cur = lin(layer.wo, ql.wo, cur);
    cb(cur, "attn_output", il);
    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_gdn(llm_graph_input_rs * inp, ggml_tensor * rs, ggml_tensor * cur, int il) {
    const auto & layer    = model.layers[il];
    const auto & ql       = model.q4x_layers[il];
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

    ggml_tensor * qkv_mixed = lin(layer.wqkv, ql.wqkv, cur);
    if (ql.wqkv.trellis && !m1_skip_vheads()) {
        const int64_t qk_dim = 2*head_k_dim*num_k_heads;
        ggml_tensor * qk = ggml_view_2d(ctx0, qkv_mixed, qk_dim, qkv_mixed->ne[1], qkv_mixed->nb[1], 0);
        ggml_tensor * v  = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv_mixed, qkv_mixed->ne[0] - qk_dim, qkv_mixed->ne[1],
                    qkv_mixed->nb[1], qk_dim*ggml_element_size(qkv_mixed)));
        qkv_mixed = ggml_concat(ctx0, qk, v_heads(v, head_v_dim, true), 0);
    }
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    ggml_tensor * z = lin(layer.wqkv_gate, ql.wqkv_gate, cur);
    if (ql.wqkv_gate.trellis && !(ql.ssm_out.trellis && num_v_heads / num_k_heads > 1)) {
        z = v_heads(z, head_v_dim, true);
    }

    ggml_tensor * beta = lin(layer.ssm_beta, ql.ssm_beta, cur);
    if (ql.ssm_beta.trellis) {
        beta = v_heads(beta, 1, true);
    }
    beta = ggml_sigmoid(ctx0, ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs));

    ggml_tensor * alpha = lin(layer.ssm_alpha, ql.ssm_alpha, cur);
    if (ql.ssm_alpha.trellis) {
        alpha = v_heads(alpha, 1, true);
    }
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor * gate = ggml_mul(ctx0, ggml_softplus(ctx0, ggml_add(ctx0, alpha, layer.ssm_dt)), layer.ssm_a);
    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * r_all = mctx_cur->get_r_l(il);
    const int64_t kconv    = layer.ssm_conv1d->ne[0];
    const int64_t channels = d_inner + 2*hparams.ssm_n_group*hparams.ssm_d_state;
    const int64_t row_cnt  = (kconv - 1)*channels;

    ggml_tensor * conv_states = ggml_view_3d(ctx0, rs, kconv - 1, channels, n_seqs,
            ggml_row_size(rs->type, kconv - 1), rs->nb[1], 0);
    conv_states = ggml_cont(ctx0, conv_states);
    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, ggml_transpose(ctx0, qkv_mixed), 0);
    cb(conv_input, "conv_input", il);
    {
        ggml_tensor * last = ggml_view_3d(ctx0, conv_input, kconv - 1, channels, n_seqs,
                conv_input->nb[1], conv_input->nb[2], ggml_row_size(conv_input->type, n_seq_tokens));
        ggml_tensor * dst  = ggml_view_2d(ctx0, r_all, row_cnt, n_seqs, r_all->nb[1], mctx_cur->get_head()*r_all->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, last, dst));
    }

    ggml_tensor * conv_out = ggml_silu(ctx0, ggml_ssm_conv(ctx0, conv_input, layer.ssm_conv1d));
    cb(conv_out, "conv_output_silu", il);

    const int64_t qkv_dim = head_k_dim*num_k_heads*2 + head_v_dim*num_v_heads;
    const size_t  nb1_qkv = ggml_row_size(conv_out->type, qkv_dim);

    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_out, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_out->type, head_k_dim), nb1_qkv, nb1_qkv*n_seq_tokens, 0);
    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_out, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_out->type, head_k_dim), nb1_qkv, nb1_qkv*n_seq_tokens,
            ggml_row_size(conv_out->type, head_k_dim*num_k_heads));
    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_out, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_out->type, head_v_dim), nb1_qkv, nb1_qkv*n_seq_tokens,
            ggml_row_size(conv_out->type, 2*head_k_dim*num_k_heads));

    q_conv = ggml_scale(ctx0, ggml_rms_norm(ctx0, q_conv, 1e-6f/(float) head_k_dim), 1.0f/sqrtf((float) head_k_dim));
    k_conv = ggml_scale(ctx0, ggml_rms_norm(ctx0, k_conv, 1e-6f/(float) head_k_dim), 1.0f/sqrtf((float) head_k_dim));

    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    const int64_t r_v = num_v_heads / num_k_heads;
    const bool grouped = ql.wqkv_gate.trellis && ql.ssm_out.trellis && r_v > 1 && !m1_skip_vheads();
    ggml_tensor * z4;
    ggml_tensor * on;
    if (grouped) {
        z4 = ggml_reshape_4d(ctx0, z, head_v_dim, r_v, num_k_heads, n_seq_tokens*n_seqs);
        ggml_tensor * og = ggml_reshape_4d(ctx0, output, head_v_dim, num_k_heads, r_v, n_seq_tokens*n_seqs);
        on = build_norm(ggml_permute(ctx0, og, 0, 2, 1, 3), layer.ssm_norm, nullptr, LLM_NORM_RMS, il);
    } else {
        z4 = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
        on = build_norm(output, layer.ssm_norm, nullptr, LLM_NORM_RMS, il);
    }
    on = ggml_mul(ctx0, on, hparams.ssm_gate_sigmoid ? ggml_sigmoid(ctx0, z4) : ggml_silu(ctx0, z4));

    cur = ggml_reshape_3d(ctx0, on, head_v_dim*num_v_heads, n_seq_tokens, n_seqs);
    if (ql.ssm_out.trellis && !grouped) {
        cur = v_heads(cur, head_v_dim, false);
    }
    cur = lin(layer.ssm_out, ql.ssm_out, cur);
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens*n_seqs);
    cb(cur, "linear_attn_out", il);
    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(ggml_tensor * cur, int il) {
    const auto & layer = model.layers[il];
    const auto & ql    = model.q4x_layers[il];

    ggml_tensor * moe_out = ql.exps[0].trellis ? build_moe_d4(cur, il) : build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, layer.ffn_gate_up_exps,
            layer.ffn_up_exps_s,
            layer.ffn_gate_exps_s,
            layer.ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    ggml_tensor * sh;
    if (ql.gate_shexp.trellis || ql.up_shexp.trellis || ql.down_shexp.trellis) {
        ggml_tensor * g = lin(layer.ffn_gate_shexp, ql.gate_shexp, cur);
        ggml_tensor * u = lin(layer.ffn_up_shexp,   ql.up_shexp,   cur);
        sh = lin(layer.ffn_down_shexp, ql.down_shexp, ggml_swiglu_split(ctx0, g, u));
    } else {
        sh = build_ffn(cur,
                layer.ffn_up_shexp,   nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    }
    ggml_tensor * sg = ggml_sigmoid(ctx0, build_lora_mm(layer.ffn_gate_inp_shexp, cur));
    sh = ggml_mul(ctx0, sh, sg);
    cb(sh, "ffn_shexp_gated", il);

    return ggml_add(ctx0, moe_out, sh);
}

ggml_tensor * llama_model_qwen4exp::graph::build_moe_d4(ggml_tensor * cur, int il) {
    const auto & layer = model.layers[il];
    const auto & ql    = model.q4x_layers[il];
    const int64_t nt   = cur->ne[1];

    ggml_tensor * logits = build_lora_mm(layer.ffn_gate_inp, cur);
    cb(logits, "ffn_moe_logits", il);
    ggml_tensor * probs = ggml_soft_max(ctx0, logits);
    cb(probs, "ffn_moe_probs", il);

    ggml_tensor * selected = ggml_argsort_top_k(ctx0, probs, n_expert_used);
    cb(selected, "ffn_moe_topk", il);

    ggml_tensor * weights = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, probs, 1, n_expert, nt), selected);
    if (hparams.expert_weights_norm) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, nt);
        ggml_tensor * wsum = ggml_clamp(ctx0, ggml_sum_rows(ctx0, weights), 6.103515625e-5, INFINITY);
        weights = ggml_reshape_3d(ctx0, ggml_div(ctx0, weights, wsum), 1, n_expert_used, nt);
    }
    if (hparams.expert_weights_scale != 0.0f && hparams.expert_weights_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, hparams.expert_weights_scale);
    }
    cb(weights, "ffn_moe_weights", il);
    ggml_build_forward_expand(gf, weights);

    const int32_t * hash = model.m1_d4_hash.empty() ? nullptr : model.m1_d4_hash.data();
    auto d4 = [&](const m1_d4 & e, ggml_tensor * x) {
        return ggml_mach1_d4_mm(ctx0, e.trellis, e.offs, e.su, e.sv, e.gw, e.zt, e.units,
                                selected, ggml_is_contiguous(x) ? x : ggml_cont(ctx0, x), hash);
    };
    ggml_tensor * xexp = ggml_reshape_3d(ctx0, ggml_is_contiguous(cur) ? cur : ggml_cont(ctx0, cur), n_embd, 1, nt);
    ggml_tensor * gate = d4(ql.exps[0], xexp);
    ggml_tensor * up   = d4(ql.exps[1], xexp);
    cb(gate, "ffn_moe_gate", il);
    cb(up,   "ffn_moe_up",   il);
    ggml_tensor * experts = d4(ql.exps[2], ggml_swiglu_split(ctx0, gate, up));
    cb(experts, "ffn_moe_down", il);
    experts = ggml_mul(ctx0, experts, weights);
    ggml_build_forward_expand(gf, experts);

    ggml_tensor * views[LLAMA_MAX_EXPERTS] = {};
    for (int64_t i = 0; i < n_expert_used; ++i) {
        views[i] = ggml_view_2d(ctx0, experts, n_embd, nt, experts->nb[2], i*experts->nb[1]);
        ggml_build_forward_expand(gf, views[i]);
    }
    ggml_tensor * out = views[0];
    for (int64_t i = 1; i < n_expert_used; ++i) {
        out = ggml_add(ctx0, out, views[i]);
        ggml_build_forward_expand(gf, out);
    }
    return n_expert_used == 1 ? ggml_cont(ctx0, out) : out;
}
