#include "llama_decoder_adapter.h"
#include "cpu/cpu_ops.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

using namespace relic::cpu;

LlamaDecoderAdapter::LlamaDecoderAdapter(OpenClBackend *backend) : cl(backend) {}

bool LlamaDecoderAdapter::init(const ArchitectureSpec &spec, int64_t max_seq_len, const ExecutionPlan *plan)
{
    arch = spec;
    seq_limit = max_seq_len;
    plan_ = plan;

    int64_t n_embd = arch.n_embd;
    int64_t n_head = arch.n_head;
    int64_t n_kv_head = arch.n_head_kv;
    int64_t head_dim = n_embd / n_head;
    int64_t n_ff = arch.n_ff;
    int64_t q_size = n_head * head_dim;
    int64_t kv_size = n_kv_head * head_dim;

    int64_t scratch_size = n_embd * 3 + q_size + kv_size * 2 + n_ff * 2 + n_head * max_seq_len + n_embd;
    act.resize((size_t)scratch_size, 0.0f);
    weights.resize((size_t)std::max(n_embd * (int64_t)4096, n_embd * n_ff * 4), 0.0f);
    kv_cache.assign((size_t)(arch.n_layer * 2 * max_seq_len * n_embd), 0.0f);

    return true;
}

void LlamaDecoderAdapter::reset()
{
    std::fill(kv_cache.begin(), kv_cache.end(), 0.0f);
    std::fill(act.begin(), act.end(), 0.0f);
}

void LlamaDecoderAdapter::save_state_checkpoint()
{
    snapshot_kv_cache_ = kv_cache;
    snapshot_act_ = act;
}

void LlamaDecoderAdapter::restore_state_checkpoint()
{
    kv_cache = snapshot_kv_cache_;
    act = snapshot_act_;
}

int LlamaDecoderAdapter::forward(const LlamaModel &model, int token_id, int64_t position, float *logits, bool compute_output)
{
    if (position >= seq_limit)
    {
        fprintf(stderr, "Maximum sequence length exceeded\n");
        return -1;
    }

    int64_t n_embd = arch.n_embd;
    int64_t n_vocab = arch.n_vocab;
    int64_t n_head = arch.n_head;
    int64_t n_kv_head = arch.n_head_kv;
    int64_t head_dim = n_embd / n_head;
    int64_t n_ff = arch.n_ff;
    int64_t q_size = n_head * head_dim;
    int64_t kv_size = n_kv_head * head_dim;

    float *hidden = act.data();
    float *residual = hidden + n_embd;
    float *q_buf = residual + n_embd;
    float *k_buf = q_buf + q_size;
    float *v_buf = k_buf + kv_size;
    float *gate_buf = v_buf + kv_size;
    float *up_buf = gate_buf + n_ff;
    float *scores = up_buf + n_ff;
    float *attn_out = scores + n_head * seq_limit;

    // Embedding lookup
    auto emb_it = model.tensors.find("token_embd.weight");
    if (emb_it == model.tensors.end())
        emb_it = model.tensors.find("tok_embeddings.weight");
    if (emb_it == model.tensors.end() || emb_it->second.data.empty())
    {
        fprintf(stderr, "No token embedding tensor found\n");
        return -1;
    }

    const auto &embed = emb_it->second;
    if (embed.type == GgmlType::F32)
    {
        const float *emb_data = (const float *)embed.data.data();
        memcpy(hidden, emb_data + token_id * n_embd, (size_t)(n_embd * sizeof(float)));
    }
    else
    {
        model.dequantize_rows_to_f32(embed, token_id, 1, hidden);
    }

    auto dequant_run = [&](const char *name, float *buf)
    {
        auto it = model.tensors.find(name);
        if (it != model.tensors.end())
            model.dequantize_to_f32(it->second, buf);
    };

    for (int64_t layer = 0; layer < arch.n_layer; layer++)
    {
        std::string prefix = "blk." + std::to_string(layer) + ".";
        float *k_slice = kv_cache.data() + layer * 2 * seq_limit * n_embd;
        float *v_slice = k_slice + seq_limit * n_embd;

        memcpy(residual, hidden, n_embd * sizeof(float));

        dequant_run((prefix + "attn_norm.weight").c_str(), weights.data());
        rms_norm_cpu(hidden, hidden, weights.data(), n_embd, 1, arch.norm_eps);

        auto w_Q = model.tensors.find(prefix + "attn_q.weight");
        auto w_K = model.tensors.find(prefix + "attn_k.weight");
        auto w_V = model.tensors.find(prefix + "attn_v.weight");
        auto w_O = model.tensors.find(prefix + "attn_output.weight");

        if (w_Q != model.tensors.end())
        {
            model.dequantize_to_f32(w_Q->second, weights.data());
            matmul_nt_cpu(q_buf, hidden, weights.data(), 1, q_size, n_embd);
        }
        if (w_K != model.tensors.end())
        {
            model.dequantize_to_f32(w_K->second, weights.data());
            matmul_nt_cpu(k_buf, hidden, weights.data(), 1, kv_size, n_embd);
        }
        if (w_V != model.tensors.end())
        {
            model.dequantize_to_f32(w_V->second, weights.data());
            matmul_nt_cpu(v_buf, hidden, weights.data(), 1, kv_size, n_embd);
        }

        rope_cpu(q_buf, q_size, n_head, (int)position, 1, arch.rope_freq_base);
        rope_cpu(k_buf, kv_size, n_kv_head, (int)position, 1, arch.rope_freq_base);

        memcpy(k_slice + position * n_embd, k_buf, kv_size * sizeof(float));
        memcpy(v_slice + position * n_embd, v_buf, kv_size * sizeof(float));

        int64_t S = position + 1;
        float inv_scale = 1.0f / sqrtf((float)head_dim);
        int64_t q_per_kv = n_head / n_kv_head;

        for (int64_t h = 0; h < n_head; h++)
        {
            int64_t h_kv = h / q_per_kv;
            for (int64_t s = 0; s < S; s++)
            {
                float sum = 0.0f;
                for (int64_t d = 0; d < head_dim; d++)
                {
                    sum += q_buf[h * head_dim + d] * k_slice[s * n_embd + h_kv * head_dim + d];
                }
                scores[h * S + s] = sum * inv_scale;
            }
        }

        for (int64_t h = 0; h < n_head; h++)
        {
            int64_t offset = h * S;
            float maxv = scores[offset];
            for (int64_t s = 0; s < S; s++)
                if (scores[offset + s] > maxv)
                    maxv = scores[offset + s];
            float sum = 0.0f;
            for (int64_t s = 0; s < S; s++)
            {
                scores[offset + s] = expf(scores[offset + s] - maxv);
                sum += scores[offset + s];
            }
            float inv_sum = 1.0f / sum;
            for (int64_t s = 0; s < S; s++)
                scores[offset + s] *= inv_sum;
        }

        memset(attn_out, 0, n_embd * sizeof(float));
        for (int64_t h = 0; h < n_head; h++)
        {
            int64_t h_kv = h / q_per_kv;
            for (int64_t s = 0; s < S; s++)
            {
                float w = scores[h * S + s];
                for (int64_t d = 0; d < head_dim; d++)
                {
                    attn_out[h * head_dim + d] += w * v_slice[s * n_embd + h_kv * head_dim + d];
                }
            }
        }

        if (w_O != model.tensors.end())
        {
            model.dequantize_to_f32(w_O->second, weights.data());
            matmul_nt_cpu(gate_buf, attn_out, weights.data(), 1, n_embd, n_embd);
            memcpy(attn_out, gate_buf, n_embd * sizeof(float));
        }

        add_cpu(hidden, residual, attn_out, n_embd);

        // FFN
        memcpy(residual, hidden, n_embd * sizeof(float));
        dequant_run((prefix + "ffn_norm.weight").c_str(), weights.data());
        rms_norm_cpu(hidden, hidden, weights.data(), n_embd, 1, arch.norm_eps);

        auto w_gate = model.tensors.find(prefix + "ffn_gate.weight");
        auto w_up = model.tensors.find(prefix + "ffn_up.weight");
        auto w_down = model.tensors.find(prefix + "ffn_down.weight");

        if (w_gate != model.tensors.end() && w_up != model.tensors.end())
        {
            model.dequantize_to_f32(w_gate->second, weights.data());
            matmul_nt_cpu(gate_buf, hidden, weights.data(), 1, n_ff, n_embd);

            model.dequantize_to_f32(w_up->second, weights.data());
            matmul_nt_cpu(up_buf, hidden, weights.data(), 1, n_ff, n_embd);

            silu_cpu(gate_buf, gate_buf, n_ff);
            for (int64_t i = 0; i < n_ff; i++)
                gate_buf[i] *= up_buf[i];

            if (w_down != model.tensors.end())
            {
                model.dequantize_to_f32(w_down->second, weights.data());
                matmul_nt_cpu(hidden, gate_buf, weights.data(), 1, n_embd, n_ff);
            }

            add_cpu(hidden, residual, hidden, n_embd);
        }
    }

    // Final RMSNorm
    auto norm_w = model.tensors.find("output_norm.weight");
    if (norm_w == model.tensors.end())
        norm_w = model.tensors.find("norm.weight");
    if (norm_w != model.tensors.end())
    {
        model.dequantize_to_f32(norm_w->second, weights.data());
        rms_norm_cpu(hidden, hidden, weights.data(), n_embd, 1, arch.norm_eps);
    }

    // Output logits projection
    if (compute_output && logits != nullptr)
    {
        auto out_w = model.tensors.find("output.weight");
        if (out_w == model.tensors.end())
            out_w = model.tensors.find("token_embd.weight");
        if (out_w != model.tensors.end())
        {
            const int64_t chunk_v = 4096;
            for (int64_t v_start = 0; v_start < n_vocab; v_start += chunk_v)
            {
                int64_t cur_v = std::min(chunk_v, n_vocab - v_start);
                model.dequantize_rows_to_f32(out_w->second, v_start, cur_v, weights.data());
                matmul_nt_cpu(logits + v_start, hidden, weights.data(), 1, cur_v, n_embd);
            }
        }
    }

    return 0;
}
