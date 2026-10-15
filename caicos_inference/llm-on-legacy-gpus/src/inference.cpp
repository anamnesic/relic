#include "inference.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <random>

bool InferenceEngine::init(LlamaModel *m, Tokenizer *tok, OpenClBackend *backend,
                            int64_t max_seq_len) {
    model = m;
    tokenizer = tok;
    cl = backend;
    state.max_seq_len = max_seq_len;

    int64_t n_embd = model->n_embd;
    int64_t n_head = model->n_head;
    int64_t n_kv_head = model->n_head_kv;
    int64_t head_dim = model->n_embd_head_k;
    int64_t n_ff = model->n_ff;
    int64_t q_size = n_head * head_dim;
    int64_t kv_size = n_kv_head * head_dim;

    // Dedicated scratch arena: no buffer reuse
    // hidden[n_embd] + residual[n_embd] + q_buf[q_size] + k_buf[kv_size]
    // + v_buf[kv_size] + gate_buf[n_ff] + up_buf[n_ff]
    // + scores[n_head * max_seq_len] + attn_out[n_embd]
    int64_t scratch_size = n_embd * 3 + q_size + kv_size * 2
                         + n_ff * 2 + n_head * max_seq_len + n_embd;
    act.resize((size_t)scratch_size);

    // Weight buffer: largest tensor in the model (output.weight can be huge)
    int64_t max_tensor_ne = 0;
    for (auto &[name, t] : m->tensors) {
        max_tensor_ne = std::max(max_tensor_ne, t.nelements());
    }
    weights.resize((size_t)max_tensor_ne);

    // Largest layer weight (for GPU upload, typically Q/K/V/O/gate/up/down = ~7 * n_embd^2 for SmolLM2)
    int64_t max_layer_weight = std::max(n_embd * n_embd * 4,
                                        n_embd * n_ff * 4 * 3);

    // KV cache: [n_layer, 2, max_seq_len, n_embd]
    state.kv_cache.resize((size_t)(model->n_layer * 2 * max_seq_len * n_embd), 0.0f);

    if (cl && cl->initialized) {
        state.use_gpu = true;
        state.n_embd_state = n_embd;
        state.gpu_act.alloc(cl->dev.context, (size_t)scratch_size);
        state.gpu_weights.alloc(cl->dev.context, (size_t)max_layer_weight);
        state.gpu_k.alloc(cl->dev.context, (size_t)(max_seq_len * n_embd * 4));
        state.gpu_v.alloc(cl->dev.context, (size_t)(max_seq_len * n_embd * 4));
        state.gpu_score.alloc(cl->dev.context, (size_t)(max_seq_len * 4));
        state.gpu_ctx.alloc(cl->dev.context, (size_t)(n_embd * 4));
    }

    return true;
}

void InferenceEngine::free_buffers() {
    act.clear();
    weights.clear();
    state.kv_cache.clear();
}

//------------------------------------------------------------------------------
// CPU fallback operations (used when GPU is not available or for streaming)
//------------------------------------------------------------------------------

void InferenceEngine::rms_norm_cpu(float *out, const float *x, const float *weight,
                                    int64_t n, int64_t rows) {
    for (int64_t r = 0; r < rows; r++) {
        float ss = 0.0f;
        for (int64_t i = 0; i < n; i++) ss += x[r * n + i] * x[r * n + i];
        float s = 1.0f / sqrtf(ss / (float)n + 1e-5f);
        for (int64_t i = 0; i < n; i++) out[r * n + i] = x[r * n + i] * s * weight[i];
    }
}

void InferenceEngine::silu_cpu(float *out, const float *x, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        out[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

void InferenceEngine::matmul_cpu(float *dst, const float *a, const float *b,
                                  int64_t M, int64_t N, int64_t K) {
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            dst[i * N + j] = sum;
        }
    }
}

void InferenceEngine::matmul_nt_cpu(float *dst, const float *a, const float *b,
                                     int64_t M, int64_t N, int64_t K) {
    // b is stored as [N, K] (transposed in memory)
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                sum += a[i * K + k] * b[j * K + k];
            }
            dst[i * N + j] = sum;
        }
    }
}

void InferenceEngine::rope_cpu(float *x, int64_t n_embd, int64_t n_head,
                                int64_t pos, int64_t n_tokens) {
    int64_t head_dim = n_embd / n_head;
    float base = model ? model->rope_freq_base : 10000.0f;
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t h = 0; h < n_head; h++) {
            for (int64_t hh = 0; hh < head_dim / 2; hh++) {
                float theta = (float)pos * powf(base, -2.0f * (float)hh / (float)head_dim);
                float cos_t = cosf(theta);
                float sin_t = sinf(theta);
                float *row = x + t * n_embd + h * head_dim;
                float v0 = row[hh];
                float v1 = row[hh + head_dim / 2];
                row[hh] = v0 * cos_t - v1 * sin_t;
                row[hh + head_dim / 2] = v0 * sin_t + v1 * cos_t;
            }
        }
    }
}

void InferenceEngine::softmax_cpu(float *x, int64_t n, int64_t rows) {
    for (int64_t r = 0; r < rows; r++) {
        float *row = x + r * n;
        float maxv = row[0];
        for (int64_t i = 1; i < n; i++) if (row[i] > maxv) maxv = row[i];
        float sum = 0.0f;
        for (int64_t i = 0; i < n; i++) { row[i] = expf(row[i] - maxv); sum += row[i]; }
        float inv = 1.0f / sum;
        for (int64_t i = 0; i < n; i++) row[i] *= inv;
    }
}

void InferenceEngine::add_cpu(float *dst, const float *a, const float *b, int64_t n) {
    for (int64_t i = 0; i < n; i++) dst[i] = a[i] + b[i];
}

//------------------------------------------------------------------------------
// Layer forward (streaming - one layer at a time)
//------------------------------------------------------------------------------

void InferenceEngine::forward_layer(int layer, int token_id, float *hidden,
                                     float *k_slice, float *v_slice,
                                     float *scores, float *attn_out,
                                     float *residual,
                                     float *q_buf, float *k_buf, float *v_buf,
                                     float *gate_buf, float *up_buf) {
    int64_t n_embd = model->n_embd;
    int64_t n_head = model->n_head;
    int64_t n_kv_head = model->n_head_kv;
    int64_t head_dim = model->n_embd_head_k;
    int64_t n_past = state.n_past;
    int64_t n_ff = model->n_ff;
    bool use_gpu = state.use_gpu && cl && cl->initialized;
    (void)token_id; // used for MoE routing in the future

    auto &t = model->tensors;
    auto dequant_run = [&](const char *name, float *buf) {
        auto it = t.find(name);
        if (it != t.end()) {
            model->dequantize_to_f32(it->second, buf);
        }
    };

    //--------------------------------------------------------------------------
    // RMS Norm
    //--------------------------------------------------------------------------
    auto tensor_name = [&](const std::string &base) -> std::string {
        return "blk." + std::to_string(layer) + "." + base + ".weight";
    };

    auto get_weight = [&](const std::string &base) -> const LlamaModel::Tensor * {
        std::string name = tensor_name(base);
        auto it = t.find(name);
        return it != t.end() ? &it->second : nullptr;
    };

    // Save residual before attention norm (original hidden state)
    memcpy(residual, hidden, n_embd * sizeof(float));

    dequant_run(tensor_name("attn_norm").c_str(), weights.data());
    rms_norm_cpu(hidden, hidden, weights.data(), n_embd, 1);

    //--------------------------------------------------------------------------
    // QKV Projection
    //--------------------------------------------------------------------------
    auto w_Q = get_weight("attn_q");
    auto w_K = get_weight("attn_k");
    auto w_V = get_weight("attn_v");
    auto w_O = get_weight("attn_output");

    int64_t q_size = n_head * head_dim;
    int64_t kv_size = n_kv_head * head_dim;

    float *q = q_buf;
    float *k = k_buf;
    float *v = v_buf;

    if (w_Q) {
        dequant_run(w_Q->name.c_str(), weights.data());
        matmul_nt_cpu(q, hidden, weights.data(), 1, q_size, n_embd);
    }
    if (w_K) {
        dequant_run(w_K->name.c_str(), weights.data());
        matmul_nt_cpu(k, hidden, weights.data(), 1, kv_size, n_embd);
    }
    if (w_V) {
        dequant_run(w_V->name.c_str(), weights.data());
        matmul_nt_cpu(v, hidden, weights.data(), 1, kv_size, n_embd);
    }



    // RoPE
    rope_cpu(q, q_size, n_head, n_past, 1);
    rope_cpu(k, kv_size, n_kv_head, n_past, 1);

    // Store KV cache
    memcpy(k_slice + n_past * n_embd, k, kv_size * sizeof(float));
    memcpy(v_slice + n_past * n_embd, v, kv_size * sizeof(float));

    //--------------------------------------------------------------------------
    // Attention
    //--------------------------------------------------------------------------
    int64_t S = n_past + 1;

    // scores = q * K^T / sqrt(head_dim)  [n_head, S]
    float inv_scale = 1.0f / sqrtf((float)head_dim);
    int64_t q_per_kv = n_head / n_kv_head;
    for (int64_t h = 0; h < n_head; h++) {
        int64_t h_kv = h / q_per_kv;
        for (int64_t s = 0; s < S; s++) {
            float sum = 0.0f;
            for (int64_t d = 0; d < head_dim; d++) {
                sum += q[h * head_dim + d] * k_slice[s * n_embd + h_kv * head_dim + d];
            }
            scores[h * S + s] = sum * inv_scale;
        }
    }

    // causal mask + softmax
    for (int64_t h = 0; h < n_head; h++) {
        int64_t offset = h * S;
        float maxv = scores[offset];
        for (int64_t s = 0; s < S; s++) {
            if (s > n_past) scores[offset + s] = -INFINITY;
            else if (scores[offset + s] > maxv) maxv = scores[offset + s];
        }
        float sum = 0.0f;
        for (int64_t s = 0; s <= n_past; s++) {
            scores[offset + s] = expf(scores[offset + s] - maxv);
            sum += scores[offset + s];
        }
        float inv_sum = 1.0f / sum;
        for (int64_t s = 0; s < S; s++) scores[offset + s] *= inv_sum;
    }

    // attention output = scores * V
    memset(attn_out, 0, n_embd * sizeof(float));
    for (int64_t h = 0; h < n_head; h++) {
        int64_t h_kv = h / q_per_kv;
        for (int64_t s = 0; s <= n_past; s++) {
            float w = scores[h * S + s];
            for (int64_t d = 0; d < head_dim; d++) {
                attn_out[h * head_dim + d] += w * v_slice[s * n_embd + h_kv * head_dim + d];
            }
        }
    }

    // Output projection (use gate_buf as temp — copy to attn_out, gate overwrites later)
    if (w_O) {
        dequant_run(w_O->name.c_str(), weights.data());
        matmul_nt_cpu(gate_buf, attn_out, weights.data(), 1, n_embd, n_embd);
        memcpy(attn_out, gate_buf, n_embd * sizeof(float));
    }

    // Residual: hidden = residual (original) + attention output
    add_cpu(hidden, residual, attn_out, n_embd);

    //--------------------------------------------------------------------------
    // FFN
    //--------------------------------------------------------------------------
    // Save residual before FFN norm
    memcpy(residual, hidden, n_embd * sizeof(float));

    dequant_run(tensor_name("ffn_norm").c_str(), weights.data());
    rms_norm_cpu(hidden, hidden, weights.data(), n_embd, 1);

    auto w_ffn_gate = get_weight("ffn_gate");
    auto w_ffn_up = get_weight("ffn_up");
    auto w_ffn_down = get_weight("ffn_down");

    if (w_ffn_gate && w_ffn_up) {
        dequant_run(w_ffn_gate->name.c_str(), weights.data());
        matmul_nt_cpu(gate_buf, hidden, weights.data(), 1, n_ff, n_embd);

        dequant_run(w_ffn_up->name.c_str(), weights.data());
        matmul_nt_cpu(up_buf, hidden, weights.data(), 1, n_ff, n_embd);

        silu_cpu(gate_buf, gate_buf, n_ff);
        for (int64_t i = 0; i < n_ff; i++) gate_buf[i] *= up_buf[i];

        if (w_ffn_down) {
            dequant_run(w_ffn_down->name.c_str(), weights.data());
            matmul_nt_cpu(hidden, gate_buf, weights.data(), 1, n_embd, n_ff);
        }

        add_cpu(hidden, residual, hidden, n_embd);
    }
}

//------------------------------------------------------------------------------
// Forward pass for one token
//------------------------------------------------------------------------------

int InferenceEngine::forward(int token_id, float *logits) {
    int64_t n_embd = model->n_embd;
    int64_t n_vocab = model->n_vocab;
    int64_t n_layers = model->n_layer;
    float *hidden = act.data();
    int64_t S = state.n_past + 1;

    if (S > state.max_seq_len) {
        fprintf(stderr, "Maximum sequence length exceeded\n");
        return -1;
    }

    // Token embedding lookup
    auto &tok_embd = model->tensors["token_embd.weight"];
    if (tok_embd.data.empty()) {
        // Try alternate name
        auto alt = model->tensors.find("tok_embeddings.weight");
        if (alt == model->tensors.end()) {
            // Try blk.0. token embedding
            alt = model->tensors.find("blk.0.attn_norm.weight");
            if (alt == model->tensors.end()) {
                fprintf(stderr, "No token embedding found\n");
                return -1;
            }
        }
    }

    // Get embedding for this token
    auto emb_it = model->tensors.find("token_embd.weight");
    if (emb_it == model->tensors.end())
        emb_it = model->tensors.find("tok_embeddings.weight");
    if (emb_it == model->tensors.end())
        emb_it = model->tensors.find("gpt.embd.weight");

    if (emb_it == model->tensors.end() || emb_it->second.data.empty()) {
        fprintf(stderr, "No token embedding tensor found\n");
        return -1;
    }

    const LlamaModel::Tensor &embed = emb_it->second;
    int64_t emb_row_size = embed.dims[0];

    // Dequantize entire embedding row
    if (embed.type == GgmlType::F32) {
        const float *emb_data = (const float *)embed.data.data();
        memcpy(hidden, emb_data + token_id * emb_row_size, (size_t)(emb_row_size * 4));
    } else {
        model->dequantize_to_f32(embed, weights.data());
        memcpy(hidden, weights.data() + token_id * emb_row_size, (size_t)(emb_row_size * 4));
    }

    int64_t n_head_ = model->n_head;
    int64_t n_kv_head_ = model->n_head_kv;
    int64_t head_dim_ = model->n_embd_head_k;
    int64_t q_size_ = n_head_ * head_dim_;
    int64_t kv_size_ = n_kv_head_ * head_dim_;
    int64_t n_ff = model->n_ff;

    // Dedicated scratch buffers (no overlap, no reuse)
    float *residual = hidden + n_embd;
    float *q_buf = residual + n_embd;
    float *k_buf = q_buf + q_size_;
    float *v_buf = k_buf + kv_size_;
    float *gate_buf = v_buf + kv_size_;
    float *up_buf = gate_buf + n_ff;
    float *scores = up_buf + n_ff;
    float *attn_out = scores + n_head_ * state.max_seq_len;

    for (int64_t layer = 0; layer < n_layers; layer++) {
        float *k_slice = state.kv_cache.data() + layer * 2 * state.max_seq_len * n_embd;
        float *v_slice = k_slice + state.max_seq_len * n_embd;

        forward_layer((int)layer, token_id, hidden, k_slice, v_slice,
                      scores, attn_out, residual,
                      q_buf, k_buf, v_buf, gate_buf, up_buf);
    }

    // Final RMS Norm
    auto norm_w = model->tensors.find("output_norm.weight");
    if (norm_w == model->tensors.end())
        norm_w = model->tensors.find("norm.weight");
    if (norm_w != model->tensors.end()) {
        model->dequantize_to_f32(norm_w->second, weights.data());
        rms_norm_cpu(hidden, hidden, weights.data(), n_embd, 1);
    }

    // Output projection
    auto output_w = model->tensors.find("output.weight");
    if (output_w == model->tensors.end())
        output_w = model->tensors.find("token_embd.weight");

    if (output_w != model->tensors.end()) {
        model->dequantize_to_f32(output_w->second, weights.data());
        matmul_nt_cpu(logits, hidden, weights.data(), 1,
                      (int64_t)output_w->second.dims[1],
                      (int64_t)output_w->second.dims[0]);
    } else {
        if (embed.type == GgmlType::F32) {
            const float *emb_data = (const float *)embed.data.data();
            matmul_nt_cpu(logits, hidden, emb_data, 1, n_vocab, n_embd);
        } else {
            model->dequantize_to_f32(embed, weights.data());
            matmul_nt_cpu(logits, hidden, weights.data(), 1, n_vocab, n_embd);
        }
    }

    state.n_past++;
    return 0;
}

//------------------------------------------------------------------------------
// Text generation
//------------------------------------------------------------------------------

std::string InferenceEngine::generate(const std::string &prompt, int max_tokens,
                                        float temperature, int top_k) {
    if (!tokenizer || !model) return "";

    std::vector<int> input_tokens = tokenizer->encode(prompt);
    if (input_tokens.empty()) {
        fprintf(stderr, "Failed to tokenize prompt\n");
        return "";
    }
    // No BOS prepend for this model (SmolLM2 doesn't use it)

    std::vector<float> logits((size_t)model->n_vocab);
    std::string output;
    std::mt19937 rng(42);

    for (int tok : input_tokens) {
        forward(tok, logits.data());
    }

    // logits now predict the next token after the complete prompt
    int last_token = 0;
    for (int i = 0; i < max_tokens; i++) {
        // Sample from logits (first iteration uses prompt-result logits)
        if (temperature < 0.01f) {
            last_token = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
        } else {
            for (auto &l : logits) l /= temperature;
            if (top_k > 0 && top_k < (int)logits.size()) {
                std::vector<std::pair<float, int>> scored;
                for (int j = 0; j < (int)logits.size(); j++)
                    scored.emplace_back(logits[j], j);
                std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                                  [](auto &a, auto &b) { return a.first > b.first; });
                float threshold = scored[top_k - 1].first;
                for (int j = 0; j < (int)logits.size(); j++)
                    if (logits[j] < threshold) logits[j] = -INFINITY;
            }
            float maxv = logits[0];
            for (auto &l : logits) if (l > maxv) maxv = l;
            float sum = 0.0f;
            for (auto &l : logits) { l = expf(l - maxv); sum += l; }
            for (auto &l : logits) l /= sum;
            float r = (float)rng() / (float)rng.max();
            float cum = 0.0f;
            last_token = (int)logits.size() - 1;
            for (int j = 0; j < (int)logits.size(); j++) {
                cum += logits[j];
                if (r < cum) { last_token = j; break; }
            }
        }

        if (last_token == tokenizer->eos_id) break;
        std::string piece = tokenizer->decode({last_token});
        output += piece;
        fprintf(stdout, "%s", piece.c_str());
        fflush(stdout);

        // Feed the sampled token to get logits for the next step
        forward(last_token, logits.data());
    }

    fprintf(stdout, "\n");
    return output;
}

#undef n_head
