#include "decoder.h"
#include "opencl_backend.h"
#include "qwen35_state.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

//------------------------------------------------------------------------------
// CPU reference operations
//------------------------------------------------------------------------------

void rms_norm_cpu(float *out, const float *x, const float *weight, int64_t n, int64_t rows, float eps = 1e-5f) {
    for (int64_t r = 0; r < rows; r++) {
        float ss = 0.0f;
        for (int64_t i = 0; i < n; i++) ss += x[r * n + i] * x[r * n + i];
        float s = 1.0f / sqrtf(ss / (float)n + eps);
        for (int64_t i = 0; i < n; i++) out[r * n + i] = x[r * n + i] * s * weight[i];
    }
}

void silu_cpu(float *out, const float *x, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        out[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

void matmul_nt_cpu(float *dst, const float *a, const float *b, int64_t M, int64_t N, int64_t K) {
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

void rope_cpu(float *x, int64_t n_embd, int64_t n_head, int64_t pos, int64_t n_tokens, float base = 10000.0f) {
    int64_t head_dim = n_embd / n_head;
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

void add_cpu(float *dst, const float *a, const float *b, int64_t n) {
    for (int64_t i = 0; i < n; i++) dst[i] = a[i] + b[i];
}

//------------------------------------------------------------------------------
// Hexagonal Adapter: LlamaDecoderAdapter
//------------------------------------------------------------------------------

class LlamaDecoderAdapter final : public ArchitectureDecoder {
public:
    explicit LlamaDecoderAdapter(OpenClBackend *backend) : cl(backend) {}

    bool init(const ArchitectureSpec &spec, int64_t max_seq_len) override {
        arch = spec;
        seq_limit = max_seq_len;

        int64_t n_embd = arch.n_embd;
        int64_t n_head = arch.n_head;
        int64_t n_kv_head = arch.n_head_kv;
        int64_t head_dim = n_embd / n_head;
        int64_t n_ff = arch.n_ff;
        int64_t q_size = n_head * head_dim;
        int64_t kv_size = n_kv_head * head_dim;

        int64_t scratch_size = n_embd * 3 + q_size + kv_size * 2
                             + n_ff * 2 + n_head * max_seq_len + n_embd;
        act.resize((size_t)scratch_size, 0.0f);
        weights.resize((size_t)std::max(n_embd * n_vocab_estimate, n_embd * n_ff * 4), 0.0f);
        kv_cache.assign((size_t)(arch.n_layer * 2 * max_seq_len * n_embd), 0.0f);

        return true;
    }

    void reset() override {
        std::fill(kv_cache.begin(), kv_cache.end(), 0.0f);
        std::fill(act.begin(), act.end(), 0.0f);
    }

    int forward(const LlamaModel &model, int token_id, int64_t position, float *logits) override {
        if (position >= seq_limit) {
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

        if (weights.size() < (size_t)(n_embd * std::max(n_vocab, n_ff))) {
            weights.resize((size_t)(n_embd * std::max(n_vocab, n_ff * 4)));
        }

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
        if (emb_it == model.tensors.end()) emb_it = model.tensors.find("tok_embeddings.weight");
        if (emb_it == model.tensors.end() || emb_it->second.data.empty()) {
            fprintf(stderr, "No token embedding tensor found\n");
            return -1;
        }

        const auto &embed = emb_it->second;
        int64_t emb_row_size = embed.dims[0];
        if (embed.type == GgmlType::F32) {
            const float *emb_data = (const float *)embed.data.data();
            memcpy(hidden, emb_data + token_id * emb_row_size, (size_t)(emb_row_size * sizeof(float)));
        } else {
            model.dequantize_to_f32(embed, weights.data());
            memcpy(hidden, weights.data() + token_id * emb_row_size, (size_t)(emb_row_size * sizeof(float)));
        }

        auto dequant_run = [&](const char *name, float *buf) {
            auto it = model.tensors.find(name);
            if (it != model.tensors.end()) model.dequantize_to_f32(it->second, buf);
        };

        for (int64_t layer = 0; layer < arch.n_layer; layer++) {
            std::string prefix = "blk." + std::to_string(layer) + ".";
            float *k_slice = kv_cache.data() + layer * 2 * seq_limit * n_embd;
            float *v_slice = k_slice + seq_limit * n_embd;

            memcpy(residual, hidden, n_embd * sizeof(float));

            // Attention RMSNorm
            dequant_run((prefix + "attn_norm.weight").c_str(), weights.data());
            rms_norm_cpu(hidden, hidden, weights.data(), n_embd, 1, arch.norm_eps);

            // QKV projection
            auto w_Q = model.tensors.find(prefix + "attn_q.weight");
            auto w_K = model.tensors.find(prefix + "attn_k.weight");
            auto w_V = model.tensors.find(prefix + "attn_v.weight");
            auto w_O = model.tensors.find(prefix + "attn_output.weight");

            if (w_Q != model.tensors.end()) {
                model.dequantize_to_f32(w_Q->second, weights.data());
                matmul_nt_cpu(q_buf, hidden, weights.data(), 1, q_size, n_embd);
            }
            if (w_K != model.tensors.end()) {
                model.dequantize_to_f32(w_K->second, weights.data());
                matmul_nt_cpu(k_buf, hidden, weights.data(), 1, kv_size, n_embd);
            }
            if (w_V != model.tensors.end()) {
                model.dequantize_to_f32(w_V->second, weights.data());
                matmul_nt_cpu(v_buf, hidden, weights.data(), 1, kv_size, n_embd);
            }

            // RoPE
            rope_cpu(q_buf, q_size, n_head, position, 1, arch.rope_freq_base);
            rope_cpu(k_buf, kv_size, n_kv_head, position, 1, arch.rope_freq_base);

            // Store KV
            memcpy(k_slice + position * n_embd, k_buf, kv_size * sizeof(float));
            memcpy(v_slice + position * n_embd, v_buf, kv_size * sizeof(float));

            // Causal Attention
            int64_t S = position + 1;
            float inv_scale = 1.0f / sqrtf((float)head_dim);
            int64_t q_per_kv = n_head / n_kv_head;

            for (int64_t h = 0; h < n_head; h++) {
                int64_t h_kv = h / q_per_kv;
                for (int64_t s = 0; s < S; s++) {
                    float sum = 0.0f;
                    for (int64_t d = 0; d < head_dim; d++) {
                        sum += q_buf[h * head_dim + d] * k_slice[s * n_embd + h_kv * head_dim + d];
                    }
                    scores[h * S + s] = sum * inv_scale;
                }
            }

            // Softmax
            for (int64_t h = 0; h < n_head; h++) {
                int64_t offset = h * S;
                float maxv = scores[offset];
                for (int64_t s = 0; s < S; s++) if (scores[offset + s] > maxv) maxv = scores[offset + s];
                float sum = 0.0f;
                for (int64_t s = 0; s < S; s++) {
                    scores[offset + s] = expf(scores[offset + s] - maxv);
                    sum += scores[offset + s];
                }
                float inv_sum = 1.0f / sum;
                for (int64_t s = 0; s < S; s++) scores[offset + s] *= inv_sum;
            }

            // Attention output = scores * V
            memset(attn_out, 0, n_embd * sizeof(float));
            for (int64_t h = 0; h < n_head; h++) {
                int64_t h_kv = h / q_per_kv;
                for (int64_t s = 0; s < S; s++) {
                    float w = scores[h * S + s];
                    for (int64_t d = 0; d < head_dim; d++) {
                        attn_out[h * head_dim + d] += w * v_slice[s * n_embd + h_kv * head_dim + d];
                    }
                }
            }

            if (w_O != model.tensors.end()) {
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

            if (w_gate != model.tensors.end() && w_up != model.tensors.end()) {
                model.dequantize_to_f32(w_gate->second, weights.data());
                matmul_nt_cpu(gate_buf, hidden, weights.data(), 1, n_ff, n_embd);

                model.dequantize_to_f32(w_up->second, weights.data());
                matmul_nt_cpu(up_buf, hidden, weights.data(), 1, n_ff, n_embd);

                silu_cpu(gate_buf, gate_buf, n_ff);
                for (int64_t i = 0; i < n_ff; i++) gate_buf[i] *= up_buf[i];

                if (w_down != model.tensors.end()) {
                    model.dequantize_to_f32(w_down->second, weights.data());
                    matmul_nt_cpu(hidden, gate_buf, weights.data(), 1, n_embd, n_ff);
                }

                add_cpu(hidden, residual, hidden, n_embd);
            }
        }

        // Final RMSNorm
        auto norm_w = model.tensors.find("output_norm.weight");
        if (norm_w == model.tensors.end()) norm_w = model.tensors.find("norm.weight");
        if (norm_w != model.tensors.end()) {
            model.dequantize_to_f32(norm_w->second, weights.data());
            rms_norm_cpu(hidden, hidden, weights.data(), n_embd, 1, arch.norm_eps);
        }

        // Output logits projection
        auto out_w = model.tensors.find("output.weight");
        if (out_w == model.tensors.end()) out_w = model.tensors.find("token_embd.weight");
        if (out_w != model.tensors.end()) {
            model.dequantize_to_f32(out_w->second, weights.data());
            matmul_nt_cpu(logits, hidden, weights.data(), 1, n_vocab, n_embd);
        }

        return 0;
    }

private:
    ArchitectureSpec arch;
    OpenClBackend *cl = nullptr;
    int64_t seq_limit = 2048;
    int64_t n_vocab_estimate = 32000;
    std::vector<float> act;
    std::vector<float> weights;
    std::vector<float> kv_cache;
};

//------------------------------------------------------------------------------
// Hexagonal Adapter: Qwen35DecoderAdapter
//------------------------------------------------------------------------------

class Qwen35DecoderAdapter final : public ArchitectureDecoder {
public:
    bool init(const ArchitectureSpec &spec, int64_t max_seq_len) override {
        arch = spec;
        seq_limit = max_seq_len;

        if (!recurrent_state.init(spec)) {
            fprintf(stderr, "Failed to initialize Qwen3.5 recurrent state\n");
            return false;
        }

        int64_t n_embd = arch.n_embd;
        int64_t n_head = arch.n_head;
        int64_t n_kv_head = arch.n_head_kv;
        int64_t head_dim = n_embd / n_head;
        int64_t n_ff = arch.n_ff;
        int64_t q_size = n_head * head_dim;
        int64_t kv_size = n_kv_head * head_dim;

        int64_t linear_inner = arch.linear_inner_size;
        int64_t conv_channels = recurrent_state.channels();

        int64_t scratch_size = n_embd * 3 + q_size + kv_size * 2
                             + n_ff * 2 + n_head * max_seq_len + n_embd
                             + conv_channels * 2 + linear_inner * 2;
        act.resize((size_t)scratch_size, 0.0f);
        weights.resize((size_t)std::max(n_embd * arch.n_vocab, n_embd * n_ff * 4), 0.0f);
        full_attn_kv.assign((size_t)(arch.n_layer * 2 * max_seq_len * n_embd), 0.0f);

        return true;
    }

    void reset() override {
        recurrent_state.reset();
        std::fill(full_attn_kv.begin(), full_attn_kv.end(), 0.0f);
        std::fill(act.begin(), act.end(), 0.0f);
    }

    int forward(const LlamaModel &model, int token_id, int64_t position, float *logits) override {
        if (position >= seq_limit) {
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

        int64_t conv_channels = recurrent_state.channels();
        int64_t linear_inner = arch.linear_inner_size;
        int64_t key_dim = arch.linear_key_head_dim;
        int64_t key_heads = arch.linear_key_heads;
        int64_t value_heads = arch.linear_value_heads;
        int64_t qk_dim = key_heads * key_dim;

        if (weights.size() < (size_t)(n_embd * std::max(n_vocab, n_ff))) {
            weights.resize((size_t)(n_embd * std::max(n_vocab, n_ff * 4)));
        }

        float *hidden = act.data();
        float *residual = hidden + n_embd;
        float *q_buf = residual + n_embd;
        float *k_buf = q_buf + q_size;
        float *v_buf = k_buf + kv_size;
        float *gate_buf = v_buf + kv_size;
        float *up_buf = gate_buf + n_ff;
        float *scores = up_buf + n_ff;
        float *attn_out = scores + n_head * seq_limit;
        float *conv_in = attn_out + n_embd;
        float *conv_out = conv_in + conv_channels;
        float *delta_out = conv_out + conv_channels;

        // Embedding lookup
        auto emb_it = model.tensors.find("token_embd.weight");
        if (emb_it == model.tensors.end()) emb_it = model.tensors.find("tok_embeddings.weight");
        if (emb_it == model.tensors.end() || emb_it->second.data.empty()) {
            fprintf(stderr, "No token embedding tensor found\n");
            return -1;
        }

        const auto &embed = emb_it->second;
        int64_t emb_row_size = embed.dims[0];
        if (embed.type == GgmlType::F32) {
            const float *emb_data = (const float *)embed.data.data();
            memcpy(hidden, emb_data + token_id * emb_row_size, (size_t)(emb_row_size * sizeof(float)));
        } else {
            model.dequantize_to_f32(embed, weights.data());
            memcpy(hidden, weights.data() + token_id * emb_row_size, (size_t)(emb_row_size * sizeof(float)));
        }

        auto dequant_run = [&](const char *name, float *buf) {
            auto it = model.tensors.find(name);
            if (it != model.tensors.end()) model.dequantize_to_f32(it->second, buf);
        };

        for (int64_t layer = 0; layer < arch.n_layer; layer++) {
            std::string prefix = "blk." + std::to_string(layer) + ".";
            bool is_full_attn = (arch.full_attention_interval > 0) &&
                                ((layer + 1) % arch.full_attention_interval == 0);

            memcpy(residual, hidden, n_embd * sizeof(float));

            // Layer Norm
            dequant_run((prefix + "attn_norm.weight").c_str(), weights.data());
            rms_norm_cpu(hidden, hidden, weights.data(), n_embd, 1, arch.norm_eps);

            if (is_full_attn) {
                // Full Attention Layer
                float *k_slice = full_attn_kv.data() + layer * 2 * seq_limit * n_embd;
                float *v_slice = k_slice + seq_limit * n_embd;

                auto w_Q = model.tensors.find(prefix + "attn_q.weight");
                auto w_K = model.tensors.find(prefix + "attn_k.weight");
                auto w_V = model.tensors.find(prefix + "attn_v.weight");
                auto w_O = model.tensors.find(prefix + "attn_output.weight");

                if (w_Q != model.tensors.end()) {
                    model.dequantize_to_f32(w_Q->second, weights.data());
                    matmul_nt_cpu(q_buf, hidden, weights.data(), 1, q_size, n_embd);
                }
                if (w_K != model.tensors.end()) {
                    model.dequantize_to_f32(w_K->second, weights.data());
                    matmul_nt_cpu(k_buf, hidden, weights.data(), 1, kv_size, n_embd);
                }
                if (w_V != model.tensors.end()) {
                    model.dequantize_to_f32(w_V->second, weights.data());
                    matmul_nt_cpu(v_buf, hidden, weights.data(), 1, kv_size, n_embd);
                }

                rope_cpu(q_buf, q_size, n_head, position, 1, arch.rope_freq_base);
                rope_cpu(k_buf, kv_size, n_kv_head, position, 1, arch.rope_freq_base);

                memcpy(k_slice + position * n_embd, k_buf, kv_size * sizeof(float));
                memcpy(v_slice + position * n_embd, v_buf, kv_size * sizeof(float));

                int64_t S = position + 1;
                float inv_scale = 1.0f / sqrtf((float)head_dim);
                int64_t q_per_kv = n_head / n_kv_head;

                for (int64_t h = 0; h < n_head; h++) {
                    int64_t h_kv = h / q_per_kv;
                    for (int64_t s = 0; s < S; s++) {
                        float sum = 0.0f;
                        for (int64_t d = 0; d < head_dim; d++) {
                            sum += q_buf[h * head_dim + d] * k_slice[s * n_embd + h_kv * head_dim + d];
                        }
                        scores[h * S + s] = sum * inv_scale;
                    }
                }

                for (int64_t h = 0; h < n_head; h++) {
                    int64_t offset = h * S;
                    float maxv = scores[offset];
                    for (int64_t s = 0; s < S; s++) if (scores[offset + s] > maxv) maxv = scores[offset + s];
                    float sum = 0.0f;
                    for (int64_t s = 0; s < S; s++) {
                        scores[offset + s] = expf(scores[offset + s] - maxv);
                        sum += scores[offset + s];
                    }
                    float inv_sum = 1.0f / sum;
                    for (int64_t s = 0; s < S; s++) scores[offset + s] *= inv_sum;
                }

                memset(attn_out, 0, n_embd * sizeof(float));
                for (int64_t h = 0; h < n_head; h++) {
                    int64_t h_kv = h / q_per_kv;
                    for (int64_t s = 0; s < S; s++) {
                        float w = scores[h * S + s];
                        for (int64_t d = 0; d < head_dim; d++) {
                            attn_out[h * head_dim + d] += w * v_slice[s * n_embd + h_kv * head_dim + d];
                        }
                    }
                }

                if (w_O != model.tensors.end()) {
                    model.dequantize_to_f32(w_O->second, weights.data());
                    matmul_nt_cpu(gate_buf, attn_out, weights.data(), 1, n_embd, n_embd);
                    memcpy(attn_out, gate_buf, n_embd * sizeof(float));
                }

                add_cpu(hidden, residual, attn_out, n_embd);
            } else {
                // Recurrent (Gated DeltaNet) Layer
                auto w_Q = model.tensors.find(prefix + "attn_q.weight");
                auto w_K = model.tensors.find(prefix + "attn_k.weight");
                auto w_V = model.tensors.find(prefix + "attn_v.weight");
                auto w_conv = model.tensors.find(prefix + "ssm_conv1d.weight");
                auto w_alpha = model.tensors.find(prefix + "ssm_alpha.weight");
                auto w_beta = model.tensors.find(prefix + "ssm_beta.weight");
                auto w_O = model.tensors.find(prefix + "attn_output.weight");

                // Q projection (size qk_dim)
                if (w_Q != model.tensors.end()) {
                    model.dequantize_to_f32(w_Q->second, weights.data());
                    matmul_nt_cpu(conv_in, hidden, weights.data(), 1, qk_dim, n_embd);
                }
                // K projection (size qk_dim)
                if (w_K != model.tensors.end()) {
                    model.dequantize_to_f32(w_K->second, weights.data());
                    matmul_nt_cpu(conv_in + qk_dim, hidden, weights.data(), 1, qk_dim, n_embd);
                }
                // V projection (size linear_inner)
                if (w_V != model.tensors.end()) {
                    model.dequantize_to_f32(w_V->second, weights.data());
                    matmul_nt_cpu(conv_in + 2 * qk_dim, hidden, weights.data(), 1, linear_inner, n_embd);
                }

                // Causal Depthwise Conv1D
                if (w_conv != model.tensors.end()) {
                    model.dequantize_to_f32(w_conv->second, weights.data());
                    recurrent_state.conv1d(layer, conv_in, weights.data(), conv_out);
                    silu_cpu(conv_out, conv_out, conv_channels);
                } else {
                    memcpy(conv_out, conv_in, conv_channels * sizeof(float));
                }

                // Split conv_out into q, k, v
                float *q_rec = conv_out;
                float *k_rec = conv_out + qk_dim;
                float *v_rec = conv_out + 2 * qk_dim;

                // Expand Q & K heads if key_heads < value_heads
                std::vector<float> q_expanded(value_heads * key_dim);
                std::vector<float> k_expanded(value_heads * key_dim);
                int64_t heads_per_group = value_heads / key_heads;
                for (int64_t vh = 0; vh < value_heads; vh++) {
                    int64_t kh = vh / heads_per_group;
                    memcpy(q_expanded.data() + vh * key_dim, q_rec + kh * key_dim, key_dim * sizeof(float));
                    memcpy(k_expanded.data() + vh * key_dim, k_rec + kh * key_dim, key_dim * sizeof(float));
                }

                // SSM alpha / decay and beta
                std::vector<float> alpha(value_heads, 0.0f);
                std::vector<float> beta(value_heads, 1.0f);
                if (w_alpha != model.tensors.end()) {
                    model.dequantize_to_f32(w_alpha->second, alpha.data());
                }
                if (w_beta != model.tensors.end()) {
                    model.dequantize_to_f32(w_beta->second, beta.data());
                }

                recurrent_state.delta_step(layer, q_expanded.data(), k_expanded.data(), v_rec,
                                           alpha.data(), beta.data(), delta_out);

                // Output projection
                if (w_O != model.tensors.end()) {
                    model.dequantize_to_f32(w_O->second, weights.data());
                    matmul_nt_cpu(attn_out, delta_out, weights.data(), 1, n_embd, linear_inner);
                } else {
                    memcpy(attn_out, delta_out, std::min(n_embd, linear_inner) * sizeof(float));
                }

                add_cpu(hidden, residual, attn_out, n_embd);
            }

            // SwiGLU FFN
            memcpy(residual, hidden, n_embd * sizeof(float));
            dequant_run((prefix + "ffn_norm.weight").c_str(), weights.data());
            rms_norm_cpu(hidden, hidden, weights.data(), n_embd, 1, arch.norm_eps);

            auto w_gate = model.tensors.find(prefix + "ffn_gate.weight");
            auto w_up = model.tensors.find(prefix + "ffn_up.weight");
            auto w_down = model.tensors.find(prefix + "ffn_down.weight");

            if (w_gate != model.tensors.end() && w_up != model.tensors.end()) {
                model.dequantize_to_f32(w_gate->second, weights.data());
                matmul_nt_cpu(gate_buf, hidden, weights.data(), 1, n_ff, n_embd);

                model.dequantize_to_f32(w_up->second, weights.data());
                matmul_nt_cpu(up_buf, hidden, weights.data(), 1, n_ff, n_embd);

                silu_cpu(gate_buf, gate_buf, n_ff);
                for (int64_t i = 0; i < n_ff; i++) gate_buf[i] *= up_buf[i];

                if (w_down != model.tensors.end()) {
                    model.dequantize_to_f32(w_down->second, weights.data());
                    matmul_nt_cpu(hidden, gate_buf, weights.data(), 1, n_embd, n_ff);
                }

                add_cpu(hidden, residual, hidden, n_embd);
            }
        }

        // Final RMSNorm
        auto norm_w = model.tensors.find("output_norm.weight");
        if (norm_w == model.tensors.end()) norm_w = model.tensors.find("norm.weight");
        if (norm_w != model.tensors.end()) {
            model.dequantize_to_f32(norm_w->second, weights.data());
            rms_norm_cpu(hidden, hidden, weights.data(), n_embd, 1, arch.norm_eps);
        }

        // Output logits projection
        auto out_w = model.tensors.find("output.weight");
        if (out_w == model.tensors.end()) out_w = model.tensors.find("token_embd.weight");
        if (out_w != model.tensors.end()) {
            model.dequantize_to_f32(out_w->second, weights.data());
            matmul_nt_cpu(logits, hidden, weights.data(), 1, n_vocab, n_embd);
        }

        return 0;
    }

private:
    ArchitectureSpec arch;
    int64_t seq_limit = 2048;
    Qwen35RecurrentState recurrent_state;
    std::vector<float> act;
    std::vector<float> weights;
    std::vector<float> full_attn_kv;
};

} // namespace

std::unique_ptr<ArchitectureDecoder> create_decoder(const ArchitectureSpec &spec, OpenClBackend *backend) {
    if (spec.kind == ArchitectureKind::Llama) {
        return std::make_unique<LlamaDecoderAdapter>(backend);
    }
    if (spec.kind == ArchitectureKind::Qwen35) {
        return std::make_unique<Qwen35DecoderAdapter>();
    }
    return nullptr;
}
