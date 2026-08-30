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
        weights.resize((size_t)std::max(n_embd * arch.n_vocab, n_embd * n_ff * 4), 0.0f);
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
        if (embed.type == GgmlType::F32) {
            const float *emb_data = (const float *)embed.data.data();
            memcpy(hidden, emb_data + token_id * n_embd, (size_t)(n_embd * sizeof(float)));
        } else {
            model.dequantize_rows_to_f32(embed, token_id, 1, hidden);
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

            dequant_run((prefix + "attn_norm.weight").c_str(), weights.data());
            rms_norm_cpu(hidden, hidden, weights.data(), n_embd, 1, arch.norm_eps);

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

        auto norm_w = model.tensors.find("output_norm.weight");
        if (norm_w == model.tensors.end()) norm_w = model.tensors.find("norm.weight");
        if (norm_w != model.tensors.end()) {
            model.dequantize_to_f32(norm_w->second, weights.data());
            rms_norm_cpu(hidden, hidden, weights.data(), n_embd, 1, arch.norm_eps);
        }

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
    std::vector<float> act;
    std::vector<float> weights;
    std::vector<float> kv_cache;
};

//------------------------------------------------------------------------------
// Hexagonal Adapter: Qwen35DecoderAdapter
//------------------------------------------------------------------------------

class Qwen35DecoderAdapter final : public ArchitectureDecoder {
public:
    explicit Qwen35DecoderAdapter(OpenClBackend *backend) : cl(backend) {}

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
        int64_t head_dim = arch.linear_key_head_dim > 0 ? arch.linear_key_head_dim : (n_embd / n_head);
        int64_t n_ff = arch.n_ff;
        int64_t q_size = n_head * head_dim;
        int64_t kv_size = n_kv_head * head_dim;

        int64_t linear_inner = arch.linear_inner_size;
        int64_t key_dim = arch.linear_key_head_dim;
        int64_t key_heads = arch.linear_key_heads;
        int64_t qk_dim = key_heads * key_dim;
        int64_t total_qkv = 2 * qk_dim + linear_inner;
        int64_t scratch_size = n_embd * 4 + q_size + kv_size * 2
                             + n_ff * 2 + n_head * max_seq_len + n_embd
                             + total_qkv * 2 + linear_inner * 2;
        act.resize((size_t)scratch_size, 0.0f);
        int64_t max_layer_elements = std::max(n_embd * (int64_t)4096, std::max(n_embd * n_ff, total_qkv * n_embd));
        weights.resize((size_t)max_layer_elements, 0.0f);
        full_attn_kv.assign((size_t)(arch.n_layer * 2 * max_seq_len * n_embd), 0.0f);

        if (cl && cl->initialized) {
            use_gpu = true;
            size_t max_w_bytes = (size_t)max_layer_elements * sizeof(float);
            size_t max_act_bytes = (size_t)std::max((int64_t)4096, std::max(n_ff, total_qkv)) * sizeof(float);
            gpu_act_a.alloc(cl->dev.context, max_act_bytes);
            gpu_act_dst.alloc(cl->dev.context, max_act_bytes);
            gpu_weights.alloc(cl->dev.context, max_w_bytes);
            gpu_norm_w.alloc(cl->dev.context, (size_t)n_embd * sizeof(float));
        }

        return true;
    }

    void reset() override {
        recurrent_state.reset();
        std::fill(full_attn_kv.begin(), full_attn_kv.end(), 0.0f);
        std::fill(act.begin(), act.end(), 0.0f);
    }

    void matmul_nt(float *dst, const float *a, const float *b, int64_t M, int64_t N, int64_t K) {
        if (use_gpu && cl && cl->initialized) {
            const int64_t chunk_n = 4096;
            clEnqueueWriteBuffer(cl->dev.queue, gpu_act_a.mem, CL_FALSE, 0, (size_t)(M * K * sizeof(float)), a, 0, nullptr, nullptr);
            for (int64_t start_n = 0; start_n < N; start_n += chunk_n) {
                int64_t cur_n = std::min(chunk_n, N - start_n);
                const float *b_chunk = b + start_n * K;
                clEnqueueWriteBuffer(cl->dev.queue, gpu_weights.mem, CL_FALSE, 0, (size_t)(cur_n * K * sizeof(float)), b_chunk, 0, nullptr, nullptr);
                cl->matmul_f32_nt(gpu_act_dst, gpu_act_a, gpu_weights, M, cur_n, K);
                clEnqueueReadBuffer(cl->dev.queue, gpu_act_dst.mem, CL_TRUE, 0, (size_t)(M * cur_n * sizeof(float)), dst + start_n, 0, nullptr, nullptr);
            }
        } else {
            matmul_nt_cpu(dst, a, b, M, N, K);
        }
    }

    void rms_norm(float *out, const float *x, const float *weight, int64_t n, int64_t rows, float eps = 1e-5f) {
        if (use_gpu && cl && cl->initialized) {
            clEnqueueWriteBuffer(cl->dev.queue, gpu_act_a.mem, CL_FALSE, 0, (size_t)(rows * n * sizeof(float)), x, 0, nullptr, nullptr);
            clEnqueueWriteBuffer(cl->dev.queue, gpu_norm_w.mem, CL_FALSE, 0, (size_t)(n * sizeof(float)), weight, 0, nullptr, nullptr);
            cl->rms_norm(gpu_act_dst, gpu_act_a, gpu_norm_w, n, rows);
            clEnqueueReadBuffer(cl->dev.queue, gpu_act_dst.mem, CL_TRUE, 0, (size_t)(rows * n * sizeof(float)), out, 0, nullptr, nullptr);
        } else {
            rms_norm_cpu(out, x, weight, n, rows, eps);
        }
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
        int64_t head_dim = arch.linear_key_head_dim > 0 ? arch.linear_key_head_dim : (n_embd / n_head);
        int64_t n_ff = arch.n_ff;
        int64_t q_size = n_head * head_dim;
        int64_t kv_size = n_kv_head * head_dim;

        int64_t conv_channels = recurrent_state.channels();
        int64_t linear_inner = arch.linear_inner_size;
        int64_t key_dim = arch.linear_key_head_dim;
        int64_t key_heads = arch.linear_key_heads;
        int64_t value_heads = arch.linear_value_heads;
        int64_t qk_dim = key_heads * key_dim;
        int64_t total_qkv = 2 * qk_dim + linear_inner;

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
        float *conv_out = conv_in + total_qkv;
        float *delta_out = conv_out + total_qkv;

        // Embedding lookup
        auto emb_it = model.tensors.find("token_embd.weight");
        if (emb_it == model.tensors.end()) emb_it = model.tensors.find("tok_embeddings.weight");
        if (emb_it == model.tensors.end() || emb_it->second.data.empty()) {
            fprintf(stderr, "No token embedding tensor found\n");
            return -1;
        }

        const auto &embed = emb_it->second;
        if (embed.type == GgmlType::F32) {
            const float *emb_data = (const float *)embed.data.data();
            memcpy(hidden, emb_data + token_id * n_embd, (size_t)(n_embd * sizeof(float)));
        } else {
            model.dequantize_rows_to_f32(embed, token_id, 1, hidden);
        }

        auto dequant_run = [&](const char *name, float *buf) -> bool {
            auto it = model.tensors.find(name);
            if (it != model.tensors.end()) {
                model.dequantize_to_f32(it->second, buf);
                return true;
            }
            return false;
        };

        for (int64_t layer = 0; layer < arch.n_layer; layer++) {
            std::string prefix = "blk." + std::to_string(layer) + ".";
            bool is_full_attn = (arch.full_attention_interval > 0) &&
                                ((layer + 1) % arch.full_attention_interval == 0);

            memcpy(residual, hidden, n_embd * sizeof(float));

            if (!dequant_run((prefix + "attn_norm.weight").c_str(), weights.data())) {
                dequant_run((prefix + "norm.weight").c_str(), weights.data());
            }
            rms_norm(hidden, hidden, weights.data(), n_embd, 1, arch.norm_eps);

            if (is_full_attn) {
                float *k_slice = full_attn_kv.data() + layer * 2 * seq_limit * n_embd;
                float *v_slice = k_slice + seq_limit * n_embd;

                auto w_Q = model.tensors.find(prefix + "attn_q.weight");
                auto w_K = model.tensors.find(prefix + "attn_k.weight");
                auto w_V = model.tensors.find(prefix + "attn_v.weight");
                auto w_O = model.tensors.find(prefix + "attn_output.weight");

                if (w_Q != model.tensors.end()) {
                    int64_t actual_q_dim = w_Q->second.dims.size() > 1 ? w_Q->second.dims[1] : q_size;
                    model.dequantize_to_f32(w_Q->second, weights.data());
                    matmul_nt(q_buf, hidden, weights.data(), 1, actual_q_dim, n_embd);
                }
                if (w_K != model.tensors.end()) {
                    int64_t actual_k_dim = w_K->second.dims.size() > 1 ? w_K->second.dims[1] : kv_size;
                    model.dequantize_to_f32(w_K->second, weights.data());
                    matmul_nt(k_buf, hidden, weights.data(), 1, actual_k_dim, n_embd);
                }
                if (w_V != model.tensors.end()) {
                    int64_t actual_v_dim = w_V->second.dims.size() > 1 ? w_V->second.dims[1] : kv_size;
                    model.dequantize_to_f32(w_V->second, weights.data());
                    matmul_nt(v_buf, hidden, weights.data(), 1, actual_v_dim, n_embd);
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
                    matmul_nt(gate_buf, attn_out, weights.data(), 1, n_embd, n_embd);
                    memcpy(attn_out, gate_buf, n_embd * sizeof(float));
                }

                add_cpu(hidden, residual, attn_out, n_embd);
            } else {
                // Recurrent layer: support fused attn_qkv or separate attn_q/k/v
                auto w_qkv = model.tensors.find(prefix + "attn_qkv.weight");
                auto w_Q = model.tensors.find(prefix + "attn_q.weight");
                auto w_K = model.tensors.find(prefix + "attn_k.weight");
                auto w_V = model.tensors.find(prefix + "attn_v.weight");
                auto w_conv = model.tensors.find(prefix + "ssm_conv1d.weight");
                auto w_alpha = model.tensors.find(prefix + "ssm_alpha.weight");
                auto w_beta = model.tensors.find(prefix + "ssm_beta.weight");
                auto w_O = model.tensors.find(prefix + "ssm_out.weight");
                if (w_O == model.tensors.end()) w_O = model.tensors.find(prefix + "attn_output.weight");

                if (w_qkv != model.tensors.end()) {
                    int64_t total_qkv = w_qkv->second.dims.size() > 1 ? w_qkv->second.dims[1] : (2 * qk_dim + linear_inner);
                    model.dequantize_to_f32(w_qkv->second, weights.data());
                    matmul_nt(conv_in, hidden, weights.data(), 1, total_qkv, n_embd);
                } else {
                    if (w_Q != model.tensors.end()) {
                        model.dequantize_to_f32(w_Q->second, weights.data());
                        matmul_nt(conv_in, hidden, weights.data(), 1, qk_dim, n_embd);
                    }
                    if (w_K != model.tensors.end()) {
                        model.dequantize_to_f32(w_K->second, weights.data());
                        matmul_nt(conv_in + qk_dim, hidden, weights.data(), 1, qk_dim, n_embd);
                    }
                    if (w_V != model.tensors.end()) {
                        model.dequantize_to_f32(w_V->second, weights.data());
                        matmul_nt(conv_in + 2 * qk_dim, hidden, weights.data(), 1, linear_inner, n_embd);
                    }
                }

                if (w_conv != model.tensors.end()) {
                    model.dequantize_to_f32(w_conv->second, weights.data());
                    recurrent_state.conv1d(layer, conv_in, weights.data(), conv_out);
                    silu_cpu(conv_out, conv_out, total_qkv);
                } else {
                    memcpy(conv_out, conv_in, total_qkv * sizeof(float));
                }

                float *q_rec = conv_out;
                float *k_rec = conv_out + qk_dim;
                float *v_rec = conv_out + 2 * qk_dim;

                std::vector<float> q_expanded((size_t)(value_heads * key_dim));
                std::vector<float> k_expanded((size_t)(value_heads * key_dim));
                int64_t heads_per_group = std::max((int64_t)1, value_heads / key_heads);
                for (int64_t vh = 0; vh < value_heads; vh++) {
                    int64_t kh = vh / heads_per_group;
                    memcpy(q_expanded.data() + vh * key_dim, q_rec + kh * key_dim, (size_t)(key_dim * sizeof(float)));
                    memcpy(k_expanded.data() + vh * key_dim, k_rec + kh * key_dim, (size_t)(key_dim * sizeof(float)));
                }

                std::vector<float> alpha((size_t)value_heads, 0.0f);
                std::vector<float> beta((size_t)value_heads, 1.0f);
                if (w_alpha != model.tensors.end()) {
                    if (w_alpha->second.nelements() == value_heads) {
                        model.dequantize_to_f32(w_alpha->second, alpha.data());
                    } else {
                        model.dequantize_to_f32(w_alpha->second, weights.data());
                        matmul_nt(alpha.data(), hidden, weights.data(), 1, value_heads, n_embd);
                    }
                }
                if (w_beta != model.tensors.end()) {
                    if (w_beta->second.nelements() == value_heads) {
                        model.dequantize_to_f32(w_beta->second, beta.data());
                    } else {
                        model.dequantize_to_f32(w_beta->second, weights.data());
                        matmul_nt(beta.data(), hidden, weights.data(), 1, value_heads, n_embd);
                    }
                }

                recurrent_state.delta_step(layer, q_expanded.data(), k_expanded.data(), v_rec,
                                           alpha.data(), beta.data(), delta_out);

                if (w_O != model.tensors.end()) {
                    model.dequantize_to_f32(w_O->second, weights.data());
                    matmul_nt(attn_out, delta_out, weights.data(), 1, n_embd, linear_inner);
                } else {
                    memcpy(attn_out, delta_out, (size_t)(std::min(n_embd, linear_inner) * sizeof(float)));
                }

                add_cpu(hidden, residual, attn_out, n_embd);
            }

            // FFN
            memcpy(residual, hidden, n_embd * sizeof(float));
            if (!dequant_run((prefix + "ffn_norm.weight").c_str(), weights.data())) {
                dequant_run((prefix + "post_attention_norm.weight").c_str(), weights.data());
            }
            rms_norm(hidden, hidden, weights.data(), n_embd, 1, arch.norm_eps);

            auto w_gate = model.tensors.find(prefix + "ffn_gate.weight");
            auto w_up = model.tensors.find(prefix + "ffn_up.weight");
            auto w_down = model.tensors.find(prefix + "ffn_down.weight");

            if (w_gate != model.tensors.end() && w_up != model.tensors.end()) {
                model.dequantize_to_f32(w_gate->second, weights.data());
                matmul_nt(gate_buf, hidden, weights.data(), 1, n_ff, n_embd);

                model.dequantize_to_f32(w_up->second, weights.data());
                matmul_nt(up_buf, hidden, weights.data(), 1, n_ff, n_embd);

                silu_cpu(gate_buf, gate_buf, n_ff);
                for (int64_t i = 0; i < n_ff; i++) gate_buf[i] *= up_buf[i];

                if (w_down != model.tensors.end()) {
                    model.dequantize_to_f32(w_down->second, weights.data());
                    matmul_nt(hidden, gate_buf, weights.data(), 1, n_embd, n_ff);
                }

                add_cpu(hidden, residual, hidden, n_embd);
            }
        }

        // Final RMSNorm
        auto norm_w = model.tensors.find("output_norm.weight");
        if (norm_w == model.tensors.end()) norm_w = model.tensors.find("norm.weight");
        if (norm_w != model.tensors.end()) {
            model.dequantize_to_f32(norm_w->second, weights.data());
            rms_norm(hidden, hidden, weights.data(), n_embd, 1, arch.norm_eps);
        }

        // Output logits projection
        auto out_w = model.tensors.find("output.weight");
        if (out_w == model.tensors.end()) out_w = model.tensors.find("token_embd.weight");
        if (out_w != model.tensors.end()) {
            const int64_t chunk_v = 4096;
            for (int64_t v_start = 0; v_start < n_vocab; v_start += chunk_v) {
                int64_t cur_v = std::min(chunk_v, n_vocab - v_start);
                model.dequantize_rows_to_f32(out_w->second, v_start, cur_v, weights.data());
                matmul_nt(logits + v_start, hidden, weights.data(), 1, cur_v, n_embd);
            }
        }

        return 0;
    }

private:
    ArchitectureSpec arch;
    OpenClBackend *cl = nullptr;
    bool use_gpu = false;
    int64_t seq_limit = 2048;
    Qwen35RecurrentState recurrent_state;
    std::vector<float> act;
    std::vector<float> weights;
    std::vector<float> full_attn_kv;

    ClBuffer gpu_act_a;
    ClBuffer gpu_act_dst;
    ClBuffer gpu_weights;
    ClBuffer gpu_norm_w;
};

} // namespace

std::unique_ptr<ArchitectureDecoder> create_decoder(const ArchitectureSpec &spec, OpenClBackend *backend) {
    if (spec.kind == ArchitectureKind::Llama) {
        return std::make_unique<LlamaDecoderAdapter>(backend);
    }
    if (spec.kind == ArchitectureKind::Qwen35) {
        return std::make_unique<Qwen35DecoderAdapter>(backend);
    }
    return nullptr;
}
