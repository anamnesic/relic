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

void rope_cpu(float *x, int64_t n_embd, int64_t n_head, int pos, int n_tokens, float base = 10000.0f) {
    int64_t head_dim = n_embd / n_head;
    for (int t = 0; t < n_tokens; t++) {
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
        weights.resize((size_t)std::max(n_embd * (int64_t)4096, n_embd * n_ff * 4), 0.0f);
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

            rope_cpu(q_buf, q_size, n_head, (int)position, 1, arch.rope_freq_base);
            rope_cpu(k_buf, kv_size, n_kv_head, (int)position, 1, arch.rope_freq_base);

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
            const int64_t chunk_v = 4096;
            for (int64_t v_start = 0; v_start < n_vocab; v_start += chunk_v) {
                int64_t cur_v = std::min(chunk_v, n_vocab - v_start);
                model.dequantize_rows_to_f32(out_w->second, v_start, cur_v, weights.data());
                matmul_nt_cpu(logits + v_start, hidden, weights.data(), 1, cur_v, n_embd);
            }
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
// Hexagonal Adapter: Qwen35DecoderAdapter (GPU Resident & Fused Kernels)
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

            gpu_hidden.alloc(cl->dev.context, (size_t)n_embd * sizeof(float));
            gpu_residual.alloc(cl->dev.context, (size_t)n_embd * sizeof(float));
            gpu_attn_out.alloc(cl->dev.context, (size_t)std::max(n_embd, linear_inner) * sizeof(float));
            gpu_gate.alloc(cl->dev.context, (size_t)n_ff * sizeof(float));
            gpu_up.alloc(cl->dev.context, (size_t)n_ff * sizeof(float));
            gpu_ffn_act.alloc(cl->dev.context, (size_t)n_ff * sizeof(float));
            gpu_conv_in.alloc(cl->dev.context, (size_t)total_qkv * sizeof(float));
            gpu_delta_out.alloc(cl->dev.context, (size_t)linear_inner * sizeof(float));
            gpu_q.alloc(cl->dev.context, (size_t)q_size * sizeof(float));
            gpu_k.alloc(cl->dev.context, (size_t)kv_size * sizeof(float));
            gpu_v.alloc(cl->dev.context, (size_t)kv_size * sizeof(float));
            gpu_alpha.alloc(cl->dev.context, (size_t)arch.linear_value_heads * sizeof(float));
            gpu_beta.alloc(cl->dev.context, (size_t)arch.linear_value_heads * sizeof(float));
            gpu_logits.alloc(cl->dev.context, (size_t)arch.n_vocab * sizeof(float));
        }

        return true;
    }

    void reset() override {
        recurrent_state.reset();
        std::fill(full_attn_kv.begin(), full_attn_kv.end(), 0.0f);
        std::fill(act.begin(), act.end(), 0.0f);
    }

    void ensure_weights_uploaded(const LlamaModel &model) {
        if (!use_gpu || !cl || !cl->initialized || weights_uploaded) return;
        fprintf(stdout, "Uploading model weights to GPU VRAM...\n");
        fflush(stdout);
        size_t total_uploaded = 0;
        for (const auto &kv : model.tensors) {
            if (gpu_store.upload(cl->dev.context, cl->dev.queue, kv.first, kv.second.data.data(), kv.second.data.size())) {
                total_uploaded += kv.second.data.size();
            }
        }
        clFinish(cl->dev.queue);
        fprintf(stdout, "VRAM Residency active: %.2f MB resident in GPU memory.\n", (double)total_uploaded / (1024.0 * 1024.0));
        fflush(stdout);
        weights_uploaded = true;
    }

    void dispatch_gemv(ClBuffer &dst, ClBuffer &in, const std::string &name, const LlamaModel::Tensor &t, int64_t N, int64_t K) {
        ClBuffer *w_buf = gpu_store.get(name);
        if (w_buf) {
            if (t.type == GgmlType::Q8_0) {
                cl->gemv_q8_0(dst, in, *w_buf, N, K);
                return;
            } else if (t.type == GgmlType::Q4_0) {
                cl->gemv_q4_0(dst, in, *w_buf, N, K);
                return;
            } else if (t.type == GgmlType::F32) {
                cl->gemv_f32_nt(dst, in, *w_buf, N, K);
                return;
            }
        }
        // Fallback: chunked dequantization to staging buffer
        const int64_t chunk_n = 4096;
        for (int64_t start_n = 0; start_n < N; start_n += chunk_n) {
            int64_t cur_n = std::min(chunk_n, N - start_n);
            model_dequant_rows(t, start_n, cur_n, weights.data());
            clEnqueueWriteBuffer(cl->dev.queue, gpu_weights.mem, CL_FALSE, 0, (size_t)(cur_n * K * sizeof(float)), weights.data(), 0, nullptr, nullptr);
            cl->matmul_f32_nt(gpu_act_dst, in, gpu_weights, 1, cur_n, K);
            clEnqueueCopyBuffer(cl->dev.queue, gpu_act_dst.mem, dst.mem, 0, (size_t)(start_n * sizeof(float)), (size_t)(cur_n * sizeof(float)), 0, nullptr, nullptr);
        }
    }

    void model_dequant_rows(const LlamaModel::Tensor &t, int64_t start_row, int64_t num_rows, float *out) {
        int64_t row_size = t.dims.empty() ? t.nelements() : t.dims[0];
        int64_t total_rows = t.dims.size() > 1 ? t.dims[1] : (row_size > 0 ? t.nelements() / row_size : 1);
        num_rows = std::min(num_rows, total_rows - start_row);
        if (num_rows <= 0) return;

        if (t.type == GgmlType::F32) {
            memcpy(out, ((const float *)t.data.data()) + start_row * row_size, (size_t)(num_rows * row_size * sizeof(float)));
        } else if (t.type == GgmlType::Q8_0) {
            size_t bytes_per_row = (size_t)((row_size + 31) / 32) * 34;
            for (int64_t r = 0; r < num_rows; r++) {
                const uint8_t *src_row = t.data.data() + (start_row + r) * bytes_per_row;
                float *dst_row = out + r * row_size;
                int n_blocks = (int)(row_size / 32);
                for (int b = 0; b < n_blocks; b++) {
                    const uint8_t *b_ptr = src_row + b * 34;
                    uint16_t d_bits = (uint16_t)b_ptr[0] | ((uint16_t)b_ptr[1] << 8);
                    float d = float_to_half_val(d_bits);
                    const int8_t *qs = (const int8_t *)(b_ptr + 2);
                    for (int i = 0; i < 32; i++) dst_row[b * 32 + i] = (float)qs[i] * d;
                }
            }
        }
    }

    static float float_to_half_val(uint16_t h) {
        uint32_t sign = ((uint32_t)h >> 15) & 1;
        uint32_t exp  = ((uint32_t)h >> 10) & 0x1f;
        uint32_t mant = (uint32_t)h & 0x3ff;
        if (exp == 0) {
            if (mant == 0) return sign ? -0.0f : 0.0f;
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            exp++; mant &= 0x3ff;
        } else if (exp == 31) {
            return (mant == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
        }
        exp = exp + (127 - 15);
        uint32_t u = (sign << 31) | (exp << 23) | (mant << 13);
        float f; memcpy(&f, &u, 4); return f;
    }

    int forward(const LlamaModel &model, int token_id, int64_t position, float *logits) override {
        if (position >= seq_limit) {
            fprintf(stderr, "Maximum sequence length exceeded\n");
            return -1;
        }

        if (use_gpu) ensure_weights_uploaded(model);

        int64_t n_embd = arch.n_embd;
        int64_t n_vocab = arch.n_vocab;
        int64_t n_head = arch.n_head;
        int64_t n_kv_head = arch.n_head_kv;
        int64_t head_dim = arch.linear_key_head_dim > 0 ? arch.linear_key_head_dim : (n_embd / n_head);
        int64_t n_ff = arch.n_ff;
        int64_t q_size = n_head * head_dim;
        int64_t kv_size = n_kv_head * head_dim;

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

        if (use_gpu && cl && cl->initialized) {
            clEnqueueWriteBuffer(cl->dev.queue, gpu_hidden.mem, CL_FALSE, 0, (size_t)(n_embd * sizeof(float)), hidden, 0, nullptr, nullptr);

            for (int64_t layer = 0; layer < arch.n_layer; layer++) {
                std::string prefix = "blk." + std::to_string(layer) + ".";
                bool is_full_attn = (arch.full_attention_interval > 0) &&
                                    ((layer + 1) % arch.full_attention_interval == 0);

                cl->copy(gpu_residual, gpu_hidden, n_embd);

                // RMSNorm
                std::string norm_name = prefix + "attn_norm.weight";
                if (model.tensors.find(norm_name) == model.tensors.end()) norm_name = prefix + "norm.weight";
                auto norm_it = model.tensors.find(norm_name);
                if (norm_it != model.tensors.end()) {
                    model.dequantize_to_f32(norm_it->second, weights.data());
                    clEnqueueWriteBuffer(cl->dev.queue, gpu_norm_w.mem, CL_FALSE, 0, (size_t)(n_embd * sizeof(float)), weights.data(), 0, nullptr, nullptr);
                    cl->rms_norm(gpu_hidden, gpu_hidden, gpu_norm_w, n_embd, 1);
                }

                if (is_full_attn) {
                    float *k_slice = full_attn_kv.data() + layer * 2 * seq_limit * n_embd;
                    float *v_slice = k_slice + seq_limit * n_embd;

                    auto w_Q = model.tensors.find(prefix + "attn_q.weight");
                    auto w_K = model.tensors.find(prefix + "attn_k.weight");
                    auto w_V = model.tensors.find(prefix + "attn_v.weight");
                    auto w_O = model.tensors.find(prefix + "attn_output.weight");

                    if (w_Q != model.tensors.end()) {
                        int64_t actual_q_dim = w_Q->second.dims.size() > 1 ? w_Q->second.dims[1] : q_size;
                        dispatch_gemv(gpu_q, gpu_hidden, prefix + "attn_q.weight", w_Q->second, actual_q_dim, n_embd);
                    }
                    if (w_K != model.tensors.end()) {
                        int64_t actual_k_dim = w_K->second.dims.size() > 1 ? w_K->second.dims[1] : kv_size;
                        dispatch_gemv(gpu_k, gpu_hidden, prefix + "attn_k.weight", w_K->second, actual_k_dim, n_embd);
                    }
                    if (w_V != model.tensors.end()) {
                        int64_t actual_v_dim = w_V->second.dims.size() > 1 ? w_V->second.dims[1] : kv_size;
                        dispatch_gemv(gpu_v, gpu_hidden, prefix + "attn_v.weight", w_V->second, actual_v_dim, n_embd);
                    }

                    clEnqueueReadBuffer(cl->dev.queue, gpu_q.mem, CL_FALSE, 0, (size_t)(q_size * sizeof(float)), q_buf, 0, nullptr, nullptr);
                    clEnqueueReadBuffer(cl->dev.queue, gpu_k.mem, CL_FALSE, 0, (size_t)(kv_size * sizeof(float)), k_buf, 0, nullptr, nullptr);
                    clEnqueueReadBuffer(cl->dev.queue, gpu_v.mem, CL_TRUE, 0, (size_t)(kv_size * sizeof(float)), v_buf, 0, nullptr, nullptr);

                    rope_cpu(q_buf, q_size, n_head, (int)position, 1, arch.rope_freq_base);
                    rope_cpu(k_buf, kv_size, n_kv_head, (int)position, 1, arch.rope_freq_base);

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

                    clEnqueueWriteBuffer(cl->dev.queue, gpu_attn_out.mem, CL_FALSE, 0, (size_t)(n_embd * sizeof(float)), attn_out, 0, nullptr, nullptr);

                    if (w_O != model.tensors.end()) {
                        dispatch_gemv(gpu_gate, gpu_attn_out, prefix + "attn_output.weight", w_O->second, n_embd, n_embd);
                        cl->copy(gpu_attn_out, gpu_gate, n_embd);
                    }

                    cl->add(gpu_hidden, gpu_residual, gpu_attn_out, n_embd);
                } else {
                    auto w_qkv = model.tensors.find(prefix + "attn_qkv.weight");
                    auto w_conv = model.tensors.find(prefix + "ssm_conv1d.weight");
                    auto w_alpha = model.tensors.find(prefix + "ssm_alpha.weight");
                    auto w_beta = model.tensors.find(prefix + "ssm_beta.weight");
                    auto w_O = model.tensors.find(prefix + "ssm_out.weight");
                    if (w_O == model.tensors.end()) w_O = model.tensors.find(prefix + "attn_output.weight");

                    if (w_qkv != model.tensors.end()) {
                        int64_t act_qkv = w_qkv->second.dims.size() > 1 ? w_qkv->second.dims[1] : total_qkv;
                        dispatch_gemv(gpu_conv_in, gpu_hidden, prefix + "attn_qkv.weight", w_qkv->second, act_qkv, n_embd);
                    }

                    if (w_alpha != model.tensors.end()) {
                        dispatch_gemv(gpu_alpha, gpu_hidden, prefix + "ssm_alpha.weight", w_alpha->second, value_heads, n_embd);
                    }
                    if (w_beta != model.tensors.end()) {
                        dispatch_gemv(gpu_beta, gpu_hidden, prefix + "ssm_beta.weight", w_beta->second, value_heads, n_embd);
                    }

                    clEnqueueReadBuffer(cl->dev.queue, gpu_conv_in.mem, CL_FALSE, 0, (size_t)(total_qkv * sizeof(float)), conv_in, 0, nullptr, nullptr);
                    std::vector<float> alpha((size_t)value_heads, 0.0f);
                    std::vector<float> beta((size_t)value_heads, 1.0f);
                    clEnqueueReadBuffer(cl->dev.queue, gpu_alpha.mem, CL_FALSE, 0, (size_t)(value_heads * sizeof(float)), alpha.data(), 0, nullptr, nullptr);
                    clEnqueueReadBuffer(cl->dev.queue, gpu_beta.mem, CL_TRUE, 0, (size_t)(value_heads * sizeof(float)), beta.data(), 0, nullptr, nullptr);

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

                    recurrent_state.delta_step(layer, q_expanded.data(), k_expanded.data(), v_rec,
                                               alpha.data(), beta.data(), delta_out);

                    clEnqueueWriteBuffer(cl->dev.queue, gpu_delta_out.mem, CL_FALSE, 0, (size_t)(linear_inner * sizeof(float)), delta_out, 0, nullptr, nullptr);

                    if (w_O != model.tensors.end()) {
                        dispatch_gemv(gpu_attn_out, gpu_delta_out, prefix + "ssm_out.weight", w_O->second, n_embd, linear_inner);
                    } else {
                        cl->copy(gpu_attn_out, gpu_delta_out, std::min(n_embd, linear_inner));
                    }

                    cl->add(gpu_hidden, gpu_residual, gpu_attn_out, n_embd);
                }

                // FFN
                cl->copy(gpu_residual, gpu_hidden, n_embd);

                std::string ffn_norm_name = prefix + "ffn_norm.weight";
                if (model.tensors.find(ffn_norm_name) == model.tensors.end()) ffn_norm_name = prefix + "post_attention_norm.weight";
                auto ffn_norm_it = model.tensors.find(ffn_norm_name);
                if (ffn_norm_it != model.tensors.end()) {
                    model.dequantize_to_f32(ffn_norm_it->second, weights.data());
                    clEnqueueWriteBuffer(cl->dev.queue, gpu_norm_w.mem, CL_FALSE, 0, (size_t)(n_embd * sizeof(float)), weights.data(), 0, nullptr, nullptr);
                    cl->rms_norm(gpu_hidden, gpu_hidden, gpu_norm_w, n_embd, 1);
                }

                auto w_gate = model.tensors.find(prefix + "ffn_gate.weight");
                auto w_up = model.tensors.find(prefix + "ffn_up.weight");
                auto w_down = model.tensors.find(prefix + "ffn_down.weight");

                if (w_gate != model.tensors.end() && w_up != model.tensors.end()) {
                    dispatch_gemv(gpu_gate, gpu_hidden, prefix + "ffn_gate.weight", w_gate->second, n_ff, n_embd);
                    dispatch_gemv(gpu_up, gpu_hidden, prefix + "ffn_up.weight", w_up->second, n_ff, n_embd);
                    cl->swiglu(gpu_ffn_act, gpu_gate, gpu_up, n_ff);

                    if (w_down != model.tensors.end()) {
                        dispatch_gemv(gpu_hidden, gpu_ffn_act, prefix + "ffn_down.weight", w_down->second, n_embd, n_ff);
                    }
                    cl->add(gpu_hidden, gpu_residual, gpu_hidden, n_embd);
                }
            }

            // Final RMSNorm
            auto norm_w = model.tensors.find("output_norm.weight");
            if (norm_w == model.tensors.end()) norm_w = model.tensors.find("norm.weight");
            if (norm_w != model.tensors.end()) {
                model.dequantize_to_f32(norm_w->second, weights.data());
                clEnqueueWriteBuffer(cl->dev.queue, gpu_norm_w.mem, CL_FALSE, 0, (size_t)(n_embd * sizeof(float)), weights.data(), 0, nullptr, nullptr);
                cl->rms_norm(gpu_hidden, gpu_hidden, gpu_norm_w, n_embd, 1);
            }

            // Output logits projection
            auto out_w = model.tensors.find("output.weight");
            if (out_w == model.tensors.end()) out_w = model.tensors.find("token_embd.weight");
            if (out_w != model.tensors.end()) {
                std::string out_name = (out_w->first == "output.weight") ? "output.weight" : "token_embd.weight";
                dispatch_gemv(gpu_logits, gpu_hidden, out_name, out_w->second, n_vocab, n_embd);
                clEnqueueReadBuffer(cl->dev.queue, gpu_logits.mem, CL_TRUE, 0, (size_t)(n_vocab * sizeof(float)), logits, 0, nullptr, nullptr);
            }
        } else {
            // CPU reference fallback
            for (int64_t layer = 0; layer < arch.n_layer; layer++) {
                std::string prefix = "blk." + std::to_string(layer) + ".";
                bool is_full_attn = (arch.full_attention_interval > 0) &&
                                    ((layer + 1) % arch.full_attention_interval == 0);

                memcpy(residual, hidden, n_embd * sizeof(float));

                std::string norm_name = prefix + "attn_norm.weight";
                if (model.tensors.find(norm_name) == model.tensors.end()) norm_name = prefix + "norm.weight";
                auto norm_it = model.tensors.find(norm_name);
                if (norm_it != model.tensors.end()) {
                    model.dequantize_to_f32(norm_it->second, weights.data());
                    rms_norm_cpu(hidden, hidden, weights.data(), n_embd, 1, arch.norm_eps);
                }

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
                        matmul_nt_cpu(q_buf, hidden, weights.data(), 1, actual_q_dim, n_embd);
                    }
                    if (w_K != model.tensors.end()) {
                        int64_t actual_k_dim = w_K->second.dims.size() > 1 ? w_K->second.dims[1] : kv_size;
                        model.dequantize_to_f32(w_K->second, weights.data());
                        matmul_nt_cpu(k_buf, hidden, weights.data(), 1, actual_k_dim, n_embd);
                    }
                    if (w_V != model.tensors.end()) {
                        int64_t actual_v_dim = w_V->second.dims.size() > 1 ? w_V->second.dims[1] : kv_size;
                        model.dequantize_to_f32(w_V->second, weights.data());
                        matmul_nt_cpu(v_buf, hidden, weights.data(), 1, actual_v_dim, n_embd);
                    }

                    rope_cpu(q_buf, q_size, n_head, (int)position, 1, arch.rope_freq_base);
                    rope_cpu(k_buf, kv_size, n_kv_head, (int)position, 1, arch.rope_freq_base);

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
                    auto w_qkv = model.tensors.find(prefix + "attn_qkv.weight");
                    auto w_conv = model.tensors.find(prefix + "ssm_conv1d.weight");
                    auto w_alpha = model.tensors.find(prefix + "ssm_alpha.weight");
                    auto w_beta = model.tensors.find(prefix + "ssm_beta.weight");
                    auto w_O = model.tensors.find(prefix + "ssm_out.weight");
                    if (w_O == model.tensors.end()) w_O = model.tensors.find(prefix + "attn_output.weight");

                    if (w_qkv != model.tensors.end()) {
                        int64_t total_qkv = w_qkv->second.dims.size() > 1 ? w_qkv->second.dims[1] : (2 * qk_dim + linear_inner);
                        model.dequantize_to_f32(w_qkv->second, weights.data());
                        matmul_nt_cpu(conv_in, hidden, weights.data(), 1, total_qkv, n_embd);
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
                        model.dequantize_to_f32(w_alpha->second, weights.data());
                        matmul_nt_cpu(alpha.data(), hidden, weights.data(), 1, value_heads, n_embd);
                    }
                    if (w_beta != model.tensors.end()) {
                        model.dequantize_to_f32(w_beta->second, weights.data());
                        matmul_nt_cpu(beta.data(), hidden, weights.data(), 1, value_heads, n_embd);
                    }

                    recurrent_state.delta_step(layer, q_expanded.data(), k_expanded.data(), v_rec,
                                               alpha.data(), beta.data(), delta_out);

                    if (w_O != model.tensors.end()) {
                        model.dequantize_to_f32(w_O->second, weights.data());
                        matmul_nt_cpu(attn_out, delta_out, weights.data(), 1, n_embd, linear_inner);
                    } else {
                        memcpy(attn_out, delta_out, (size_t)(std::min(n_embd, linear_inner) * sizeof(float)));
                    }

                    add_cpu(hidden, residual, attn_out, n_embd);
                }

                // FFN
                memcpy(residual, hidden, n_embd * sizeof(float));

                std::string ffn_norm_name = prefix + "ffn_norm.weight";
                if (model.tensors.find(ffn_norm_name) == model.tensors.end()) ffn_norm_name = prefix + "post_attention_norm.weight";
                auto ffn_norm_it = model.tensors.find(ffn_norm_name);
                if (ffn_norm_it != model.tensors.end()) {
                    model.dequantize_to_f32(ffn_norm_it->second, weights.data());
                    rms_norm_cpu(hidden, hidden, weights.data(), n_embd, 1, arch.norm_eps);
                }

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
                const int64_t chunk_v = 4096;
                for (int64_t v_start = 0; v_start < n_vocab; v_start += chunk_v) {
                    int64_t cur_v = std::min(chunk_v, n_vocab - v_start);
                    model.dequantize_rows_to_f32(out_w->second, v_start, cur_v, weights.data());
                    matmul_nt_cpu(logits + v_start, hidden, weights.data(), 1, cur_v, n_embd);
                }
            }
        }

        return 0;
    }

private:
    ArchitectureSpec arch;
    OpenClBackend *cl = nullptr;
    bool use_gpu = false;
    bool weights_uploaded = false;
    int64_t seq_limit = 2048;
    Qwen35RecurrentState recurrent_state;
    std::vector<float> act;
    std::vector<float> weights;
    std::vector<float> full_attn_kv;

    GpuTensorStore gpu_store;

    ClBuffer gpu_hidden;
    ClBuffer gpu_residual;
    ClBuffer gpu_attn_out;
    ClBuffer gpu_gate;
    ClBuffer gpu_up;
    ClBuffer gpu_ffn_act;
    ClBuffer gpu_conv_in;
    ClBuffer gpu_delta_out;
    ClBuffer gpu_q;
    ClBuffer gpu_k;
    ClBuffer gpu_v;
    ClBuffer gpu_alpha;
    ClBuffer gpu_beta;
    ClBuffer gpu_logits;
    ClBuffer gpu_norm_w;
    ClBuffer gpu_act_a;
    ClBuffer gpu_act_dst;
    ClBuffer gpu_weights;
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
