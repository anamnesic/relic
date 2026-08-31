#include "decoder.h"
#include "opencl_backend.h"
#include "qwen35_state.h"
#include "planner/adaptive_planner.h"
#include "memory/pinned_host_pool.h"
#include "memory/async_prefetcher.h"
#include "backends/intel_uhd_backend.h"
#include "memory/memory_engine.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{

    //------------------------------------------------------------------------------
    // CPU reference operations
    //------------------------------------------------------------------------------

    void rms_norm_cpu(float *out, const float *x, const float *weight, int64_t n, int64_t rows, float eps = 1e-5f)
    {
        for (int64_t r = 0; r < rows; r++)
        {
            float ss = 0.0f;
            for (int64_t i = 0; i < n; i++)
                ss += x[r * n + i] * x[r * n + i];
            float s = 1.0f / sqrtf(ss / (float)n + eps);
            for (int64_t i = 0; i < n; i++)
                out[r * n + i] = x[r * n + i] * s * weight[i];
        }
    }

    void silu_cpu(float *out, const float *x, int64_t n)
    {
        for (int64_t i = 0; i < n; i++)
        {
            out[i] = x[i] / (1.0f + expf(-x[i]));
        }
    }

    void matmul_cpu(float *out, const float *a, const float *b, int64_t m, int64_t n, int64_t k)
    {
        for (int64_t r = 0; r < m; r++)
        {
            for (int64_t c = 0; c < n; c++)
            {
                float sum = 0.0f;
                for (int64_t i = 0; i < k; i++)
                {
                    sum += a[r * k + i] * b[i * n + c];
                }
                out[r * n + c] = sum;
            }
        }
    }

    void matmul_nt_cpu(float *out, const float *a, const float *b, int64_t m, int64_t n, int64_t k)
    {
        for (int64_t r = 0; r < m; r++)
        {
            for (int64_t c = 0; c < n; c++)
            {
                float sum = 0.0f;
                for (int64_t i = 0; i < k; i++)
                {
                    sum += a[r * k + i] * b[c * k + i];
                }
                out[r * n + c] = sum;
            }
        }
    }

    void add_cpu(float *dst, const float *a, const float *b, int64_t n)
    {
        for (int64_t i = 0; i < n; i++)
            dst[i] = a[i] + b[i];
    }

    void rope_cpu(float *x, int64_t n_embd, int64_t n_head, int pos, int n_tokens, float base = 10000.0f)
    {
        int64_t head_dim = n_embd / n_head;
        for (int t = 0; t < n_tokens; t++)
        {
            for (int64_t h = 0; h < n_head; h++)
            {
                for (int64_t hh = 0; hh < head_dim / 2; hh++)
                {
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

    //------------------------------------------------------------------------------
    // Hexagonal Adapter: LlamaDecoderAdapter
    //------------------------------------------------------------------------------

    class LlamaDecoderAdapter final : public ArchitectureDecoder
    {
    public:
        explicit LlamaDecoderAdapter(OpenClBackend *backend) : cl(backend) {}

        bool init(const ArchitectureSpec &spec, int64_t max_seq_len, const ExecutionPlan *plan = nullptr) override
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

        void reset() override
        {
            std::fill(kv_cache.begin(), kv_cache.end(), 0.0f);
            std::fill(act.begin(), act.end(), 0.0f);
        }

        void save_state_checkpoint() override
        {
            snapshot_kv_cache_ = kv_cache;
            snapshot_act_ = act;
        }

        void restore_state_checkpoint() override
        {
            kv_cache = snapshot_kv_cache_;
            act = snapshot_act_;
        }

        int forward(const LlamaModel &model, int token_id, int64_t position, float *logits) override
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

            return 0;
        }

    private:
        ArchitectureSpec arch;
        OpenClBackend *cl = nullptr;
        const ExecutionPlan *plan_ = nullptr;
        int64_t seq_limit = 2048;
        std::vector<float> act;
        std::vector<float> weights;
        std::vector<float> kv_cache;
        std::vector<float> snapshot_kv_cache_;
        std::vector<float> snapshot_act_;
    };

    //------------------------------------------------------------------------------
    // Hexagonal Adapter: Qwen35DecoderAdapter (100% GPU VRAM In-Place Inference)
    //------------------------------------------------------------------------------

    class Qwen35DecoderAdapter final : public ArchitectureDecoder
    {
    public:
        explicit Qwen35DecoderAdapter(OpenClBackend *backend) : cl(backend) {}

        bool init(const ArchitectureSpec &spec, int64_t max_seq_len, const ExecutionPlan *plan = nullptr) override
        {
            arch = spec;
            seq_limit = max_seq_len;
            plan_ = plan;

            if (!recurrent_state.init(spec))
            {
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

            int64_t scratch_size = n_embd * 4 + q_size + kv_size * 2 + n_ff * 2 + n_head * max_seq_len + n_embd + total_qkv * 2 + linear_inner * 2;
            act.resize((size_t)scratch_size, 0.0f);
            int64_t max_layer_elements = std::max(n_embd * (int64_t)4096, std::max(n_embd * n_ff, total_qkv * n_embd));
            weights.resize((size_t)max_layer_elements, 0.0f);
            full_attn_kv.assign((size_t)(arch.n_layer * 2 * max_seq_len * n_embd), 0.0f);

            if (cl && cl->initialized)
            {
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
                gpu_conv_out.alloc(cl->dev.context, (size_t)total_qkv * sizeof(float));
                gpu_delta_out.alloc(cl->dev.context, (size_t)linear_inner * sizeof(float));
                gpu_q.alloc(cl->dev.context, (size_t)q_size * sizeof(float));
                gpu_k.alloc(cl->dev.context, (size_t)kv_size * sizeof(float));
                gpu_v.alloc(cl->dev.context, (size_t)kv_size * sizeof(float));
                gpu_alpha.alloc(cl->dev.context, (size_t)arch.linear_value_heads * sizeof(float));
                gpu_beta.alloc(cl->dev.context, (size_t)arch.linear_value_heads * sizeof(float));
                gpu_logits.alloc(cl->dev.context, (size_t)arch.n_vocab * sizeof(float));

                // Allocate in-VRAM GPU recurrent state, snapshots and GPU KV cache
                gpu_ssm_states.resize((size_t)arch.n_layer);
                gpu_conv_states.resize((size_t)arch.n_layer);
                gpu_ssm_snapshot_.resize((size_t)arch.n_layer);
                gpu_conv_snapshot_.resize((size_t)arch.n_layer);
                gpu_k_caches.resize((size_t)arch.n_layer);
                gpu_v_caches.resize((size_t)arch.n_layer);
                gpu_k_snapshot_.resize((size_t)arch.n_layer);
                gpu_v_snapshot_.resize((size_t)arch.n_layer);

                for (size_t l = 0; l < (size_t)arch.n_layer; l++)
                {
                    gpu_ssm_states[l].alloc(cl->dev.context, 16 * 128 * 128 * sizeof(float));
                    gpu_conv_states[l].alloc(cl->dev.context, 3 * total_qkv * sizeof(float));
                    gpu_ssm_snapshot_[l].alloc(cl->dev.context, 16 * 128 * 128 * sizeof(float));
                    gpu_conv_snapshot_[l].alloc(cl->dev.context, 3 * total_qkv * sizeof(float));
                    cl->fill(gpu_ssm_states[l], 0.0f, 16 * 128 * 128);
                    cl->fill(gpu_conv_states[l], 0.0f, 3 * total_qkv);

                    bool is_full_attn = (arch.full_attention_interval > 0) &&
                                        (((int64_t)l + 1) % arch.full_attention_interval == 0);
                    if (is_full_attn)
                    {
                        gpu_k_caches[l].alloc(cl->dev.context, (size_t)(max_seq_len * n_embd * sizeof(float)));
                        gpu_v_caches[l].alloc(cl->dev.context, (size_t)(max_seq_len * n_embd * sizeof(float)));
                        gpu_k_snapshot_[l].alloc(cl->dev.context, (size_t)(max_seq_len * n_embd * sizeof(float)));
                        gpu_v_snapshot_[l].alloc(cl->dev.context, (size_t)(max_seq_len * n_embd * sizeof(float)));
                        cl->fill(gpu_k_caches[l], 0.0f, max_seq_len * n_embd);
                        cl->fill(gpu_v_caches[l], 0.0f, max_seq_len * n_embd);
                    }
                }

                // Initialize AsyncPrefetcher with double staging slots
                prefetcher = std::make_unique<AsyncPrefetcher>(cl->dev.context, cl->dev.queue);
                prefetcher->initialize(64 * 1024 * 1024);
            }

            // Initialize Pinned Host Memory Pool with constant 128 MB sliding staging window
            pinned_pool = std::make_unique<PinnedHostPool>(128 * 1024 * 1024);

            // Initialize Intel UHD Shared Memory Backend with persistent activation buffers
            intel_backend = std::make_unique<IntelUhdBackend>();
            if (!intel_backend->initialize())
            {
                intel_backend.reset();
            }
            else
            {
                uhd_in_buf_ = intel_backend->allocate(16384 * sizeof(float), MemoryTier::TIER1_SHARED_IGPU);
                uhd_dst_buf_ = intel_backend->allocate(16384 * sizeof(float), MemoryTier::TIER1_SHARED_IGPU);
            }

            return true;
        }

        void save_state_checkpoint() override
        {
            snapshot_recurrent_ = recurrent_state.save_snapshot();
            snapshot_full_attn_kv_ = full_attn_kv;
            snapshot_act_ = act;
            if (use_gpu && cl && cl->initialized)
            {
                int64_t total_qkv = 2 * arch.linear_key_heads * arch.linear_key_head_dim + arch.linear_inner_size;
                for (size_t l = 0; l < gpu_ssm_states.size(); l++)
                {
                    if (gpu_ssm_states[l].mem && l < gpu_ssm_snapshot_.size() && gpu_ssm_snapshot_[l].mem)
                        clEnqueueCopyBuffer(cl->dev.queue, gpu_ssm_states[l].mem, gpu_ssm_snapshot_[l].mem, 0, 0, 16 * 128 * 128 * sizeof(float), 0, nullptr, nullptr);
                    if (gpu_conv_states[l].mem && l < gpu_conv_snapshot_.size() && gpu_conv_snapshot_[l].mem)
                        clEnqueueCopyBuffer(cl->dev.queue, gpu_conv_states[l].mem, gpu_conv_snapshot_[l].mem, 0, 0, (size_t)(3 * total_qkv * sizeof(float)), 0, nullptr, nullptr);
                }
                clFinish(cl->dev.queue);
            }
        }

        void restore_state_checkpoint() override
        {
            recurrent_state.restore_snapshot(snapshot_recurrent_);
            full_attn_kv = snapshot_full_attn_kv_;
            act = snapshot_act_;
            if (use_gpu && cl && cl->initialized)
            {
                int64_t total_qkv = 2 * arch.linear_key_heads * arch.linear_key_head_dim + arch.linear_inner_size;
                for (size_t l = 0; l < gpu_ssm_states.size(); l++)
                {
                    if (gpu_ssm_states[l].mem && l < gpu_ssm_snapshot_.size() && gpu_ssm_snapshot_[l].mem)
                        clEnqueueCopyBuffer(cl->dev.queue, gpu_ssm_snapshot_[l].mem, gpu_ssm_states[l].mem, 0, 0, 16 * 128 * 128 * sizeof(float), 0, nullptr, nullptr);
                    if (gpu_conv_states[l].mem && l < gpu_conv_snapshot_.size() && gpu_conv_snapshot_[l].mem)
                        clEnqueueCopyBuffer(cl->dev.queue, gpu_conv_snapshot_[l].mem, gpu_conv_states[l].mem, 0, 0, (size_t)(3 * total_qkv * sizeof(float)), 0, nullptr, nullptr);
                }
                clFinish(cl->dev.queue);
            }
        }

        void reset() override
        {
            recurrent_state.reset();
            std::fill(full_attn_kv.begin(), full_attn_kv.end(), 0.0f);
            std::fill(act.begin(), act.end(), 0.0f);
            if (use_gpu && cl && cl->initialized)
            {
                int64_t total_qkv = 2 * arch.linear_key_heads * arch.linear_key_head_dim + arch.linear_inner_size;
                for (size_t l = 0; l < gpu_ssm_states.size(); l++)
                {
                    if (gpu_ssm_states[l].mem)
                        cl->fill(gpu_ssm_states[l], 0.0f, 16 * 128 * 128);
                    if (gpu_conv_states[l].mem)
                        cl->fill(gpu_conv_states[l], 0.0f, 3 * total_qkv);
                    if (gpu_k_caches[l].mem)
                        cl->fill(gpu_k_caches[l], 0.0f, seq_limit * arch.n_embd);
                    if (gpu_v_caches[l].mem)
                        cl->fill(gpu_v_caches[l], 0.0f, seq_limit * arch.n_embd);
                }
            }
        }

        void warm_up(const LlamaModel &model) override
        {
            ensure_weights_uploaded(model);
        }

        void ensure_weights_uploaded(const LlamaModel &model)
        {
            if (!use_gpu || !cl || !cl->initialized || weights_uploaded)
                return;
            fprintf(stdout, "Uploading model weights to GPU VRAM (ExecutionPlan guided with On-the-Fly Q4 Repacking)...\n");
            fflush(stdout);
            size_t total_uploaded = 0;
            size_t total_original = 0;
            size_t total_offloaded_to_host = 0;

            for (const auto &kv : model.tensors)
            {
                total_original += kv.second.data.size();
                size_t uploaded_bytes = 0;

                BackendDeviceType target_dev = BackendDeviceType::NVIDIA_GPU;
                bool keep_in_vram = true;
                if (plan_)
                {
                    auto it = plan_->tensor_placements.find(kv.first);
                    if (it != plan_->tensor_placements.end())
                    {
                        target_dev = it->second.target_device;
                        keep_in_vram = it->second.keep_resident_in_vram;
                    }
                }

                if (target_dev == BackendDeviceType::INTEL_IGPU && intel_backend)
                {
                    auto uhd_buf = intel_backend->allocate(kv.second.data.size(), MemoryTier::TIER1_SHARED_IGPU);
                    if (uhd_buf)
                    {
                        intel_backend->upload(*uhd_buf, kv.second.data.data(), kv.second.data.size());
                        intel_tensors[kv.first] = std::move(uhd_buf);
                        total_uploaded += kv.second.data.size();
                    }
                }
                else if (target_dev == BackendDeviceType::NVIDIA_GPU && keep_in_vram)
                {
                    if (kv.first.find("norm") != std::string::npos || kv.first.find("bias") != std::string::npos)
                    {
                        // Pre-dequantize all norm weights to resident F32 in VRAM once!
                        std::vector<float> norm_f32(kv.second.nelements());
                        model.dequantize_to_f32(kv.second, norm_f32.data());
                        if (gpu_store.upload_auto_q4(cl->dev.context, cl->dev.queue, kv.first, norm_f32.data(), norm_f32.size() * sizeof(float), GgmlType::F32, kv.second.nelements(), uploaded_bytes))
                        {
                            total_uploaded += uploaded_bytes;
                        }
                    }
                    else if (gpu_store.upload_auto_q4(cl->dev.context, cl->dev.queue, kv.first, kv.second.data.data(), kv.second.data.size(), kv.second.type, kv.second.nelements(), uploaded_bytes))
                    {
                        total_uploaded += uploaded_bytes;
                    }
                }
                else
                {
                    // Stored in Host RAM: Allocate in PinnedHostPool for page-locked asynchronous DMA transfers!
                    if (pinned_pool)
                    {
                        void *p = pinned_pool->allocate_pinned(kv.second.data.size());
                        if (p)
                        {
                            memcpy(p, kv.second.data.data(), kv.second.data.size());
                            pinned_tensor_ptrs[kv.first] = p;
                            total_offloaded_to_host += kv.second.data.size();
                        }
                    }
                }
            }
            clFinish(cl->dev.queue);
            double footprint_red = (total_original > 0) ? ((double)total_original - (double)total_uploaded) / (double)total_original * 100.0 : 0.0;
            double pcie_red = (plan_) ? plan_->pcie_traffic_reduction_pct : ((total_original > 0) ? ((double)total_uploaded / (double)total_original * 100.0) : 100.0);

            fprintf(stdout, "VRAM Residency active: %.2f MB resident in GPU memory (Footprint Reduction: %.1f%%, PCIe Traffic Reduction: %.1f%%).\n",
                    (double)total_uploaded / (1024.0 * 1024.0), footprint_red, pcie_red);
            if (total_offloaded_to_host > 0)
            {
                fprintf(stdout, "[ExecutionPlan Sub-Layer Offload] %.2f MB managed in Pinned Host RAM via AsyncPrefetcher staging.\n",
                        (double)total_offloaded_to_host / (1024.0 * 1024.0));
            }
            fflush(stdout);
            weights_uploaded = true;
        }

        void dispatch_gemv(ClBuffer &dst, ClBuffer &in, const std::string &name, const LlamaModel::Tensor &t, int64_t N, int64_t K, int64_t layer_idx = 0)
        {
            // 1. Check if resident in dedicated VRAM
            ClBuffer *w_buf = gpu_store.get(name);
            GgmlType actual_type = gpu_store.get_type(name, t.type);
            if (w_buf)
            {
                if (actual_type == GgmlType::Q4_0)
                {
                    cl->gemv_q4_0(dst, in, *w_buf, N, K);
                    return;
                }
                else if (actual_type == GgmlType::Q8_0)
                {
                    cl->gemv_q8_0(dst, in, *w_buf, N, K);
                    return;
                }
                else if (actual_type == GgmlType::F32)
                {
                    cl->gemv_f32_nt(dst, in, *w_buf, N, K);
                    return;
                }
            }

            // 2. Check if resident on Intel UHD shared memory (True Cross-Context Execution Bridge)
            auto uhd_it = intel_tensors.find(name);
            if (uhd_it != intel_tensors.end() && intel_backend && uhd_in_buf_ && uhd_dst_buf_)
            {
                // Cross-Context Bridge:
                // GTX in -> Host Staging -> Intel UHD Buffer -> Compute on Intel iGPU -> Host Staging -> GTX dst
                if (weights.size() < (size_t)std::max(N, K))
                    weights.resize((size_t)std::max(N, K));

                clEnqueueReadBuffer(cl->dev.queue, in.mem, CL_TRUE, 0, (size_t)(K * sizeof(float)), weights.data(), 0, nullptr, nullptr);

                intel_backend->upload(*uhd_in_buf_, weights.data(), (size_t)(K * sizeof(float)));
                if (t.type == GgmlType::Q4_0)
                {
                    intel_backend->gemv_q4_0(*uhd_dst_buf_, *uhd_in_buf_, *(uhd_it->second), N, K);
                }
                else if (t.type == GgmlType::Q8_0)
                {
                    intel_backend->gemv_q8_0(*uhd_dst_buf_, *uhd_in_buf_, *(uhd_it->second), N, K);
                }
                intel_backend->synchronize();
                intel_backend->download(weights.data(), *uhd_dst_buf_, (size_t)(N * sizeof(float)));

                clEnqueueWriteBuffer(cl->dev.queue, dst.mem, CL_TRUE, 0, (size_t)(N * sizeof(float)), weights.data(), 0, nullptr, nullptr);
                return;
            }

            // 3. Check if stored in Pinned Host Pool (AsyncPrefetcher DMA Stream)
            auto pin_it = pinned_tensor_ptrs.find(name);
            if (pin_it != pinned_tensor_ptrs.end() && prefetcher)
            {
                int slot = (int)(layer_idx % 2);
                size_t raw_bytes = t.data.size();
                prefetcher->prefetch_async(pin_it->second, raw_bytes, slot);
                prefetcher->wait_ready(slot);
                cl_mem staging_mem = prefetcher->get_staging_mem(slot);
                if (staging_mem)
                {
                    ClBuffer staging_buf;
                    staging_buf.mem = staging_mem;
                    if (t.type == GgmlType::Q4_0)
                    {
                        cl->gemv_q4_0(dst, in, staging_buf, N, K);
                        return;
                    }
                    else if (t.type == GgmlType::Q8_0)
                    {
                        cl->gemv_q8_0(dst, in, staging_buf, N, K);
                        return;
                    }
                }
            }

            // Fallback: chunked dequantization to staging buffer
            const int64_t chunk_n = 4096;
            for (int64_t start_n = 0; start_n < N; start_n += chunk_n)
            {
                int64_t cur_n = std::min(chunk_n, N - start_n);
                model_dequant_rows(t, start_n, cur_n, weights.data());
                clEnqueueWriteBuffer(cl->dev.queue, gpu_weights.mem, CL_FALSE, 0, (size_t)(cur_n * K * sizeof(float)), weights.data(), 0, nullptr, nullptr);
                cl->matmul_f32_nt(gpu_act_dst, in, gpu_weights, 1, cur_n, K);
                clEnqueueCopyBuffer(cl->dev.queue, gpu_act_dst.mem, dst.mem, 0, (size_t)(start_n * sizeof(float)), (size_t)(cur_n * sizeof(float)), 0, nullptr, nullptr);
            }
        }

        void model_dequant_rows(const LlamaModel::Tensor &t, int64_t start_row, int64_t num_rows, float *out)
        {
            int64_t row_size = t.dims.empty() ? t.nelements() : t.dims[0];
            int64_t total_rows = t.dims.size() > 1 ? t.dims[1] : (row_size > 0 ? t.nelements() / row_size : 1);
            num_rows = std::min(num_rows, total_rows - start_row);
            if (num_rows <= 0)
                return;

            if (t.type == GgmlType::F32)
            {
                memcpy(out, ((const float *)t.data.data()) + start_row * row_size, (size_t)(num_rows * row_size * sizeof(float)));
            }
            else if (t.type == GgmlType::Q8_0)
            {
                size_t bytes_per_row = (size_t)((row_size + 31) / 32) * 34;
                for (int64_t r = 0; r < num_rows; r++)
                {
                    const uint8_t *src_row = t.data.data() + (start_row + r) * bytes_per_row;
                    float *dst_row = out + r * row_size;
                    int n_blocks = (int)(row_size / 32);
                    for (int b = 0; b < n_blocks; b++)
                    {
                        const uint8_t *b_ptr = src_row + b * 34;
                        uint16_t d_bits = (uint16_t)b_ptr[0] | ((uint16_t)b_ptr[1] << 8);
                        float d = float_to_half_val(d_bits);
                        const int8_t *qs = (const int8_t *)(b_ptr + 2);
                        for (int i = 0; i < 32; i++)
                            dst_row[b * 32 + i] = (float)qs[i] * d;
                    }
                }
            }
        }

        static float float_to_half_val(uint16_t h)
        {
            uint32_t sign = ((uint32_t)h >> 15) & 1;
            uint32_t exp = ((uint32_t)h >> 10) & 0x1f;
            uint32_t mant = (uint32_t)h & 0x3ff;
            if (exp == 0)
            {
                if (mant == 0)
                    return sign ? -0.0f : 0.0f;
                while ((mant & 0x400) == 0)
                {
                    mant <<= 1;
                    exp--;
                }
                exp++;
                mant &= 0x3ff;
            }
            else if (exp == 31)
            {
                return (mant == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
            }
            exp = exp + (127 - 15);
            uint32_t u = (sign << 31) | (exp << 23) | (mant << 13);
            float f;
            memcpy(&f, &u, 4);
            return f;
        }

        int forward(const LlamaModel &model, int token_id, int64_t position, float *logits) override
        {
            if (position >= seq_limit)
            {
                fprintf(stderr, "Maximum sequence length exceeded\n");
                return -1;
            }

            if (use_gpu)
                ensure_weights_uploaded(model);

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
            if (emb_it == model.tensors.end())
                emb_it = model.tensors.find("tok_embeddings.weight");
            if (emb_it == model.tensors.end() || emb_it->second.data.empty())
            {
                fprintf(stderr, "No token embedding tensor found\n");
                return -1;
            }

            const auto &embed = emb_it->second;

            if (use_gpu && cl && cl->initialized)
            {
                std::string emb_name = (emb_it->first == "token_embd.weight") ? "token_embd.weight" : "tok_embeddings.weight";
                ClBuffer *emb_buf = gpu_store.get(emb_name);
                GgmlType emb_type = gpu_store.get_type(emb_name, embed.type);
                if (emb_buf && emb_type == GgmlType::Q4_0)
                {
                    // Pure In-VRAM GPU Embedding Lookup (Zero CPU overhead & Zero PCIe writes)
                    cl->embed_lookup(gpu_hidden, *emb_buf, token_id, n_embd);
                }
                else
                {
                    if (embed.type == GgmlType::F32)
                    {
                        const float *emb_data = (const float *)embed.data.data();
                        memcpy(hidden, emb_data + token_id * n_embd, (size_t)(n_embd * sizeof(float)));
                    }
                    else
                    {
                        model.dequantize_rows_to_f32(embed, token_id, 1, hidden);
                    }
                    clEnqueueWriteBuffer(cl->dev.queue, gpu_hidden.mem, CL_FALSE, 0, (size_t)(n_embd * sizeof(float)), hidden, 0, nullptr, nullptr);
                }

                for (int64_t layer = 0; layer < arch.n_layer; layer++)
                {
                    std::string prefix = "blk." + std::to_string(layer) + ".";
                    bool is_full_attn = (arch.full_attention_interval > 0) &&
                                        ((layer + 1) % arch.full_attention_interval == 0);

                    cl->copy(gpu_residual, gpu_hidden, n_embd);

                    // In-VRAM RMSNorm (Zero CPU dequantization & Zero PCIe writes)
                    std::string norm_name = prefix + "attn_norm.weight";
                    ClBuffer *attn_norm_buf = gpu_store.get(norm_name);
                    if (!attn_norm_buf)
                    {
                        norm_name = prefix + "norm.weight";
                        attn_norm_buf = gpu_store.get(norm_name);
                    }
                    if (attn_norm_buf)
                    {
                        cl->rms_norm(gpu_hidden, gpu_hidden, *attn_norm_buf, n_embd, 1);
                    }

                    if (is_full_attn)
                    {
                        auto w_Q = model.tensors.find(prefix + "attn_q.weight");
                        auto w_K = model.tensors.find(prefix + "attn_k.weight");
                        auto w_V = model.tensors.find(prefix + "attn_v.weight");
                        auto w_O = model.tensors.find(prefix + "attn_output.weight");

                        if (w_Q != model.tensors.end())
                        {
                            int64_t actual_q_dim = w_Q->second.dims.size() > 1 ? w_Q->second.dims[1] : q_size;
                            dispatch_gemv(gpu_q, gpu_hidden, prefix + "attn_q.weight", w_Q->second, actual_q_dim, n_embd, layer);
                        }
                        if (w_K != model.tensors.end())
                        {
                            int64_t actual_k_dim = w_K->second.dims.size() > 1 ? w_K->second.dims[1] : kv_size;
                            dispatch_gemv(gpu_k, gpu_hidden, prefix + "attn_k.weight", w_K->second, actual_k_dim, n_embd, layer);
                        }
                        if (w_V != model.tensors.end())
                        {
                            int64_t actual_v_dim = w_V->second.dims.size() > 1 ? w_V->second.dims[1] : kv_size;
                            dispatch_gemv(gpu_v, gpu_hidden, prefix + "attn_v.weight", w_V->second, actual_v_dim, n_embd, layer);
                        }

                        cl->rope(gpu_q, q_size, n_head, position, 1);
                        cl->rope(gpu_k, kv_size, n_kv_head, position, 1);

                        // Pure GPU Causal Full Attention
                        cl->qwen_attention_step(gpu_q, gpu_k, gpu_v, gpu_k_caches[layer], gpu_v_caches[layer],
                                                gpu_attn_out, n_head, n_kv_head, head_dim, n_embd, position, seq_limit);

                        if (w_O != model.tensors.end())
                        {
                            dispatch_gemv(gpu_gate, gpu_attn_out, prefix + "attn_output.weight", w_O->second, n_embd, n_embd, layer);
                            cl->copy(gpu_attn_out, gpu_gate, n_embd);
                        }

                        cl->add(gpu_hidden, gpu_residual, gpu_attn_out, n_embd);
                    }
                    else
                    {
                        auto w_qkv = model.tensors.find(prefix + "attn_qkv.weight");
                        auto w_conv = model.tensors.find(prefix + "ssm_conv1d.weight");
                        auto w_alpha = model.tensors.find(prefix + "ssm_alpha.weight");
                        auto w_beta = model.tensors.find(prefix + "ssm_beta.weight");
                        auto w_O = model.tensors.find(prefix + "ssm_out.weight");
                        if (w_O == model.tensors.end())
                            w_O = model.tensors.find(prefix + "attn_output.weight");

                        if (w_qkv != model.tensors.end())
                        {
                            int64_t act_qkv = w_qkv->second.dims.size() > 1 ? w_qkv->second.dims[1] : total_qkv;
                            dispatch_gemv(gpu_conv_in, gpu_hidden, prefix + "attn_qkv.weight", w_qkv->second, act_qkv, n_embd, layer);
                        }

                        if (w_alpha != model.tensors.end())
                        {
                            dispatch_gemv(gpu_alpha, gpu_hidden, prefix + "ssm_alpha.weight", w_alpha->second, value_heads, n_embd, layer);
                        }
                        if (w_beta != model.tensors.end())
                        {
                            dispatch_gemv(gpu_beta, gpu_hidden, prefix + "ssm_beta.weight", w_beta->second, value_heads, n_embd, layer);
                        }

                        // Pure GPU in-VRAM Conv1D
                        ClBuffer *w_conv_buf = gpu_store.get(prefix + "ssm_conv1d.weight");
                        if (w_conv_buf)
                        {
                            cl->qwen_conv1d(gpu_conv_states[layer], gpu_conv_in, *w_conv_buf, gpu_conv_out, total_qkv);
                        }
                        else
                        {
                            cl->copy(gpu_conv_out, gpu_conv_in, total_qkv);
                        }

                        // Pure GPU in-VRAM Gated DeltaNet Step (Zero CPU-GPU sync!)
                        cl->qwen_deltanet(gpu_ssm_states[layer], gpu_conv_out, gpu_alpha, gpu_beta, gpu_delta_out,
                                          key_dim, qk_dim, linear_inner);

                        if (w_O != model.tensors.end())
                        {
                            dispatch_gemv(gpu_attn_out, gpu_delta_out, prefix + "ssm_out.weight", w_O->second, n_embd, linear_inner, layer);
                        }
                        else
                        {
                            cl->copy(gpu_attn_out, gpu_delta_out, std::min(n_embd, linear_inner));
                        }

                        cl->add(gpu_hidden, gpu_residual, gpu_attn_out, n_embd);
                    }

                    // FFN
                    cl->copy(gpu_residual, gpu_hidden, n_embd);

                    // In-VRAM FFN RMSNorm (Zero CPU dequantization & Zero PCIe writes)
                    std::string ffn_norm_name = prefix + "ffn_norm.weight";
                    ClBuffer *ffn_norm_buf = gpu_store.get(ffn_norm_name);
                    if (!ffn_norm_buf)
                    {
                        ffn_norm_name = prefix + "post_attention_norm.weight";
                        ffn_norm_buf = gpu_store.get(ffn_norm_name);
                    }
                    if (ffn_norm_buf)
                    {
                        cl->rms_norm(gpu_hidden, gpu_hidden, *ffn_norm_buf, n_embd, 1);
                    }

                    auto w_gate = model.tensors.find(prefix + "ffn_gate.weight");
                    auto w_up = model.tensors.find(prefix + "ffn_up.weight");
                    if (w_gate != model.tensors.end() && w_up != model.tensors.end())
                    {
                        ClBuffer *gate_buf = gpu_store.get(prefix + "ffn_gate.weight");
                        ClBuffer *up_buf = gpu_store.get(prefix + "ffn_up.weight");
                        GgmlType gate_type = gpu_store.get_type(prefix + "ffn_gate.weight", w_gate->second.type);
                        GgmlType up_type = gpu_store.get_type(prefix + "ffn_up.weight", w_up->second.type);
                        if (gate_buf && up_buf && gate_type == GgmlType::Q4_0 && up_type == GgmlType::Q4_0)
                        {
                            // Fused Multi-Row 8x FFN (Gate + Up + SwiGLU in 1 kernel pass)
                            cl->gemv_q4_0_ffn_swiglu(gpu_ffn_act, gpu_hidden, *gate_buf, *up_buf, n_ff, n_embd);
                        }
                        else
                        {
                            dispatch_gemv(gpu_gate, gpu_hidden, prefix + "ffn_gate.weight", w_gate->second, n_ff, n_embd, layer);
                            dispatch_gemv(gpu_up, gpu_hidden, prefix + "ffn_up.weight", w_up->second, n_ff, n_embd, layer);
                            cl->swiglu(gpu_ffn_act, gpu_gate, gpu_up, n_ff);
                        }

                        auto w_down = model.tensors.find(prefix + "ffn_down.weight");
                        if (w_down != model.tensors.end())
                        {
                            dispatch_gemv(gpu_hidden, gpu_ffn_act, prefix + "ffn_down.weight", w_down->second, n_embd, n_ff, layer);
                        }
                        cl->add(gpu_hidden, gpu_residual, gpu_hidden, n_embd);
                    }
                }

                // Final In-VRAM RMSNorm (Zero CPU dequantization & Zero PCIe writes)
                ClBuffer *out_norm_buf = gpu_store.get("output_norm.weight");
                if (!out_norm_buf)
                    out_norm_buf = gpu_store.get("norm.weight");
                if (out_norm_buf)
                {
                    cl->rms_norm(gpu_hidden, gpu_hidden, *out_norm_buf, n_embd, 1);
                }

                // Output logits projection
                if (logits != nullptr)
                {
                    auto out_w = model.tensors.find("output.weight");
                    if (out_w == model.tensors.end())
                        out_w = model.tensors.find("token_embd.weight");
                    if (out_w != model.tensors.end())
                    {
                        std::string out_name = (out_w->first == "output.weight") ? "output.weight" : "token_embd.weight";
                        dispatch_gemv(gpu_logits, gpu_hidden, out_name, out_w->second, n_vocab, n_embd);
                        clEnqueueReadBuffer(cl->dev.queue, gpu_logits.mem, CL_TRUE, 0, (size_t)(n_vocab * sizeof(float)), logits, 0, nullptr, nullptr);
                    }
                }
            }
            else
            {
                // CPU reference fallback
                for (int64_t layer = 0; layer < arch.n_layer; layer++)
                {
                    std::string prefix = "blk." + std::to_string(layer) + ".";
                    bool is_full_attn = (arch.full_attention_interval > 0) &&
                                        ((layer + 1) % arch.full_attention_interval == 0);

                    memcpy(residual, hidden, n_embd * sizeof(float));

                    std::string norm_name = prefix + "attn_norm.weight";
                    if (model.tensors.find(norm_name) == model.tensors.end())
                        norm_name = prefix + "norm.weight";
                    auto norm_it = model.tensors.find(norm_name);
                    if (norm_it != model.tensors.end())
                    {
                        model.dequantize_to_f32(norm_it->second, weights.data());
                        rms_norm_cpu(hidden, hidden, weights.data(), n_embd, 1, arch.norm_eps);
                    }

                    if (is_full_attn)
                    {
                        float *k_slice = full_attn_kv.data() + layer * 2 * seq_limit * n_embd;
                        float *v_slice = k_slice + seq_limit * n_embd;

                        auto w_Q = model.tensors.find(prefix + "attn_q.weight");
                        auto w_K = model.tensors.find(prefix + "attn_k.weight");
                        auto w_V = model.tensors.find(prefix + "attn_v.weight");
                        auto w_O = model.tensors.find(prefix + "attn_output.weight");

                        if (w_Q != model.tensors.end())
                        {
                            int64_t actual_q_dim = w_Q->second.dims.size() > 1 ? w_Q->second.dims[1] : q_size;
                            model.dequantize_to_f32(w_Q->second, weights.data());
                            matmul_nt_cpu(q_buf, hidden, weights.data(), 1, actual_q_dim, n_embd);
                        }
                        if (w_K != model.tensors.end())
                        {
                            int64_t actual_k_dim = w_K->second.dims.size() > 1 ? w_K->second.dims[1] : kv_size;
                            model.dequantize_to_f32(w_K->second, weights.data());
                            matmul_nt_cpu(k_buf, hidden, weights.data(), 1, actual_k_dim, n_embd);
                        }
                        if (w_V != model.tensors.end())
                        {
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
                    }
                    else
                    {
                        auto w_qkv = model.tensors.find(prefix + "attn_qkv.weight");
                        auto w_conv = model.tensors.find(prefix + "ssm_conv1d.weight");
                        auto w_alpha = model.tensors.find(prefix + "ssm_alpha.weight");
                        auto w_beta = model.tensors.find(prefix + "ssm_beta.weight");
                        auto w_O = model.tensors.find(prefix + "ssm_out.weight");
                        if (w_O == model.tensors.end())
                            w_O = model.tensors.find(prefix + "attn_output.weight");

                        if (w_qkv != model.tensors.end())
                        {
                            int64_t total_qkv = w_qkv->second.dims.size() > 1 ? w_qkv->second.dims[1] : (2 * qk_dim + linear_inner);
                            model.dequantize_to_f32(w_qkv->second, weights.data());
                            matmul_nt_cpu(conv_in, hidden, weights.data(), 1, total_qkv, n_embd);
                        }

                        if (w_conv != model.tensors.end())
                        {
                            model.dequantize_to_f32(w_conv->second, weights.data());
                            recurrent_state.conv1d(layer, conv_in, weights.data(), conv_out);
                            silu_cpu(conv_out, conv_out, total_qkv);
                        }
                        else
                        {
                            memcpy(conv_out, conv_in, total_qkv * sizeof(float));
                        }

                        float *q_rec = conv_out;
                        float *k_rec = conv_out + qk_dim;
                        float *v_rec = conv_out + 2 * qk_dim;

                        std::vector<float> q_expanded((size_t)(value_heads * key_dim));
                        std::vector<float> k_expanded((size_t)(value_heads * key_dim));
                        int64_t heads_per_group = std::max((int64_t)1, value_heads / key_heads);
                        for (int64_t vh = 0; vh < value_heads; vh++)
                        {
                            int64_t kh = vh / heads_per_group;
                            memcpy(q_expanded.data() + vh * key_dim, q_rec + kh * key_dim, (size_t)(key_dim * sizeof(float)));
                            memcpy(k_expanded.data() + vh * key_dim, k_rec + kh * key_dim, (size_t)(key_dim * sizeof(float)));
                        }

                        std::vector<float> alpha((size_t)value_heads, 0.0f);
                        std::vector<float> beta((size_t)value_heads, 1.0f);
                        if (w_alpha != model.tensors.end())
                        {
                            model.dequantize_to_f32(w_alpha->second, weights.data());
                            matmul_nt_cpu(alpha.data(), hidden, weights.data(), 1, value_heads, n_embd);
                        }
                        if (w_beta != model.tensors.end())
                        {
                            model.dequantize_to_f32(w_beta->second, weights.data());
                            matmul_nt_cpu(beta.data(), hidden, weights.data(), 1, value_heads, n_embd);
                        }

                        recurrent_state.delta_step(layer, q_expanded.data(), k_expanded.data(), v_rec,
                                                   alpha.data(), beta.data(), delta_out);

                        if (w_O != model.tensors.end())
                        {
                            model.dequantize_to_f32(w_O->second, weights.data());
                            matmul_nt_cpu(attn_out, delta_out, weights.data(), 1, n_embd, linear_inner);
                        }
                        else
                        {
                            memcpy(attn_out, delta_out, (size_t)(std::min(n_embd, linear_inner) * sizeof(float)));
                        }

                        add_cpu(hidden, residual, attn_out, n_embd);
                    }

                    // FFN
                    memcpy(residual, hidden, n_embd * sizeof(float));

                    std::string ffn_norm_name = prefix + "ffn_norm.weight";
                    if (model.tensors.find(ffn_norm_name) == model.tensors.end())
                        ffn_norm_name = prefix + "post_attention_norm.weight";
                    auto ffn_norm_it = model.tensors.find(ffn_norm_name);
                    if (ffn_norm_it != model.tensors.end())
                    {
                        model.dequantize_to_f32(ffn_norm_it->second, weights.data());
                        rms_norm_cpu(hidden, hidden, weights.data(), n_embd, 1, arch.norm_eps);
                    }

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

    private:
        ArchitectureSpec arch;
        OpenClBackend *cl = nullptr;
        const ExecutionPlan *plan_ = nullptr;
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
        ClBuffer gpu_conv_out;
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

        std::vector<ClBuffer> gpu_ssm_states;
        std::vector<ClBuffer> gpu_conv_states;
        std::vector<ClBuffer> gpu_k_caches;
        std::vector<ClBuffer> gpu_v_caches;

        std::vector<ClBuffer> gpu_ssm_snapshot_;
        std::vector<ClBuffer> gpu_conv_snapshot_;
        std::vector<ClBuffer> gpu_k_snapshot_;
        std::vector<ClBuffer> gpu_v_snapshot_;

        std::unique_ptr<PinnedHostPool> pinned_pool;
        std::unique_ptr<AsyncPrefetcher> prefetcher;
        std::unique_ptr<IntelUhdBackend> intel_backend;
        std::shared_ptr<BackendBuffer> uhd_in_buf_;
        std::shared_ptr<BackendBuffer> uhd_dst_buf_;
        std::unordered_map<std::string, void *> pinned_tensor_ptrs;
        std::unordered_map<std::string, std::shared_ptr<BackendBuffer>> intel_tensors;

        Qwen35RecurrentState::Snapshot snapshot_recurrent_;
        std::vector<float> snapshot_full_attn_kv_;
        std::vector<float> snapshot_act_;
    };

} // namespace

std::unique_ptr<ArchitectureDecoder> create_decoder(const ArchitectureSpec &spec, OpenClBackend *backend)
{
    if (spec.kind == ArchitectureKind::Llama)
    {
        return std::make_unique<LlamaDecoderAdapter>(backend);
    }
    if (spec.kind == ArchitectureKind::Qwen35)
    {
        return std::make_unique<Qwen35DecoderAdapter>(backend);
    }
    return nullptr;
}
