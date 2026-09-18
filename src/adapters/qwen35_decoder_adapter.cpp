#include "qwen35_decoder_adapter.h"
#include "planner/adaptive_planner.h"
#include "cpu/cpu_ops.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>

using namespace relic::cpu;

Qwen35DecoderAdapter::Qwen35DecoderAdapter(OpenClBackend *backend) : cl(backend) {}

bool Qwen35DecoderAdapter::init(const ArchitectureSpec &spec, int64_t max_seq_len, const ExecutionPlan *plan)
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
    int64_t head_dim = (arch.n_head > 0) ? (n_embd / n_head) : 256;
    int64_t n_ff = arch.n_ff;
    int64_t q_size = n_head * head_dim;
    int64_t kv_size = n_kv_head * head_dim;

    int64_t linear_inner = arch.linear_inner_size;
    int64_t key_dim = arch.linear_key_head_dim;
    int64_t key_heads = arch.linear_key_heads;
    int64_t qk_dim = key_heads * key_dim;
    int64_t total_qkv = 2 * qk_dim + linear_inner;

    int64_t scratch_size = n_embd * 4 + q_size * 6 + kv_size * 2 + n_ff * 2 + n_head * max_seq_len + n_embd + total_qkv * 2 + linear_inner * 2 + 1024;
    act.resize((size_t)scratch_size, 0.0f);
    int64_t max_layer_elements = std::max(n_embd * (int64_t)4096, std::max(n_embd * n_ff, total_qkv * n_embd));
    weights.resize((size_t)max_layer_elements, 0.0f);
    full_attn_kv.assign((size_t)(arch.n_layer * 2 * max_seq_len * kv_size), 0.0f);

    if (cl && cl->initialized)
    {
        use_gpu = true;
        size_t max_w_bytes = (size_t)max_layer_elements * sizeof(float);
        size_t max_act_bytes = (size_t)std::max((int64_t)4096, std::max(n_ff, total_qkv)) * sizeof(float);
        gpu_act_a.alloc(cl->dev.context, max_act_bytes);
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
        gpu_q_full.alloc(cl->dev.context, (size_t)(2 * q_size) * sizeof(float));
        gpu_q.alloc(cl->dev.context, (size_t)q_size * sizeof(float));
        gpu_attn_gate.alloc(cl->dev.context, (size_t)q_size * sizeof(float));
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
                gpu_k_caches[l].alloc(cl->dev.context, (size_t)(max_seq_len * kv_size * sizeof(float)));
                gpu_v_caches[l].alloc(cl->dev.context, (size_t)(max_seq_len * kv_size * sizeof(float)));
                gpu_k_snapshot_[l].alloc(cl->dev.context, (size_t)(max_seq_len * kv_size * sizeof(float)));
                gpu_v_snapshot_[l].alloc(cl->dev.context, (size_t)(max_seq_len * kv_size * sizeof(float)));
                cl->fill(gpu_k_caches[l], 0.0f, max_seq_len * kv_size);
                cl->fill(gpu_v_caches[l], 0.0f, max_seq_len * kv_size);
            }
        }

        weights_mgr = std::make_unique<Qwen35WeightsManager>(cl);
        weights_mgr->init(arch, plan);
        recurrent_block = std::make_unique<Qwen35RecurrentBlock>(cl, weights_mgr.get());
        attention_block = std::make_unique<Qwen35AttentionBlock>(cl, weights_mgr.get());
        mlp_block = std::make_unique<Qwen35MlpBlock>(cl, weights_mgr.get());
    }

    return true;
}

void Qwen35DecoderAdapter::save_state_checkpoint()
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

void Qwen35DecoderAdapter::restore_state_checkpoint()
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

void Qwen35DecoderAdapter::reset()
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
            int64_t hd = (arch.n_head > 0) ? (arch.n_embd / arch.n_head) : 256;
            int64_t kv_sz = arch.n_head_kv * hd;
            if (gpu_k_caches[l].mem)
                cl->fill(gpu_k_caches[l], 0.0f, seq_limit * kv_sz);
            if (gpu_v_caches[l].mem)
                cl->fill(gpu_v_caches[l], 0.0f, seq_limit * kv_sz);
        }
    }
}

void Qwen35DecoderAdapter::warm_up(const LlamaModel &model)
{
    ensure_weights_uploaded(model);
}

void Qwen35DecoderAdapter::ensure_weights_uploaded(const LlamaModel &model)
{
    if (weights_mgr)
        weights_mgr->ensure_weights_uploaded(model);
}

void Qwen35DecoderAdapter::dispatch_gemv(ClBuffer &dst, ClBuffer &in, const std::string &name, const LlamaModel::Tensor &t, int64_t N, int64_t K, int64_t layer_idx)
{
    if (weights_mgr)
        weights_mgr->dispatch_gemv(dst, in, name, t, N, K, layer_idx);
}

void Qwen35DecoderAdapter::model_dequant_rows(const LlamaModel::Tensor &t, int64_t start_row, int64_t num_rows, float *out)
{
    int64_t row_size = t.dims.empty() ? t.nelements() : t.dims[0];
    int64_t total_rows = t.dims.size() > 1 ? t.dims[1] : (row_size > 0 ? t.nelements() / row_size : 1);
    num_rows = std::min(num_rows, total_rows - start_row);
    relic::quant::dequantize_rows(t.type, t.data.data(), row_size, start_row, num_rows, out);
}

float Qwen35DecoderAdapter::float_to_half_val(uint16_t h)
{
    return relic::quant::half_bits_to_float(h);
}

int Qwen35DecoderAdapter::forward(const LlamaModel &model, int token_id, int64_t position, float *logits, bool compute_output)
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
    int64_t head_dim = (n_head > 0) ? (n_embd / n_head) : 256;
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
    float *q_full_buf = residual + n_embd;
    float *q_buf = q_full_buf + 2 * q_size;
    float *attn_gate_buf = q_buf + q_size;
    float *k_buf = attn_gate_buf + q_size;
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
        ClBuffer *emb_buf = weights_mgr->get_tensor_buffer(emb_name);
        GgmlType emb_type = weights_mgr->get_tensor_type(emb_name, embed.type);
        if (emb_buf && emb_type == GgmlType::Q4_0)
        {
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

        // Initialize residual stream and pre-normalize for Layer 0
        cl->copy(gpu_residual, gpu_hidden, n_embd);
        std::string l0_norm_name = "blk.0.attn_norm.weight";
        ClBuffer *l0_norm_buf = weights_mgr->get_tensor_buffer(l0_norm_name);
        if (!l0_norm_buf)
        {
            l0_norm_name = "blk.0.norm.weight";
            l0_norm_buf = weights_mgr->get_tensor_buffer(l0_norm_name);
        }
        if (l0_norm_buf)
        {
            cl->rms_norm(gpu_hidden, gpu_hidden, *l0_norm_buf, n_embd, 1);
        }

        for (int64_t layer = 0; layer < arch.n_layer; layer++)
        {
            bool is_full_attn = (arch.full_attention_interval > 0) &&
                                ((layer + 1) % arch.full_attention_interval == 0);

            if (is_full_attn)
            {
                attention_block->forward(layer, position, seq_limit, arch, model,
                                         gpu_hidden, gpu_residual, gpu_attn_out,
                                         gpu_q_full, gpu_q, gpu_attn_gate,
                                         gpu_k, gpu_v, gpu_gate,
                                         gpu_k_caches[layer], gpu_v_caches[layer],
                                         q_size, kv_size, n_head, n_kv_head, head_dim);
            }
            else
            {
                recurrent_block->forward(layer, arch, model,
                                         gpu_hidden, gpu_residual, gpu_attn_out,
                                         gpu_conv_in, gpu_conv_out, gpu_delta_out,
                                         gpu_gate, gpu_alpha, gpu_beta,
                                         gpu_ssm_states[layer], gpu_conv_states[layer],
                                         total_qkv, key_dim, qk_dim, linear_inner, value_heads);
            }

            // mlp_block internally fuses: residual += attn_out, hidden = rms_norm(residual, ffn_norm)
            // and computes down-projection into gpu_attn_out
            mlp_block->forward(layer, arch, model, gpu_hidden, gpu_residual, gpu_attn_out,
                               gpu_ffn_act, gpu_gate, gpu_up);

            // Inter-layer fused transition: residual += ffn_down, hidden = rms_norm(residual, next_norm)
            ClBuffer *next_norm_buf = nullptr;
            if (layer + 1 < arch.n_layer)
            {
                std::string next_prefix = "blk." + std::to_string(layer + 1) + ".";
                next_norm_buf = weights_mgr->get_tensor_buffer(next_prefix + "attn_norm.weight");
                if (!next_norm_buf)
                    next_norm_buf = weights_mgr->get_tensor_buffer(next_prefix + "norm.weight");
            }
            else
            {
                next_norm_buf = weights_mgr->get_tensor_buffer("output_norm.weight");
                if (!next_norm_buf)
                    next_norm_buf = weights_mgr->get_tensor_buffer("norm.weight");
            }

            if (next_norm_buf)
            {
                cl->add_rms_norm(gpu_residual, gpu_attn_out, *next_norm_buf, gpu_hidden, n_embd, arch.norm_eps);
            }
            else
            {
                cl->add(gpu_residual, gpu_residual, gpu_attn_out, n_embd);
                cl->copy(gpu_hidden, gpu_residual, n_embd);
            }
        }

        // Output logits projection
        if (compute_output)
        {
            auto out_w = model.tensors.find("output.weight");
            if (out_w == model.tensors.end())
                out_w = model.tensors.find("token_embd.weight");
            if (out_w != model.tensors.end())
            {
                std::string out_name = (out_w->first == "output.weight") ? "output.weight" : "token_embd.weight";
                dispatch_gemv(gpu_logits, gpu_hidden, out_name, out_w->second, n_vocab, n_embd);
                if (logits != nullptr)
                {
                    clEnqueueReadBuffer(cl->dev.queue, gpu_logits.mem, CL_TRUE, 0, (size_t)(n_vocab * sizeof(float)), logits, 0, nullptr, nullptr);
                }
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
                float *k_slice = full_attn_kv.data() + layer * 2 * seq_limit * kv_size;
                float *v_slice = k_slice + seq_limit * kv_size;

                auto w_Q = model.tensors.find(prefix + "attn_q.weight");
                auto w_K = model.tensors.find(prefix + "attn_k.weight");
                auto w_V = model.tensors.find(prefix + "attn_v.weight");
                auto w_O = model.tensors.find(prefix + "attn_output.weight");

                if (w_Q != model.tensors.end())
                {
                    model.dequantize_to_f32(w_Q->second, weights.data());
                    matmul_nt_cpu(q_full_buf, hidden, weights.data(), 1, 2 * q_size, n_embd);
                    for (int64_t h = 0; h < n_head; h++)
                    {
                        int64_t src_base = h * (2 * head_dim);
                        for (int64_t d = 0; d < head_dim; d++)
                        {
                            q_buf[h * head_dim + d] = q_full_buf[src_base + d];
                            attn_gate_buf[h * head_dim + d] = q_full_buf[src_base + head_dim + d];
                        }
                    }
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

                auto w_qn = model.tensors.find(prefix + "attn_q_norm.weight");
                if (w_qn != model.tensors.end())
                {
                    std::vector<float> qn_w(head_dim);
                    model.dequantize_to_f32(w_qn->second, qn_w.data());
                    for (int64_t h = 0; h < n_head; h++)
                        rms_norm_cpu(q_buf + h * head_dim, q_buf + h * head_dim, qn_w.data(), head_dim, 1, arch.norm_eps);
                }
                auto w_kn = model.tensors.find(prefix + "attn_k_norm.weight");
                if (w_kn != model.tensors.end())
                {
                    std::vector<float> kn_w(head_dim);
                    model.dequantize_to_f32(w_kn->second, kn_w.data());
                    for (int64_t h = 0; h < n_kv_head; h++)
                        rms_norm_cpu(k_buf + h * head_dim, k_buf + h * head_dim, kn_w.data(), head_dim, 1, arch.norm_eps);
                }

                rope_cpu(q_buf, q_size, n_head, (int)position, 1, arch.rope_freq_base, arch.rope_dimension_count);
                rope_cpu(k_buf, kv_size, n_kv_head, (int)position, 1, arch.rope_freq_base, arch.rope_dimension_count);

                memcpy(k_slice + position * kv_size, k_buf, kv_size * sizeof(float));
                memcpy(v_slice + position * kv_size, v_buf, kv_size * sizeof(float));

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
                            sum += q_buf[h * head_dim + d] * k_slice[s * kv_size + h_kv * head_dim + d];
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
                    float inv_sum = 1.0f / (sum > 0.0f ? sum : 1.0f);
                    for (int64_t s = 0; s < S; s++)
                        scores[offset + s] *= inv_sum;
                }

                memset(attn_out, 0, q_size * sizeof(float));
                for (int64_t h = 0; h < n_head; h++)
                {
                    int64_t h_kv = h / q_per_kv;
                    for (int64_t s = 0; s < S; s++)
                    {
                        float w = scores[h * S + s];
                        for (int64_t d = 0; d < head_dim; d++)
                        {
                            attn_out[h * head_dim + d] += w * v_slice[s * kv_size + h_kv * head_dim + d];
                        }
                    }
                }

                for (int64_t i = 0; i < q_size; i++)
                {
                    float sig = 1.0f / (1.0f + expf(-attn_gate_buf[i]));
                    attn_out[i] *= sig;
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
                    int64_t total_qkv_act = w_qkv->second.dims.size() > 1 ? w_qkv->second.dims[1] : (2 * qk_dim + linear_inner);
                    model.dequantize_to_f32(w_qkv->second, weights.data());
                    matmul_nt_cpu(conv_in, hidden, weights.data(), 1, total_qkv_act, n_embd);
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
        if (out_w != model.tensors.end() && logits)
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

int Qwen35DecoderAdapter::sample_token(float temperature, int top_k, float top_p)
{
    if (use_gpu && cl && cl->initialized)
    {
        return cl->sample_logits(gpu_logits, arch.n_vocab, temperature, top_k, top_p);
    }
    return 0;
}
