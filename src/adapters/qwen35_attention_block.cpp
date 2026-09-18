#include "qwen35_attention_block.h"
#include <algorithm>
#include <string>

Qwen35AttentionBlock::Qwen35AttentionBlock(OpenClBackend *backend, Qwen35WeightsManager *weights_mgr)
    : cl_(backend), weights_mgr_(weights_mgr)
{
}

void Qwen35AttentionBlock::forward(int64_t layer, int64_t position, int64_t seq_limit,
                                   const ArchitectureSpec &arch, const LlamaModel &model,
                                   ClBuffer &gpu_hidden, ClBuffer &gpu_residual, ClBuffer &gpu_attn_out,
                                   ClBuffer &gpu_q_full, ClBuffer &gpu_q, ClBuffer &gpu_attn_gate,
                                   ClBuffer &gpu_k, ClBuffer &gpu_v, ClBuffer &gpu_gate,
                                   ClBuffer &gpu_k_cache, ClBuffer &gpu_v_cache,
                                   int64_t q_size, int64_t kv_size, int64_t n_head, int64_t n_kv_head, int64_t head_dim)
{
    int64_t n_embd = arch.n_embd;
    std::string prefix = "blk." + std::to_string(layer) + ".";

    auto w_Q = model.tensors.find(prefix + "attn_q.weight");
    auto w_K = model.tensors.find(prefix + "attn_k.weight");
    auto w_V = model.tensors.find(prefix + "attn_v.weight");
    auto w_O = model.tensors.find(prefix + "attn_output.weight");

    if (w_Q != model.tensors.end())
    {
        weights_mgr_->dispatch_gemv(gpu_q_full, gpu_hidden, prefix + "attn_q.weight", w_Q->second, 2 * q_size, n_embd, layer);
        cl_->qwen_deinterleave_q_gate(gpu_q_full, gpu_q, gpu_attn_gate, n_head, head_dim);
    }
    if (w_K != model.tensors.end())
    {
        weights_mgr_->dispatch_gemv(gpu_k, gpu_hidden, prefix + "attn_k.weight", w_K->second, kv_size, n_embd, layer);
    }
    if (w_V != model.tensors.end())
    {
        weights_mgr_->dispatch_gemv(gpu_v, gpu_hidden, prefix + "attn_v.weight", w_V->second, kv_size, n_embd, layer);
    }

    // Per-head RMSNorm on Q and K
    ClBuffer *q_norm = weights_mgr_->get_tensor_buffer(prefix + "attn_q_norm.weight");
    if (q_norm)
    {
        cl_->qwen_qk_norm(gpu_q, gpu_q, *q_norm, n_head, head_dim, arch.norm_eps);
    }
    ClBuffer *k_norm = weights_mgr_->get_tensor_buffer(prefix + "attn_k_norm.weight");
    if (k_norm)
    {
        cl_->qwen_qk_norm(gpu_k, gpu_k, *k_norm, n_kv_head, head_dim, arch.norm_eps);
    }

    // RoPE with rope_dimension_count and rope_freq_base
    cl_->rope(gpu_q, q_size, n_head, position, 1, arch.rope_freq_base, arch.rope_dimension_count);
    cl_->rope(gpu_k, kv_size, n_kv_head, position, 1, arch.rope_freq_base, arch.rope_dimension_count);

    // Copy K and V to cache at position
    size_t kv_bytes = (size_t)(kv_size * sizeof(float));
    size_t kv_offset = (size_t)(position * kv_bytes);
    clEnqueueCopyBuffer(cl_->dev.queue, gpu_k.mem, gpu_k_cache.mem, 0, kv_offset, kv_bytes, 0, nullptr, nullptr);
    clEnqueueCopyBuffer(cl_->dev.queue, gpu_v.mem, gpu_v_cache.mem, 0, kv_offset, kv_bytes, 0, nullptr, nullptr);

    // Pure GPU Causal Full Attention
    cl_->qwen_attention_step(gpu_q, gpu_k_cache, gpu_v_cache,
                            gpu_attn_out, n_head, n_kv_head, head_dim, position, seq_limit);

    // Sigmoid Attention Gate Multiply: attn_out = attn_out * sigmoid(gate)
    cl_->qwen_attn_gate_mul(gpu_attn_out, gpu_attn_gate, q_size);

    if (w_O != model.tensors.end())
    {
        weights_mgr_->dispatch_gemv(gpu_gate, gpu_attn_out, prefix + "attn_output.weight", w_O->second, n_embd, n_embd, layer);
        cl_->copy(gpu_attn_out, gpu_gate, n_embd);
    }

    cl_->add(gpu_hidden, gpu_residual, gpu_attn_out, n_embd);
}
