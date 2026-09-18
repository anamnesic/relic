#pragma once

#include "model.h"
#include "opencl_backend.h"
#include "qwen35_weights_manager.h"

class Qwen35AttentionBlock
{
public:
    explicit Qwen35AttentionBlock(OpenClBackend *backend, Qwen35WeightsManager *weights_mgr);
    ~Qwen35AttentionBlock() = default;

    void forward(int64_t layer, int64_t position, int64_t seq_limit,
                 const ArchitectureSpec &arch, const LlamaModel &model,
                 ClBuffer &gpu_hidden, ClBuffer &gpu_residual, ClBuffer &gpu_attn_out,
                 ClBuffer &gpu_q_full, ClBuffer &gpu_q, ClBuffer &gpu_attn_gate,
                 ClBuffer &gpu_k, ClBuffer &gpu_v, ClBuffer &gpu_gate,
                 ClBuffer &gpu_k_cache, ClBuffer &gpu_v_cache,
                 int64_t q_size, int64_t kv_size, int64_t n_head, int64_t n_kv_head, int64_t head_dim);

private:
    OpenClBackend *cl_ = nullptr;
    Qwen35WeightsManager *weights_mgr_ = nullptr;
};
