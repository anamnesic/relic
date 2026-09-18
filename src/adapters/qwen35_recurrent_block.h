#pragma once

#include "model.h"
#include "opencl_backend.h"
#include "qwen35_weights_manager.h"

class Qwen35RecurrentBlock
{
public:
    explicit Qwen35RecurrentBlock(OpenClBackend *backend, Qwen35WeightsManager *weights_mgr);
    ~Qwen35RecurrentBlock() = default;

    void forward(int64_t layer, const ArchitectureSpec &arch, const LlamaModel &model,
                 ClBuffer &gpu_hidden, ClBuffer &gpu_residual, ClBuffer &gpu_attn_out,
                 ClBuffer &gpu_conv_in, ClBuffer &gpu_conv_out, ClBuffer &gpu_delta_out,
                 ClBuffer &gpu_gate, ClBuffer &gpu_alpha, ClBuffer &gpu_beta,
                 ClBuffer &gpu_ssm_state, ClBuffer &gpu_conv_state,
                 int64_t total_qkv, int64_t key_dim, int64_t qk_dim, int64_t linear_inner, int64_t value_heads);

private:
    OpenClBackend *cl_ = nullptr;
    Qwen35WeightsManager *weights_mgr_ = nullptr;
};
