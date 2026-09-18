#pragma once

#include "model.h"
#include "opencl_backend.h"
#include "qwen35_weights_manager.h"

class Qwen35MlpBlock
{
public:
    explicit Qwen35MlpBlock(OpenClBackend *backend, Qwen35WeightsManager *weights_mgr);
    ~Qwen35MlpBlock() = default;

    void forward(int64_t layer, const ArchitectureSpec &arch, const LlamaModel &model,
                 ClBuffer &gpu_hidden, ClBuffer &gpu_residual, ClBuffer &gpu_attn_out,
                 ClBuffer &gpu_ffn_act, ClBuffer &gpu_gate, ClBuffer &gpu_up);

private:
    OpenClBackend *cl_ = nullptr;
    Qwen35WeightsManager *weights_mgr_ = nullptr;
};
