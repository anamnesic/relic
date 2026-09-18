#include "qwen35_recurrent_block.h"
#include <algorithm>
#include <string>

Qwen35RecurrentBlock::Qwen35RecurrentBlock(OpenClBackend *backend, Qwen35WeightsManager *weights_mgr)
    : cl_(backend), weights_mgr_(weights_mgr)
{
}

void Qwen35RecurrentBlock::forward(int64_t layer, const ArchitectureSpec &arch, const LlamaModel &model,
                                   ClBuffer &gpu_hidden, ClBuffer &gpu_residual, ClBuffer &gpu_attn_out,
                                   ClBuffer &gpu_conv_in, ClBuffer &gpu_conv_out, ClBuffer &gpu_delta_out,
                                   ClBuffer &gpu_gate, ClBuffer &gpu_alpha, ClBuffer &gpu_beta,
                                   ClBuffer &gpu_ssm_state, ClBuffer &gpu_conv_state,
                                   int64_t total_qkv, int64_t key_dim, int64_t qk_dim, int64_t linear_inner, int64_t value_heads)
{
    int64_t n_embd = arch.n_embd;
    std::string prefix = "blk." + std::to_string(layer) + ".";

    auto w_qkv = model.tensors.find(prefix + "attn_qkv.weight");
    auto w_gate = model.tensors.find(prefix + "attn_gate.weight");
    auto w_alpha = model.tensors.find(prefix + "ssm_alpha.weight");
    auto w_beta = model.tensors.find(prefix + "ssm_beta.weight");
    auto w_O = model.tensors.find(prefix + "ssm_out.weight");
    if (w_O == model.tensors.end())
        w_O = model.tensors.find(prefix + "attn_output.weight");

    if (w_qkv != model.tensors.end())
    {
        int64_t act_qkv = w_qkv->second.dims.size() > 1 ? w_qkv->second.dims[1] : total_qkv;
        weights_mgr_->dispatch_gemv(gpu_conv_in, gpu_hidden, prefix + "attn_qkv.weight", w_qkv->second, act_qkv, n_embd, layer);
    }
    if (w_gate != model.tensors.end())
    {
        weights_mgr_->dispatch_gemv(gpu_gate, gpu_hidden, prefix + "attn_gate.weight", w_gate->second, linear_inner, n_embd, layer);
    }
    if (w_alpha != model.tensors.end())
    {
        weights_mgr_->dispatch_gemv(gpu_alpha, gpu_hidden, prefix + "ssm_alpha.weight", w_alpha->second, value_heads, n_embd, layer);
    }
    if (w_beta != model.tensors.end())
    {
        weights_mgr_->dispatch_gemv(gpu_beta, gpu_hidden, prefix + "ssm_beta.weight", w_beta->second, value_heads, n_embd, layer);
    }

    // Pure GPU in-VRAM Conv1D
    ClBuffer *w_conv_buf = weights_mgr_->get_tensor_buffer(prefix + "ssm_conv1d.weight");
    if (w_conv_buf)
    {
        cl_->qwen_conv1d(gpu_conv_state, gpu_conv_in, *w_conv_buf, gpu_conv_out, total_qkv);
    }
    else
    {
        cl_->copy(gpu_conv_out, gpu_conv_in, total_qkv);
    }

    // Fused Recurrent DeltaNet Step + SSM Norm + SiLU Gate + Multiply
    std::string ssm_norm_name = prefix + "ssm_norm.weight";
    ClBuffer *ssm_norm_buf = weights_mgr_->get_tensor_buffer(ssm_norm_name);
    ClBuffer *attn_gate_buf = (w_gate != model.tensors.end()) ? &gpu_gate : nullptr;
    ClBuffer *ssm_a_buf = weights_mgr_->get_tensor_buffer(prefix + "ssm_a");
    ClBuffer *ssm_dt_buf = weights_mgr_->get_tensor_buffer(prefix + "ssm_dt");

    cl_->qwen_deltanet_fused(gpu_ssm_state, gpu_conv_out, gpu_alpha, gpu_beta,
                            ssm_a_buf, ssm_dt_buf,
                            ssm_norm_buf, attn_gate_buf, gpu_delta_out,
                            key_dim, qk_dim, linear_inner, arch.norm_eps);

    if (w_O != model.tensors.end())
    {
        weights_mgr_->dispatch_gemv(gpu_attn_out, gpu_delta_out, prefix + "ssm_out.weight", w_O->second, n_embd, linear_inner, layer);
    }
    else
    {
        cl_->copy(gpu_attn_out, gpu_delta_out, std::min(n_embd, linear_inner));
    }

    // Recurrent branch output is now in gpu_attn_out, ready for fused add_rms_norm
}
