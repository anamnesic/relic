#include "qwen35_mlp_block.h"

Qwen35MlpBlock::Qwen35MlpBlock(OpenClBackend *backend, Qwen35WeightsManager *weights_mgr)
    : cl_(backend), weights_mgr_(weights_mgr)
{
}

void Qwen35MlpBlock::forward(int64_t layer, const ArchitectureSpec &arch, const LlamaModel &model,
                             ClBuffer &gpu_hidden, ClBuffer &gpu_residual, ClBuffer &gpu_attn_out,
                             ClBuffer &gpu_ffn_act, ClBuffer &gpu_gate, ClBuffer &gpu_up)
{
    int64_t n_embd = arch.n_embd;
    int64_t n_ff = arch.n_ff;
    std::string prefix = "blk." + std::to_string(layer) + ".";

    std::string ffn_norm_name = prefix + "ffn_norm.weight";
    ClBuffer *ffn_norm_buf = weights_mgr_->get_tensor_buffer(ffn_norm_name);
    if (!ffn_norm_buf)
    {
        ffn_norm_name = prefix + "post_attention_norm.weight";
        ffn_norm_buf = weights_mgr_->get_tensor_buffer(ffn_norm_name);
    }
    if (ffn_norm_buf)
    {
        cl_->add_rms_norm(gpu_residual, gpu_attn_out, *ffn_norm_buf, gpu_hidden, n_embd, arch.norm_eps);
    }
    else
    {
        cl_->add(gpu_residual, gpu_residual, gpu_attn_out, n_embd);
        cl_->copy(gpu_hidden, gpu_residual, n_embd);
    }

    auto w_gate = model.tensors.find(prefix + "ffn_gate.weight");
    auto w_up = model.tensors.find(prefix + "ffn_up.weight");
    if (w_gate != model.tensors.end() && w_up != model.tensors.end())
    {
        ClBuffer *gate_buf = weights_mgr_->get_tensor_buffer(prefix + "ffn_gate.weight");
        ClBuffer *up_buf = weights_mgr_->get_tensor_buffer(prefix + "ffn_up.weight");
        GgmlType gate_type = weights_mgr_->get_tensor_type(prefix + "ffn_gate.weight", w_gate->second.type);
        GgmlType up_type = weights_mgr_->get_tensor_type(prefix + "ffn_up.weight", w_up->second.type);
        if (gate_buf && up_buf && gate_type == GgmlType::Q4_0 && up_type == GgmlType::Q4_0)
        {
            cl_->gemv_q4_0_ffn_swiglu(gpu_ffn_act, gpu_hidden, *gate_buf, *up_buf, n_ff, n_embd);
        }
        else
        {
            weights_mgr_->dispatch_gemv(gpu_gate, gpu_hidden, prefix + "ffn_gate.weight", w_gate->second, n_ff, n_embd, layer);
            weights_mgr_->dispatch_gemv(gpu_up, gpu_hidden, prefix + "ffn_up.weight", w_up->second, n_ff, n_embd, layer);
            cl_->swiglu(gpu_ffn_act, gpu_gate, gpu_up, n_ff);
        }

        auto w_down = model.tensors.find(prefix + "ffn_down.weight");
        if (w_down != model.tensors.end())
        {
            weights_mgr_->dispatch_gemv(gpu_attn_out, gpu_ffn_act, prefix + "ffn_down.weight", w_down->second, n_embd, n_ff, layer);
        }
    }
}
