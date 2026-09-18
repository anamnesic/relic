#pragma once

#include "decoder.h"
#include "opencl_backend.h"
#include "qwen35_state.h"
#include "qwen35_weights_manager.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class Qwen35DecoderAdapter final : public ArchitectureDecoder
{
public:
    explicit Qwen35DecoderAdapter(OpenClBackend *backend);
    ~Qwen35DecoderAdapter() override = default;

    bool init(const ArchitectureSpec &spec, int64_t max_seq_len, const ExecutionPlan *plan = nullptr) override;
    void reset() override;
    void save_state_checkpoint() override;
    void restore_state_checkpoint() override;
    void warm_up(const LlamaModel &model) override;
    int forward(const LlamaModel &model, int token_id, int64_t position, float *logits, bool compute_output = true) override;
    int sample_token(float temperature = 0.0f, int top_k = 40, float top_p = 0.9f) override;
    bool supports_device_sampling() const override { return use_gpu; }

private:
    void ensure_weights_uploaded(const LlamaModel &model);
    void dispatch_gemv(ClBuffer &dst, ClBuffer &in, const std::string &name, const LlamaModel::Tensor &t, int64_t N, int64_t K, int64_t layer_idx = 0);
    void model_dequant_rows(const LlamaModel::Tensor &t, int64_t start_row, int64_t num_rows, float *out);
    float float_to_half_val(uint16_t h);

    ArchitectureSpec arch;
    OpenClBackend *cl = nullptr;
    const ExecutionPlan *plan_ = nullptr;
    bool use_gpu = false;
    int64_t seq_limit = 2048;
    Qwen35RecurrentState recurrent_state;
    std::vector<float> act;
    std::vector<float> weights;
    std::vector<float> full_attn_kv;

    std::unique_ptr<Qwen35WeightsManager> weights_mgr;

    ClBuffer gpu_hidden;
    ClBuffer gpu_residual;
    ClBuffer gpu_attn_out;
    ClBuffer gpu_gate;
    ClBuffer gpu_up;
    ClBuffer gpu_ffn_act;
    ClBuffer gpu_conv_in;
    ClBuffer gpu_conv_out;
    ClBuffer gpu_delta_out;
    ClBuffer gpu_q_full;
    ClBuffer gpu_q;
    ClBuffer gpu_attn_gate;
    ClBuffer gpu_k;
    ClBuffer gpu_v;
    ClBuffer gpu_alpha;
    ClBuffer gpu_beta;
    ClBuffer gpu_logits;
    ClBuffer gpu_norm_w;
    ClBuffer gpu_act_a;

    std::vector<ClBuffer> gpu_ssm_states;
    std::vector<ClBuffer> gpu_conv_states;
    std::vector<ClBuffer> gpu_k_caches;
    std::vector<ClBuffer> gpu_v_caches;

    std::vector<ClBuffer> gpu_ssm_snapshot_;
    std::vector<ClBuffer> gpu_conv_snapshot_;
    std::vector<ClBuffer> gpu_k_snapshot_;
    std::vector<ClBuffer> gpu_v_snapshot_;

    Qwen35RecurrentState::Snapshot snapshot_recurrent_;
    std::vector<float> snapshot_full_attn_kv_;
    std::vector<float> snapshot_act_;
};
