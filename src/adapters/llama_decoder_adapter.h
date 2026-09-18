#pragma once

#include "decoder.h"
#include <vector>

class LlamaDecoderAdapter final : public ArchitectureDecoder
{
public:
    explicit LlamaDecoderAdapter(OpenClBackend *backend);
    ~LlamaDecoderAdapter() override = default;

    bool init(const ArchitectureSpec &spec, int64_t max_seq_len, const ExecutionPlan *plan = nullptr) override;
    void reset() override;
    void save_state_checkpoint() override;
    void restore_state_checkpoint() override;
    int forward(const LlamaModel &model, int token_id, int64_t position, float *logits, bool compute_output = true) override;

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
