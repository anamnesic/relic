#pragma once
#include <vector>
#include <string>
#include <memory>
#include "../model.h"
#include "../tokenizer.h"
#include "../decoder.h"

struct SpeculativeResult {
    std::vector<int> accepted_tokens;
    int num_drafted;
    int num_accepted;
    double draft_time_ms;
    double verification_time_ms;
};

class DistributedSpeculativeEngine {
public:
    DistributedSpeculativeEngine(
        std::unique_ptr<ArchitectureDecoder> target_decoder,
        std::unique_ptr<ArchitectureDecoder> draft_decoder,
        std::shared_ptr<LlamaModel> target_model,
        std::shared_ptr<LlamaModel> draft_model,
        std::shared_ptr<Tokenizer> tokenizer
    );
    ~DistributedSpeculativeEngine() = default;

    // Generates K draft tokens on draft device (Intel UHD / CPU) and verifies them in a single batched pass on Target (GTX 1650)
    SpeculativeResult step_speculation(const std::vector<int> &context_tokens, int max_draft = 4);

private:
    std::unique_ptr<ArchitectureDecoder> target_decoder_;
    std::unique_ptr<ArchitectureDecoder> draft_decoder_;
    std::shared_ptr<LlamaModel> target_model_;
    std::shared_ptr<LlamaModel> draft_model_;
    std::shared_ptr<Tokenizer> tokenizer_;
};
