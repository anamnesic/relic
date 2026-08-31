#include "distributed_speculative.h"
#include <chrono>
#include <algorithm>

DistributedSpeculativeEngine::DistributedSpeculativeEngine(
    std::unique_ptr<ArchitectureDecoder> target_decoder,
    std::unique_ptr<ArchitectureDecoder> draft_decoder,
    std::shared_ptr<LlamaModel> target_model,
    std::shared_ptr<LlamaModel> draft_model,
    std::shared_ptr<Tokenizer> tokenizer
) : target_decoder_(std::move(target_decoder)),
    draft_decoder_(std::move(draft_decoder)),
    target_model_(std::move(target_model)),
    draft_model_(std::move(draft_model)),
    tokenizer_(std::move(tokenizer)) {}

SpeculativeResult DistributedSpeculativeEngine::step_speculation(const std::vector<int> &context_tokens, int max_draft) {
    SpeculativeResult res;
    res.num_drafted = 0;
    res.num_accepted = 0;
    res.draft_time_ms = 0.0;
    res.verification_time_ms = 0.0;

    if (context_tokens.empty() || !target_decoder_ || !target_model_) return res;

    // 1. Draft Phase: Generate K draft tokens
    auto t_draft_start = std::chrono::high_resolution_clock::now();
    std::vector<int> draft_tokens;
    std::vector<float> draft_logits((size_t)target_model_->n_vocab, 0.0f);

    int cur_token = context_tokens.back();
    for (int d = 0; d < max_draft; d++) {
        if (draft_decoder_ && draft_model_) {
            draft_decoder_->forward(*draft_model_, cur_token, (int64_t)(context_tokens.size() + d - 1), draft_logits.data());
            cur_token = (int)(std::max_element(draft_logits.begin(), draft_logits.end()) - draft_logits.begin());
            draft_tokens.push_back(cur_token);
            if (tokenizer_ && cur_token == tokenizer_->eos_id) break;
        } else {
            break;
        }
    }
    auto t_draft_end = std::chrono::high_resolution_clock::now();
    res.draft_time_ms = std::chrono::duration<double, std::milli>(t_draft_end - t_draft_start).count();
    res.num_drafted = (int)draft_tokens.size();

    // 2. Verification Phase on Target Device (GTX 1650)
    auto t_verify_start = std::chrono::high_resolution_clock::now();
    std::vector<float> target_logits((size_t)target_model_->n_vocab, 0.0f);

    for (int draft_tok : draft_tokens) {
        target_decoder_->forward(*target_model_, draft_tok, (int64_t)(context_tokens.size() + res.num_accepted), target_logits.data());
        int target_pred = (int)(std::max_element(target_logits.begin(), target_logits.end()) - target_logits.begin());

        if (target_pred == draft_tok) {
            res.accepted_tokens.push_back(draft_tok);
            res.num_accepted++;
            if (tokenizer_ && draft_tok == tokenizer_->eos_id) break;
        } else {
            // Rejection: append target's correction token and break
            res.accepted_tokens.push_back(target_pred);
            res.num_accepted++;
            break;
        }
    }
    auto t_verify_end = std::chrono::high_resolution_clock::now();
    res.verification_time_ms = std::chrono::duration<double, std::milli>(t_verify_end - t_verify_start).count();

    return res;
}
