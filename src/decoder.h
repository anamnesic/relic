#pragma once

#include "architecture.h"
#include "model.h"
#include <memory>
#include <string>
#include <vector>

class OpenClBackend;
struct ExecutionPlan;

// Hexagonal Port: ArchitectureDecoder
// Abstract port defining the execution contract for any model architecture decoder.
class ArchitectureDecoder
{
public:
    virtual ~ArchitectureDecoder() = default;

    // Initializes internal buffers, caches, and recurrent states according to the execution plan.
    virtual bool init(const ArchitectureSpec &spec, int64_t max_seq_len, const ExecutionPlan *plan = nullptr) = 0;

    // Resets sequence position, KV caches, and recurrent history.
    virtual void reset() = 0;

    // Executes a forward pass for a single token at the specified sequence position.
    virtual int forward(const LlamaModel &model, int token_id, int64_t position, float *logits) = 0;

    // Executes a batched forward pass for multiple tokens starting at start_position.
    // logits_out must have space for token_ids.size() * model.n_vocab floats.
    virtual int forward_batch(const LlamaModel &model, const std::vector<int> &token_ids, int64_t start_position, float *logits_out)
    {
        int64_t n_vocab = model.n_vocab;
        for (size_t i = 0; i < token_ids.size(); i++)
        {
            float *cur_logits = logits_out ? (logits_out + i * (size_t)n_vocab) : nullptr;
            int res = forward(model, token_ids[i], start_position + (int64_t)i, cur_logits);
            if (res != 0)
                return res;
        }
        return 0;
    }

    // State checkpointing for non-destructive speculative verification
    virtual void save_state_checkpoint() {}
    virtual void restore_state_checkpoint() {}

    // Speculative batch verification pipeline with automatic rollback of rejected candidate state
    virtual int forward_speculative_batch(const LlamaModel &model, const std::vector<int> &draft_tokens, int64_t start_position, float *logits_out, int accepted_count)
    {
        save_state_checkpoint();
        int64_t n_vocab = model.n_vocab;
        for (size_t i = 0; i < draft_tokens.size(); i++)
        {
            float *cur_logits = logits_out ? (logits_out + i * (size_t)n_vocab) : nullptr;
            int res = forward(model, draft_tokens[i], start_position + (int64_t)i, cur_logits);
            if (res != 0)
            {
                restore_state_checkpoint();
                return res;
            }
        }
        // Rollback state if fewer than all draft tokens were accepted, and advance only the accepted tokens
        if (accepted_count < (int)draft_tokens.size())
        {
            restore_state_checkpoint();
            for (int i = 0; i < accepted_count; i++)
            {
                forward(model, draft_tokens[i], start_position + (int64_t)i, nullptr);
            }
        }
        return 0;
    }

    // Optional pre-computation / GPU VRAM weight upload ahead of generation timing.
    virtual void warm_up(const LlamaModel &model) {}
};

// Factory function to create the appropriate decoder adapter for the architecture.
std::unique_ptr<ArchitectureDecoder> create_decoder(const ArchitectureSpec &spec, OpenClBackend *backend);
