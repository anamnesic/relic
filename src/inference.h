#pragma once
#include "architecture.h"
#include "decoder.h"
#include "model.h"
#include "tokenizer.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ExecutionPlan;
class OpenClBackend;

struct InferenceEngine
{
    NeuralModel *model = nullptr;
    Tokenizer *tokenizer = nullptr;
    const ExecutionPlan *plan = nullptr;
    std::unique_ptr<ArchitectureDecoder> decoder;
    int64_t max_seq_len = 2048;
    int64_t n_past = 0;

    bool enable_speculative = false;
    int speculative_ngram = 3;
    int speculative_max_draft = 3;

    // Pure Hexagonal Port injection: inject concrete decoder adapter directly
    bool init(NeuralModel *m, Tokenizer *tok, std::unique_ptr<ArchitectureDecoder> dec,
              int64_t max_seq_len = 2048, const ExecutionPlan *plan = nullptr);

    // Convenience factory overload
    bool init(NeuralModel *m, Tokenizer *tok, OpenClBackend *backend,
              int64_t max_seq_len = 2048, const ExecutionPlan *plan = nullptr);
    void free_buffers();

    // Forward pass for one token
    int forward(int token_id, float *logits = nullptr, bool compute_output = true);

    // Generate text
    std::string generate(const std::string &prompt, int max_tokens = 256,
                         float temperature = 0.8f, int top_k = 40);

private:
    std::vector<int> find_prompt_lookup_draft(const std::vector<int> &tokens, int ngram_len, int max_draft);
};
