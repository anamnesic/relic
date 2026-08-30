#pragma once
#include "architecture.h"
#include "decoder.h"
#include "model.h"
#include "opencl_backend.h"
#include "tokenizer.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct InferenceEngine {
    LlamaModel *model = nullptr;
    Tokenizer *tokenizer = nullptr;
    OpenClBackend *cl = nullptr;
    std::unique_ptr<ArchitectureDecoder> decoder;
    int64_t max_seq_len = 2048;
    int64_t n_past = 0;

    bool enable_speculative = false;
    int speculative_ngram = 3;
    int speculative_max_draft = 3;

    bool init(LlamaModel *m, Tokenizer *tok, OpenClBackend *backend,
              int64_t max_seq_len = 2048);
    void free_buffers();

    // Forward pass for one token
    int forward(int token_id, float *logits);

    // Generate text
    std::string generate(const std::string &prompt, int max_tokens = 256,
                         float temperature = 0.8f, int top_k = 40);

private:
    std::vector<int> find_prompt_lookup_draft(const std::vector<int> &tokens, int ngram_len, int max_draft);
};
