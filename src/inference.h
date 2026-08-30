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

    bool init(LlamaModel *m, Tokenizer *tok, OpenClBackend *backend,
              int64_t max_seq_len = 2048);
    void free_buffers();

    // Forward pass for one token
    int forward(int token_id, float *logits);

    // Generate text
    std::string generate(const std::string &prompt, int max_tokens = 256,
                         float temperature = 0.8f, int top_k = 40);
};
