#pragma once

#include "architecture.h"
#include "model.h"
#include <memory>
#include <string>
#include <vector>

class OpenClBackend;

// Hexagonal Port: ArchitectureDecoder
// Abstract port defining the execution contract for any model architecture decoder.
class ArchitectureDecoder {
public:
    virtual ~ArchitectureDecoder() = default;

    // Initializes internal buffers, caches, and recurrent states.
    virtual bool init(const ArchitectureSpec &spec, int64_t max_seq_len) = 0;

    // Resets sequence position, KV caches, and recurrent history.
    virtual void reset() = 0;

    // Executes a forward pass for a single token at the specified sequence position.
    virtual int forward(const LlamaModel &model, int token_id, int64_t position, float *logits) = 0;
};

// Factory function to create the appropriate decoder adapter for the architecture.
std::unique_ptr<ArchitectureDecoder> create_decoder(const ArchitectureSpec &spec, OpenClBackend *backend);
