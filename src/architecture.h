#pragma once

#include "gguf_reader.h"
#include <cstdint>
#include <string>

enum class ArchitectureKind {
    Llama,
    Qwen35,
    Unsupported,
};

struct ArchitectureSpec {
    ArchitectureKind kind = ArchitectureKind::Unsupported;
    std::string name;
    int64_t n_vocab = 0;
    int64_t n_embd = 0;
    int64_t n_layer = 0;
    int64_t n_head = 0;
    int64_t n_head_kv = 0;
    int64_t n_ff = 0;
    float norm_eps = 1e-5f;
    float rope_freq_base = 10000.0f;

    // Qwen3.5's three recurrent Gated DeltaNet layers per full-attention layer.
    int64_t full_attention_interval = 0;
    int64_t linear_conv_kernel = 0;
    int64_t linear_key_head_dim = 0;
    int64_t linear_value_head_dim = 0;
    int64_t linear_key_heads = 0;
    int64_t linear_value_heads = 0;
};

// The core only depends on this port; each architecture adapter owns its GGUF
// validation and metadata mapping.
class ModelArchitecture {
public:
    virtual ~ModelArchitecture() = default;
    virtual bool configure(const GgufReader &reader, ArchitectureSpec &spec, std::string &error) const = 0;
};

const ModelArchitecture *find_architecture(const std::string &name);
