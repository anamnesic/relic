#pragma once

#include "architecture.h"
#include <cstdint>
#include <vector>

// CPU reference state for Qwen3.5's recurrent Gated DeltaNet layers.
// Values are stored per model layer so multiple sequences can advance token by token.
class Qwen35RecurrentState {
public:
    bool init(const ArchitectureSpec &spec);
    void reset();

    // Applies causal depthwise convolution and retains the preceding K - 1 values.
    void conv1d(int64_t layer, const float *input, const float *kernel, float *output);

    // Applies one Gated DeltaNet update. q and k are expanded to value-head count.
    void delta_step(int64_t layer, const float *q, const float *k, const float *v,
                    const float *decay, const float *beta, float *output);

    int64_t channels() const { return conv_channels; }

private:
    int64_t n_layers = 0;
    int64_t conv_kernel = 0;
    int64_t conv_channels = 0;
    int64_t key_dim = 0;
    int64_t value_dim = 0;
    int64_t value_heads = 0;
    std::vector<float> conv_history;
    std::vector<float> delta_state;
};
