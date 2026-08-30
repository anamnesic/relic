#include "qwen35_state.h"

#include <algorithm>
#include <cmath>

bool Qwen35RecurrentState::init(const ArchitectureSpec &spec) {
    n_layers = spec.n_layer;
    conv_kernel = spec.linear_conv_kernel;
    key_dim = spec.linear_key_head_dim;
    value_heads = spec.linear_value_heads;
    if (value_heads <= 0 || spec.linear_inner_size <= 0 || spec.linear_inner_size % value_heads != 0 ||
        key_dim <= 0 || conv_kernel <= 0 || n_layers <= 0) return false;

    value_dim = spec.linear_inner_size / value_heads;
    conv_channels = key_dim * spec.linear_key_heads * 2 + spec.linear_inner_size;
    conv_history.assign((size_t)(n_layers * conv_channels * (conv_kernel - 1)), 0.0f);
    delta_state.assign((size_t)(n_layers * value_heads * value_dim * key_dim), 0.0f);
    return true;
}

void Qwen35RecurrentState::reset() {
    std::fill(conv_history.begin(), conv_history.end(), 0.0f);
    std::fill(delta_state.begin(), delta_state.end(), 0.0f);
}

void Qwen35RecurrentState::conv1d(int64_t layer, const float *input, const float *kernel, float *output) {
    const int64_t history_size = conv_kernel - 1;
    float *history = conv_history.data() + layer * conv_channels * history_size;
    for (int64_t channel = 0; channel < conv_channels; ++channel) {
        const float *weights = kernel + channel * conv_kernel;
        float *past = history + channel * history_size;
        float sum = weights[history_size] * input[channel];
        for (int64_t i = 0; i < history_size; ++i) sum += weights[i] * past[i];
        output[channel] = sum;
        for (int64_t i = 0; i + 1 < history_size; ++i) past[i] = past[i + 1];
        if (history_size > 0) past[history_size - 1] = input[channel];
    }
}

void Qwen35RecurrentState::delta_step(int64_t layer, const float *q, const float *k, const float *v,
                                      const float *decay, const float *beta, float *output) {
    float *state = delta_state.data() + layer * value_heads * value_dim * key_dim;
    for (int64_t head = 0; head < value_heads; ++head) {
        const float *q_head = q + head * key_dim;
        const float *k_head = k + head * key_dim;
        const float *v_head = v + head * value_dim;
        float *head_state = state + head * value_dim * key_dim;
        const float a = expf(decay[head]);
        const float b = beta[head];

        for (int64_t row = 0; row < value_dim; ++row) {
            float *state_row = head_state + row * key_dim;
            float prediction = 0.0f;
            for (int64_t col = 0; col < key_dim; ++col) {
                state_row[col] *= a;
                prediction += state_row[col] * k_head[col];
            }
            for (int64_t col = 0; col < key_dim; ++col) {
                state_row[col] += b * (v_head[row] - prediction) * k_head[col];
            }

            float result = 0.0f;
            for (int64_t col = 0; col < key_dim; ++col) result += state_row[col] * q_head[col];
            output[head * value_dim + row] = result;
        }
    }
}
