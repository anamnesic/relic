#pragma once
#include "architecture.h"
#include <vector>
#include <string>
#include <cstring>
#include <cmath>

struct LlamaModel {
    ArchitectureSpec architecture;
    int64_t n_vocab = 32000;
    int64_t n_embd = 4096;
    int64_t n_mult = 256;
    int64_t n_head = 32;
    int64_t n_head_kv = 32;
    int64_t n_layer = 32;
    int64_t n_ff = 11008;
    float norm_eps = 1e-5f;
    int64_t n_embd_head_k = 128;
    int64_t n_embd_head_v = 128;
    int64_t n_expert = 0;
    int64_t n_expert_used = 0;
    float rope_freq_base = 10000.0f;
    float rope_freq_scale = 1.0f;
    std::string rope_scaling_type;

    struct Tensor {
        std::string name;
        GgmlType type;
        std::vector<int64_t> dims;
        std::vector<uint8_t> data;

        int64_t nelements() const {
            int64_t n = 1;
            for (auto d : dims) n *= d;
            return n;
        }
    };

    using TensorMap = std::unordered_map<std::string, Tensor>;

    TensorMap tensors;

    bool load(const char *filename);
    int get_type_size(const Tensor &t) const;
    size_t get_quantized_blocks(const Tensor &t) const;

    // Dequantize a tensor to f32
    void dequantize_to_f32(const Tensor &t, float *out) const;
    void dequantize_rows_to_f32(const Tensor &t, int64_t start_row, int64_t num_rows, float *out) const;
};

#include "quantization/quant_utils.h"

// Ubiquitous domain language alias
using NeuralModel = LlamaModel;

// Half-precision conversion utilities forwarded from relic::quant
inline uint16_t float_to_half_bits(float f) {
    return relic::quant::float_to_half_bits(f);
}

inline float half_bits_to_float(uint16_t h) {
    return relic::quant::half_bits_to_float(h);
}
