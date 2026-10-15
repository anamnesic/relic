#pragma once
#include "gguf_reader.h"
#include <vector>
#include <string>
#include <cstring>
#include <cmath>

struct LlamaModel {
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
};

// Half-precision conversion utilities
inline uint16_t float_to_half_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    uint16_t sign = (u >> 16) & 0x8000;
    int32_t exp = ((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = u & 0x007FFFFF;
    if (exp <= 0) {
        mant = (mant | 0x00800000) >> (1 - exp);
        return (uint16_t)(sign | (mant >> 13));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

inline float half_bits_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exp = ((h >> 10) & 0x1F) - 15 + 127;
    uint32_t mant = h & 0x03FF;
    if (exp <= 0) {
        mant = (mant | 0x0400) >> (1 - exp);
        exp = 0;
    }
    if (exp >= 255) {
        uint32_t u = sign | 0x7F800000 | (mant << 13);
        float f;
        memcpy(&f, &u, 4);
        return f;
    }
    uint32_t u = sign | (exp << 23) | (mant << 13);
    float f;
    memcpy(&f, &u, 4);
    return f;
}
