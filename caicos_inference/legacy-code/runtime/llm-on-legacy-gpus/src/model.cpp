#include "model.h"
#include <cstdio>
#include <cassert>

bool LlamaModel::load(const char *filename) {
    GgufReader reader;
    if (!reader.load(filename)) {
        fprintf(stderr, "Failed to open GGUF file: %s\n", filename);
        return false;
    }

    n_vocab = (int64_t)reader.get_metadata("llama.vocab_size",
               (int64_t)reader.get_metadata("tokenizer.ggml.vocab_size", 32000));
    n_embd = reader.get_metadata<int64_t>("llama.embedding_length",
             reader.get_metadata<int64_t>("llama.d_model",
             reader.get_metadata<int64_t>("llama.n_embd", 4096)));
    n_mult = reader.get_metadata<int64_t>("llama.feed_forward_length",
             reader.get_metadata<int64_t>("llama.n_mult", 256));
    n_head = reader.get_metadata<int64_t>("llama.attention.head_count",
             reader.get_metadata<int64_t>("llama.n_head", 32));
    n_head_kv = reader.get_metadata<int64_t>("llama.attention.head_count_kv",
                reader.get_metadata<int64_t>("llama.n_head_kv", n_head));
    n_layer = reader.get_metadata<int64_t>("llama.block_count",
              reader.get_metadata<int64_t>("llama.n_layer", 32));
    n_ff = reader.get_metadata<int64_t>("llama.feed_forward_length",
           reader.get_metadata<int64_t>("llama.n_ff", 4 * n_embd));
    norm_eps = reader.get_metadata<float>("llama.attention.layer_norm_rms_epsilon",
               reader.get_metadata<float>("llama.norm_eps", 1e-5f));
    n_expert = reader.get_metadata<int64_t>("llama.expert_count", 0);
    n_expert_used = reader.get_metadata<int64_t>("llama.expert_used_count", 0);
    rope_freq_base = reader.get_metadata<float>("llama.rope.freq_base", 10000.0f);
    rope_freq_scale = reader.get_metadata<float>("llama.rope.freq_scale", 1.0f);

    n_embd_head_k = n_embd / n_head;
    n_embd_head_v = n_embd_head_k;

    fprintf(stdout, "Model: vocab=%lld embd=%lld head=%lld layers=%lld ff=%lld\n",
            (long long)n_vocab, (long long)n_embd, (long long)n_head,
            (long long)n_layer, (long long)n_ff);
    fprintf(stdout, "  n_head_kv=%lld n_embd_head=%lld\n",
            (long long)n_head_kv, (long long)n_embd_head_k);

    // Read all tensors
    for (auto &[name, info] : reader.tensors) {
        Tensor t;
        t.name = name;
        t.type = info.type;
        t.dims = info.dims;

        const void *src = reader.tensor_data(name);
        if (!src) {
            fprintf(stderr, "Warning: tensor %s has no data\n", name.c_str());
            continue;
        }

        int blck_size = ggml_blck_size(t.type);
        int type_size = ggml_type_size(t.type);
        int64_t n_blocks = (t.nelements() + blck_size - 1) / blck_size;
        size_t nbytes = (size_t)n_blocks * type_size;

        t.data.resize(nbytes);
        memcpy(t.data.data(), src, nbytes);
        tensors[name] = std::move(t);
    }

    fprintf(stdout, "Loaded %zu tensors\n", tensors.size());
    return true;
}

static void dequantize_q4_0_row(const uint8_t *block, float *out, int64_t n) {
    int64_t num_blocks = (n + 31) / 32;
    for (int64_t b = 0; b < num_blocks; b++) {
        uint16_t d_bits;
        memcpy(&d_bits, block + b * 18, 2);
        float d = half_bits_to_float(d_bits);
        const uint8_t *qs = block + b * 18 + 2;
        for (int i = 0; i < 16 && b * 32 + i * 2 < n; i++) {
            uint8_t q = qs[i];
            float q0 = (float)(int8_t)((q & 0x0F) << 4) * 0.0625f;
            float q1 = (float)(int8_t)((q & 0xF0)) * 0.0625f;
            out[b * 32 + i * 2] = q0 * d;
            if (b * 32 + i * 2 + 1 < n)
                out[b * 32 + i * 2 + 1] = q1 * d;
        }
    }
}

static void dequantize_q8_0_row(const uint8_t *block, float *out, int64_t n) {
    int64_t num_blocks = (n + 31) / 32;
    for (int64_t b = 0; b < num_blocks; b++) {
        uint16_t d_bits;
        memcpy(&d_bits, block + b * 34, 2);
        float d = half_bits_to_float(d_bits);
        const int8_t *qs = (const int8_t *)(block + b * 34 + 2);
        for (int i = 0; i < 32 && b * 32 + i < n; i++) {
            out[b * 32 + i] = (float)qs[i] * d;
        }
    }
}

static void dequantize_f16_row(const uint8_t *block, float *out, int64_t n) {
    const uint16_t *f16 = (const uint16_t *)block;
    for (int64_t i = 0; i < n; i++) {
        out[i] = half_bits_to_float(f16[i]);
    }
}

void LlamaModel::dequantize_to_f32(const Tensor &t, float *out) const {
    int64_t n = t.nelements();

    switch (t.type) {
        case GgmlType::F32:
            memcpy(out, t.data.data(), n * 4);
            break;
        case GgmlType::F16:
            dequantize_f16_row(t.data.data(), out, n);
            break;
        case GgmlType::Q4_0:
            for (int64_t row = 0; row < n; row += t.dims[0]) {
                int64_t row_size = t.dims[0];
                const uint8_t *src_row = t.data.data() + (row / t.dims[0]) * ((row_size + 31) / 32) * 18;
                dequantize_q4_0_row(src_row, out + row, row_size);
            }
            break;
        case GgmlType::Q8_0:
            for (int64_t row = 0; row < n; row += t.dims[0]) {
                int64_t row_size = t.dims[0];
                const uint8_t *src_row = t.data.data() + (row / t.dims[0]) * ((row_size + 31) / 32) * 34;
                dequantize_q8_0_row(src_row, out + row, row_size);
            }
            break;
        default:
            fprintf(stderr, "Unsupported type for dequantization: %d\n", (int)t.type);
            memset(out, 0, n * 4);
            break;
    }
}
