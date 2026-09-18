#include "model.h"
#include <cstdio>
#include <cassert>

bool LlamaModel::load(const char *filename) {
    GgufReader reader;
    if (!reader.load(filename)) {
        fprintf(stderr, "Failed to open GGUF file: %s\n", filename);
        return false;
    }

    const std::string architecture_name = reader.get_metadata<std::string>("general.architecture", "");
    const ModelArchitecture *adapter = find_architecture(architecture_name);
    std::string architecture_error;
    if (!adapter || !adapter->configure(reader, architecture, architecture_error)) {
        fprintf(stderr, "Unsupported GGUF architecture: %s%s%s\n",
                architecture_name.empty() ? "unknown" : architecture_name.c_str(),
                architecture_error.empty() ? "" : " (",
                architecture_error.empty() ? "" : (architecture_error + ")").c_str());
        return false;
    }

    n_vocab = architecture.n_vocab;
    n_embd = architecture.n_embd;
    n_mult = reader.get_metadata<int64_t>("llama.feed_forward_length",
              reader.get_metadata<int64_t>("llama.n_mult", 256));
    n_head = architecture.n_head;
    n_head_kv = architecture.n_head_kv;
    n_layer = architecture.n_layer;
    n_ff = architecture.n_ff;
    norm_eps = architecture.norm_eps;
    n_expert = reader.get_metadata<int64_t>("llama.expert_count", 0);
    n_expert_used = reader.get_metadata<int64_t>("llama.expert_used_count", 0);
    rope_freq_base = architecture.rope_freq_base;
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

void LlamaModel::dequantize_to_f32(const Tensor &t, float *out) const {
    int64_t row_size = t.dims.empty() ? t.nelements() : t.dims[0];
    int64_t total_rows = t.dims.size() > 1 ? t.dims[1] : (row_size > 0 ? t.nelements() / row_size : 1);
    relic::quant::dequantize_rows(t.type, t.data.data(), row_size, 0, total_rows, out);
}

void LlamaModel::dequantize_rows_to_f32(const Tensor &t, int64_t start_row, int64_t num_rows, float *out) const {
    int64_t row_size = t.dims.empty() ? t.nelements() : t.dims[0];
    int64_t total_rows = t.dims.size() > 1 ? t.dims[1] : (row_size > 0 ? t.nelements() / row_size : 1);
    num_rows = std::min(num_rows, total_rows - start_row);
    relic::quant::dequantize_rows(t.type, t.data.data(), row_size, start_row, num_rows, out);
}
