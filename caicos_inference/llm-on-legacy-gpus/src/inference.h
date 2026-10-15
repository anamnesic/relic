#pragma once
#include "model.h"
#include "opencl_backend.h"
#include "tokenizer.h"
#include <vector>
#include <cstdint>

struct InferenceState {
    // KV cache: [n_layer, 2, max_seq_len, n_embd]
    std::vector<float> kv_cache;
    int64_t max_seq_len = 2048;
    int64_t n_past = 0;

    // GPU buffers (reusable across layers)
    ClBuffer gpu_act;      // activations
    ClBuffer gpu_weights;  // current layer weights
    ClBuffer gpu_k;        // key cache slice
    ClBuffer gpu_v;        // value cache slice
    ClBuffer gpu_score;    // attention scores
    ClBuffer gpu_ctx;      // attention output

    bool use_gpu = false;
    int64_t n_embd_state = 0;
};

struct InferenceEngine {
    LlamaModel *model = nullptr;
    Tokenizer *tokenizer = nullptr;
    OpenClBackend *cl = nullptr;
    InferenceState state;

    // CPU-side buffers for streaming
    std::vector<float> act;      // reusable activation buffer
    std::vector<float> weights;  // reusable weight buffer (dequantized)

    bool init(LlamaModel *m, Tokenizer *tok, OpenClBackend *backend,
              int64_t max_seq_len = 2048);
    void free_buffers();

    // Forward pass for one token
    int forward(int token_id, float *logits);

    // Generate text
    std::string generate(const std::string &prompt, int max_tokens = 256,
                         float temperature = 0.8f, int top_k = 40);

private:
    // Layer operations
    void rms_norm_cpu(float *out, const float *x, const float *weight, int64_t n, int64_t rows);
    void silu_cpu(float *out, const float *x, int64_t n);
    void matmul_cpu(float *dst, const float *a, const float *b, int64_t M, int64_t N, int64_t K);
    void matmul_nt_cpu(float *dst, const float *a, const float *b, int64_t M, int64_t N, int64_t K);
    void rope_cpu(float *x, int64_t n_embd, int64_t n_head, int64_t pos, int64_t n_tokens);
    void softmax_cpu(float *x, int64_t n, int64_t rows);
    void add_cpu(float *dst, const float *a, const float *b, int64_t n);

    // Core attention + FFN for one layer (uses CPU or GPU)
    void forward_layer(int layer, int token_id, float *hidden,
                       float *k_slice, float *v_slice, float *scores,
                       float *attn_out, float *residual,
                       float *q_buf, float *k_buf, float *v_buf,
                       float *gate_buf, float *up_buf);
};
