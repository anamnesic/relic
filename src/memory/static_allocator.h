#pragma once
#include <cstddef>
#include <vector>
#include <memory>
#include "../opencl_backend.h"

// Static Memory Planner for LLM Activations & Scratch Buffers
// Preallocates all intermediate buffers at initialization time to ensure
// zero allocations or reallocations in the token generation hot path.
class StaticMemoryPlanner {
public:
    struct ScratchLayout {
        size_t hidden_bytes = 0;
        size_t q_bytes = 0;
        size_t k_bytes = 0;
        size_t v_bytes = 0;
        size_t attn_out_bytes = 0;
        size_t ffn_gate_bytes = 0;
        size_t ffn_up_bytes = 0;
        size_t ffn_down_bytes = 0;
        size_t logits_bytes = 0;
        size_t total_scratch_bytes = 0;
    };

    static ScratchLayout compute_layout(int64_t n_embd, int64_t n_head, int64_t n_kv_head, int64_t n_ff, int64_t n_vocab, int64_t max_seq_len) {
        ScratchLayout l;
        int64_t head_dim = n_embd / n_head;
        l.hidden_bytes = (size_t)n_embd * sizeof(float);
        l.q_bytes = (size_t)(n_head * head_dim) * sizeof(float);
        l.k_bytes = (size_t)(n_kv_head * head_dim) * sizeof(float);
        l.v_bytes = (size_t)(n_kv_head * head_dim) * sizeof(float);
        l.attn_out_bytes = (size_t)n_embd * sizeof(float);
        l.ffn_gate_bytes = (size_t)n_ff * sizeof(float);
        l.ffn_up_bytes = (size_t)n_ff * sizeof(float);
        l.ffn_down_bytes = (size_t)n_embd * sizeof(float);
        l.logits_bytes = (size_t)n_vocab * sizeof(float);

        l.total_scratch_bytes = l.hidden_bytes * 4 + l.q_bytes + l.k_bytes + l.v_bytes
                              + l.attn_out_bytes + l.ffn_gate_bytes + l.ffn_up_bytes
                              + l.ffn_down_bytes + (size_t)(n_head * max_seq_len * sizeof(float));
        return l;
    }
};
