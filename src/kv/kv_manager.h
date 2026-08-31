#pragma once
#include <cstdint>
#include <memory>
#include "../backends/backend.h"

enum class KvQuantType {
    FP16,
    Q8_0,
    Q4_0
};

class KvManager {
public:
    KvManager(Backend &backend, int64_t n_layer, int64_t n_head_kv, int64_t head_dim, int64_t max_seq_len);
    ~KvManager();

    bool allocate_cache(KvQuantType quant_type);
    void compress_under_pressure(KvQuantType target_type);
    void reset();

    BackendBuffer* get_k_cache(int64_t layer);
    BackendBuffer* get_v_cache(int64_t layer);

private:
    Backend &backend_;
    int64_t n_layer_;
    int64_t n_head_kv_;
    int64_t head_dim_;
    int64_t max_seq_len_;
    KvQuantType current_quant_;
    std::vector<std::unique_ptr<BackendBuffer>> k_caches_;
    std::vector<std::unique_ptr<BackendBuffer>> v_caches_;
};
