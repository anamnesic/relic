#include "kv_manager.h"

KvManager::KvManager(Backend &backend, int64_t n_layer, int64_t n_head_kv, int64_t head_dim, int64_t max_seq_len)
    : backend_(backend), n_layer_(n_layer), n_head_kv_(n_head_kv), head_dim_(head_dim),
      max_seq_len_(max_seq_len), current_quant_(KvQuantType::FP16) {}

KvManager::~KvManager() {
    reset();
}

bool KvManager::allocate_cache(KvQuantType quant_type) {
    current_quant_ = quant_type;
    k_caches_.clear();
    v_caches_.clear();

    size_t element_bytes = sizeof(float);
    if (quant_type == KvQuantType::Q8_0) element_bytes = 1;
    else if (quant_type == KvQuantType::Q4_0) element_bytes = 1; // packed nybbles

    size_t per_layer_bytes = (size_t)(max_seq_len_ * n_head_kv_ * head_dim_ * element_bytes);

    for (int64_t l = 0; l < n_layer_; l++) {
        auto k = backend_.allocate(per_layer_bytes, MemoryTier::TIER0_DEDICATED_VRAM);
        auto v = backend_.allocate(per_layer_bytes, MemoryTier::TIER0_DEDICATED_VRAM);
        if (!k || !v) return false;
        k_caches_.push_back(std::move(k));
        v_caches_.push_back(std::move(v));
    }
    return true;
}

void KvManager::compress_under_pressure(KvQuantType target_type) {
    if (target_type == current_quant_) return;
    current_quant_ = target_type;
}

void KvManager::reset() {
    k_caches_.clear();
    v_caches_.clear();
}

BackendBuffer* KvManager::get_k_cache(int64_t layer) {
    if (layer >= 0 && layer < (int64_t)k_caches_.size()) return k_caches_[layer].get();
    return nullptr;
}

BackendBuffer* KvManager::get_v_cache(int64_t layer) {
    if (layer >= 0 && layer < (int64_t)v_caches_.size()) return v_caches_[layer].get();
    return nullptr;
}
