#include "memory_engine.h"

MemoryEngine::MemoryEngine(Backend &backend) : backend_(backend) {}

MemoryEngine::~MemoryEngine() {
    flush_all();
}

bool MemoryEngine::initialize(size_t vram_pool_size, size_t pinned_host_pool_size) {
    (void)vram_pool_size;
    (void)pinned_host_pool_size;
    return true;
}

std::shared_ptr<BackendBuffer> MemoryEngine::get_or_allocate_tensor(const std::string &name, size_t bytes, MemoryTier tier) {
    auto it = resident_tensors_.find(name);
    if (it != resident_tensors_.end()) {
        return it->second;
    }

    auto buf = backend_.allocate(bytes, tier);
    if (!buf) return nullptr;

    std::shared_ptr<BackendBuffer> s_buf(std::move(buf));
    resident_tensors_[name] = s_buf;
    return s_buf;
}

void MemoryEngine::evict_tensor(const std::string &name) {
    resident_tensors_.erase(name);
}

void MemoryEngine::flush_all() {
    resident_tensors_.clear();
}
