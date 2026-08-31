#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include "../backends/backend.h"

class MemoryEngine {
public:
    explicit MemoryEngine(Backend &backend);
    ~MemoryEngine();

    bool initialize(size_t vram_pool_size, size_t pinned_host_pool_size);
    std::shared_ptr<BackendBuffer> get_or_allocate_tensor(const std::string &name, size_t bytes, MemoryTier tier);
    void evict_tensor(const std::string &name);
    void flush_all();

private:
    Backend &backend_;
    std::unordered_map<std::string, std::shared_ptr<BackendBuffer>> resident_tensors_;
};
