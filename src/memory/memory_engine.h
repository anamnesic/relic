#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "../backends/backend.h"

struct MemoryStats
{
    size_t vram_capacity_bytes = 0;
    size_t vram_used_bytes = 0;
    size_t vram_peak_bytes = 0;
    size_t pinned_capacity_bytes = 0;
    size_t pinned_used_bytes = 0;
    size_t total_allocations = 0;
    size_t total_evictions = 0;
    size_t total_migrations = 0;
    size_t total_hits = 0;
    size_t total_misses = 0;
};

class MemoryEngine
{
public:
    explicit MemoryEngine(Backend &backend);
    ~MemoryEngine();

    bool initialize(size_t vram_pool_size, size_t pinned_host_pool_size);
    std::shared_ptr<BackendBuffer> get_or_allocate_tensor(const std::string &name, size_t bytes, MemoryTier tier);
    bool migrate_tensor(const std::string &name, MemoryTier target_tier);
    void evict_tensor(const std::string &name);
    void flush_all();

    MemoryStats query_stats() const;

private:
    Backend &backend_;
    size_t vram_pool_size_ = 0;
    size_t pinned_host_pool_size_ = 0;
    size_t vram_allocated_bytes_ = 0;
    size_t pinned_allocated_bytes_ = 0;
    size_t vram_peak_bytes_ = 0;
    size_t total_allocations_ = 0;
    size_t total_evictions_ = 0;
    size_t total_migrations_ = 0;
    size_t total_hits_ = 0;
    size_t total_misses_ = 0;

    std::unordered_map<std::string, std::shared_ptr<BackendBuffer>> resident_tensors_;
    std::vector<std::string> lru_list_;
};
