#include "memory_engine.h"
#include <algorithm>

MemoryEngine::MemoryEngine(Backend &backend) : backend_(backend) {}

MemoryEngine::~MemoryEngine()
{
    flush_all();
}

bool MemoryEngine::initialize(size_t vram_pool_size, size_t pinned_host_pool_size)
{
    vram_pool_size_ = vram_pool_size;
    pinned_host_pool_size_ = pinned_host_pool_size;
    vram_allocated_bytes_ = 0;
    pinned_allocated_bytes_ = 0;
    vram_peak_bytes_ = 0;
    total_allocations_ = 0;
    total_evictions_ = 0;
    total_migrations_ = 0;
    total_hits_ = 0;
    total_misses_ = 0;
    resident_tensors_.clear();
    lru_list_.clear();
    return true;
}

std::shared_ptr<BackendBuffer> MemoryEngine::get_or_allocate_tensor(const std::string &name, size_t bytes, MemoryTier tier)
{
    auto it = resident_tensors_.find(name);
    if (it != resident_tensors_.end())
    {
        total_hits_++;
        // Move to back of LRU list
        auto lru_it = std::find(lru_list_.begin(), lru_list_.end(), name);
        if (lru_it != lru_list_.end())
        {
            lru_list_.erase(lru_it);
        }
        lru_list_.push_back(name);
        return it->second;
    }

    total_misses_++;

    // Budget Enforcement & Eviction for VRAM
    if (tier == MemoryTier::TIER0_DEDICATED_VRAM && vram_pool_size_ > 0)
    {
        while (vram_allocated_bytes_ + bytes > vram_pool_size_ && !lru_list_.empty())
        {
            // Evict oldest tensor from VRAM
            std::string victim = lru_list_.front();
            evict_tensor(victim);
            total_evictions_++;
        }
    }

    auto buf = backend_.allocate(bytes, tier);
    if (!buf)
        return nullptr;

    std::shared_ptr<BackendBuffer> s_buf(std::move(buf));
    resident_tensors_[name] = s_buf;
    lru_list_.push_back(name);
    total_allocations_++;

    if (tier == MemoryTier::TIER0_DEDICATED_VRAM)
    {
        vram_allocated_bytes_ += bytes;
        if (vram_allocated_bytes_ > vram_peak_bytes_)
        {
            vram_peak_bytes_ = vram_allocated_bytes_;
        }
    }
    else if (tier == MemoryTier::TIER2_HOST_PINNED_RAM || tier == MemoryTier::TIER1_SHARED_IGPU)
    {
        pinned_allocated_bytes_ += bytes;
    }

    return s_buf;
}

bool MemoryEngine::migrate_tensor(const std::string &name, MemoryTier target_tier)
{
    auto it = resident_tensors_.find(name);
    if (it == resident_tensors_.end() || !it->second)
        return false;

    auto old_buf = it->second;
    if (old_buf->tier() == target_tier)
        return true;

    size_t sz = old_buf->size();
    auto new_buf = backend_.allocate(sz, target_tier);
    if (!new_buf)
        return false;

    // Perform copy
    backend_.copy(*new_buf, *old_buf, sz);
    backend_.synchronize();

    // Adjust accounting
    if (old_buf->tier() == MemoryTier::TIER0_DEDICATED_VRAM)
    {
        vram_allocated_bytes_ = (vram_allocated_bytes_ >= sz) ? (vram_allocated_bytes_ - sz) : 0;
    }
    else
    {
        pinned_allocated_bytes_ = (pinned_allocated_bytes_ >= sz) ? (pinned_allocated_bytes_ - sz) : 0;
    }

    if (target_tier == MemoryTier::TIER0_DEDICATED_VRAM)
    {
        vram_allocated_bytes_ += sz;
        if (vram_allocated_bytes_ > vram_peak_bytes_)
            vram_peak_bytes_ = vram_allocated_bytes_;
    }
    else
    {
        pinned_allocated_bytes_ += sz;
    }

    it->second = std::shared_ptr<BackendBuffer>(std::move(new_buf));
    total_migrations_++;
    return true;
}

void MemoryEngine::evict_tensor(const std::string &name)
{
    auto it = resident_tensors_.find(name);
    if (it != resident_tensors_.end())
    {
        if (it->second)
        {
            size_t sz = it->second->size();
            if (it->second->tier() == MemoryTier::TIER0_DEDICATED_VRAM)
            {
                vram_allocated_bytes_ = (vram_allocated_bytes_ >= sz) ? (vram_allocated_bytes_ - sz) : 0;
            }
            else
            {
                pinned_allocated_bytes_ = (pinned_allocated_bytes_ >= sz) ? (pinned_allocated_bytes_ - sz) : 0;
            }
        }
        resident_tensors_.erase(it);
    }
    auto lru_it = std::find(lru_list_.begin(), lru_list_.end(), name);
    if (lru_it != lru_list_.end())
    {
        lru_list_.erase(lru_it);
    }
}

void MemoryEngine::flush_all()
{
    resident_tensors_.clear();
    lru_list_.clear();
    vram_allocated_bytes_ = 0;
    pinned_allocated_bytes_ = 0;
}

MemoryStats MemoryEngine::query_stats() const
{
    MemoryStats st;
    st.vram_capacity_bytes = vram_pool_size_;
    st.vram_used_bytes = vram_allocated_bytes_;
    st.vram_peak_bytes = vram_peak_bytes_;
    st.pinned_capacity_bytes = pinned_host_pool_size_;
    st.pinned_used_bytes = pinned_allocated_bytes_;
    st.total_allocations = total_allocations_;
    st.total_evictions = total_evictions_;
    st.total_migrations = total_migrations_;
    st.total_hits = total_hits_;
    st.total_misses = total_misses_;
    return st;
}
