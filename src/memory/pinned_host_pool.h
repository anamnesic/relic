#pragma once
#include <cstddef>
#include <vector>
#include <memory>

class PinnedHostPool {
public:
    explicit PinnedHostPool(size_t capacity_bytes);
    ~PinnedHostPool();

    void* allocate_pinned(size_t bytes);
    void reset();

    size_t capacity() const { return capacity_; }
    size_t used() const { return used_offset_; }

private:
    size_t capacity_;
    size_t used_offset_;
    void* host_ptr_;
};
