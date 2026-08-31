#include "pinned_host_pool.h"
#include <cstdlib>
#include <cstring>
#include <windows.h>

PinnedHostPool::PinnedHostPool(size_t capacity_bytes)
    : capacity_(capacity_bytes), used_offset_(0), host_ptr_(nullptr) {
    if (capacity_ > 0) {
        // Allocate virtual memory with 64-byte alignment
        host_ptr_ = VirtualAlloc(nullptr, capacity_, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    }
}

PinnedHostPool::~PinnedHostPool() {
    if (host_ptr_) {
        VirtualFree(host_ptr_, 0, MEM_RELEASE);
        host_ptr_ = nullptr;
    }
}

void* PinnedHostPool::allocate_pinned(size_t bytes) {
    // 64-byte alignment for AVX2 and PCIe DMA bursts
    size_t aligned_bytes = (bytes + 63) & ~63;
    if (used_offset_ + aligned_bytes > capacity_) {
        return nullptr;
    }
    void *ptr = (char*)host_ptr_ + used_offset_;
    used_offset_ += aligned_bytes;
    return ptr;
}

void PinnedHostPool::reset() {
    used_offset_ = 0;
}
