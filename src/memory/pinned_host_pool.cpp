#include "pinned_host_pool.h"
#include <cstdlib>
#include <cstring>
#include <windows.h>

PinnedHostPool::PinnedHostPool(size_t capacity_bytes)
    : capacity_(capacity_bytes), used_offset_(0), host_ptr_(nullptr), is_locked_(false) {
    if (capacity_ > 0) {
        // Allocate virtual memory with 64-byte alignment
        host_ptr_ = VirtualAlloc(nullptr, capacity_, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (host_ptr_) {
            // Attempt to lock into physical RAM (page-locked / non-pageable memory)
            if (!VirtualLock(host_ptr_, capacity_)) {
                HANDLE hProc = GetCurrentProcess();
                SIZE_T min_sz = 0, max_sz = 0;
                if (GetProcessWorkingSetSize(hProc, &min_sz, &max_sz)) {
                    SetProcessWorkingSetSize(hProc, min_sz + capacity_, max_sz + capacity_ + (16 * 1024 * 1024));
                    if (VirtualLock(host_ptr_, capacity_)) {
                        is_locked_ = true;
                    }
                }
            } else {
                is_locked_ = true;
            }
        }
    }
}

PinnedHostPool::~PinnedHostPool() {
    if (host_ptr_) {
        if (is_locked_) {
            VirtualUnlock(host_ptr_, capacity_);
            is_locked_ = false;
        }
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
