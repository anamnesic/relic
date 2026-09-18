#include "pinned_host_pool.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

PinnedHostPool::PinnedHostPool(size_t capacity_bytes)
    : capacity_(capacity_bytes), used_offset_(0), host_ptr_(nullptr), is_locked_(false)
{
    if (capacity_ > 0)
    {
#ifdef _WIN32
        // Allocate virtual memory with 64-byte alignment
        host_ptr_ = VirtualAlloc(nullptr, capacity_, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (host_ptr_)
        {
            if (VirtualLock(host_ptr_, capacity_))
            {
                is_locked_ = true;
            }
            else
            {
                HANDLE hProc = GetCurrentProcess();
                SIZE_T min_sz = 0, max_sz = 0;
                if (GetProcessWorkingSetSize(hProc, &min_sz, &max_sz))
                {
                    SetProcessWorkingSetSize(hProc, min_sz + capacity_, max_sz + capacity_ + (16 * 1024 * 1024));
                    if (VirtualLock(host_ptr_, capacity_))
                    {
                        is_locked_ = true;
                    }
                }
            }
        }
#else
        host_ptr_ = mmap(nullptr, capacity_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (host_ptr_ == MAP_FAILED)
        {
            host_ptr_ = nullptr;
        }
        else
        {
            if (mlock(host_ptr_, capacity_) == 0)
            {
                is_locked_ = true;
            }
        }
#endif
    }
}

PinnedHostPool::~PinnedHostPool()
{
    if (host_ptr_)
    {
        if (is_locked_)
        {
#ifdef _WIN32
            VirtualUnlock(host_ptr_, capacity_);
#else
            munlock(host_ptr_, capacity_);
#endif
            is_locked_ = false;
        }
#ifdef _WIN32
        VirtualFree(host_ptr_, 0, MEM_RELEASE);
#else
        munmap(host_ptr_, capacity_);
#endif
        host_ptr_ = nullptr;
    }
}

void *PinnedHostPool::allocate_pinned(size_t bytes)
{
    // 64-byte alignment for AVX2 and PCIe DMA bursts
    size_t aligned_bytes = (bytes + 63) & ~63;
    if (used_offset_ + aligned_bytes > capacity_)
    {
        return nullptr;
    }
    void *ptr = (char *)host_ptr_ + used_offset_;
    used_offset_ += aligned_bytes;
    return ptr;
}

void *PinnedHostPool::get_slot(int slot_idx, size_t slot_bytes)
{
    size_t offset = (size_t)slot_idx * ((capacity_ / 2 + 63) & ~63);
    if (offset + slot_bytes > capacity_ || !host_ptr_)
    {
        return nullptr;
    }
    return (char *)host_ptr_ + offset;
}

void PinnedHostPool::reset()
{
    used_offset_ = 0;
}
