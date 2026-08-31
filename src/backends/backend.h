#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

enum class BackendDeviceType {
    NVIDIA_GPU,
    INTEL_IGPU,
    AMD_GPU,
    CPU_AVX2,
    UNKNOWN
};

enum class MemoryTier {
    TIER0_DEDICATED_VRAM,  // Dedicated GDDR (Fastest: ~128 GB/s)
    TIER1_SHARED_IGPU,     // Integrated GPU Unified Memory (~25-35 GB/s)
    TIER2_HOST_PINNED_RAM, // Pinned Host System RAM (DMA Capable)
    TIER3_HOST_PAGEABLE,   // Regular Host RAM
    TIER4_MMAP_STORAGE     // NVMe / SSD Backed
};

struct DeviceStats {
    std::string device_name;
    BackendDeviceType type;
    size_t total_memory_bytes;
    size_t free_memory_bytes;
    size_t max_alloc_bytes;
    int compute_units;
    double tflops_fp32;
    double memory_bandwidth_gbs;
    double dma_transfer_bandwidth_gbs;
};

// Abstract device memory buffer handle
class BackendBuffer {
public:
    virtual ~BackendBuffer() = default;
    virtual void* raw_handle() = 0;
    virtual size_t size() const = 0;
    virtual MemoryTier tier() const = 0;
};

// Core Backend Abstraction Interface
class Backend {
public:
    virtual ~Backend() = default;

    virtual const std::string& name() const = 0;
    virtual BackendDeviceType type() const = 0;
    virtual bool initialize() = 0;
    virtual DeviceStats query_stats() = 0;

    // Memory operations
    virtual std::unique_ptr<BackendBuffer> allocate(size_t bytes, MemoryTier tier = MemoryTier::TIER0_DEDICATED_VRAM) = 0;
    virtual bool upload(BackendBuffer &dst, const void *host_src, size_t bytes, bool async = false) = 0;
    virtual bool download(void *host_dst, const BackendBuffer &src, size_t bytes, bool async = false) = 0;
    virtual bool copy(BackendBuffer &dst, const BackendBuffer &src, size_t bytes) = 0;
    virtual void synchronize() = 0;

    // Compute Primitives
    virtual void rms_norm(BackendBuffer &out, BackendBuffer &x, BackendBuffer &weight, int64_t n, float eps = 1e-6f) = 0;
    virtual void add_rms_norm(BackendBuffer &residual, BackendBuffer &branch, BackendBuffer &weight, BackendBuffer &norm_out, int64_t n, float eps = 1e-6f) = 0;
    virtual void gemv_q4_0(BackendBuffer &dst, BackendBuffer &a, BackendBuffer &b, int64_t N, int64_t K) = 0;
    virtual void gemv_q8_0(BackendBuffer &dst, BackendBuffer &a, BackendBuffer &b, int64_t N, int64_t K) = 0;
    virtual void gemv_q4_0_fused_ffn(BackendBuffer &dst, BackendBuffer &a, BackendBuffer &gate, BackendBuffer &up, int64_t N, int64_t K) = 0;
    virtual void rope(BackendBuffer &x, int64_t n_embd, int64_t n_head, int64_t pos, int64_t n_tokens) = 0;
    virtual void embed_lookup_q4_0(BackendBuffer &hidden, BackendBuffer &embd_table, int token_id, int64_t n_embd) = 0;
    virtual void argmax(BackendBuffer &out_idx, BackendBuffer &logits, int64_t n) = 0;
};
