#pragma once

#include "backend.h"
#include <cuda_runtime_api.h>
#include <string>

class CudaBuffer final : public BackendBuffer
{
public:
    CudaBuffer() = default;
    CudaBuffer(void *ptr, size_t bytes, MemoryTier tier);
    ~CudaBuffer() override;
    CudaBuffer(const CudaBuffer &) = delete;
    CudaBuffer &operator=(const CudaBuffer &) = delete;

    void *raw_handle() override { return ptr_; }
    size_t size() const override { return bytes_; }
    MemoryTier tier() const override { return tier_; }

private:
    void *ptr_ = nullptr;
    size_t bytes_ = 0;
    MemoryTier tier_ = MemoryTier::TIER0_DEDICATED_VRAM;
};

class CudaBackend final : public Backend
{
public:
    CudaBackend() = default;
    ~CudaBackend() override;

    const std::string &name() const override { return name_; }
    BackendDeviceType type() const override { return BackendDeviceType::NVIDIA_GPU; }
    bool initialize() override;
    DeviceStats query_stats() override;

    std::unique_ptr<BackendBuffer> allocate(size_t bytes, MemoryTier tier = MemoryTier::TIER0_DEDICATED_VRAM) override;
    bool upload(BackendBuffer &dst, const void *host_src, size_t bytes, bool async = false) override;
    bool download(void *host_dst, const BackendBuffer &src, size_t bytes, bool async = false) override;
    bool copy(BackendBuffer &dst, const BackendBuffer &src, size_t bytes) override;
    void synchronize() override;

    void rms_norm(BackendBuffer &out, BackendBuffer &x, BackendBuffer &weight, int64_t n, float eps = 1e-6f) override;
    void add_rms_norm(BackendBuffer &residual, BackendBuffer &branch, BackendBuffer &weight, BackendBuffer &norm_out, int64_t n, float eps = 1e-6f) override;
    void gemv_q4_0(BackendBuffer &dst, BackendBuffer &a, BackendBuffer &b, int64_t N, int64_t K) override;
    void gemv_q8_0(BackendBuffer &dst, BackendBuffer &a, BackendBuffer &b, int64_t N, int64_t K) override;
    void gemv_q4_0_fused_ffn(BackendBuffer &dst, BackendBuffer &a, BackendBuffer &gate, BackendBuffer &up, int64_t N, int64_t K) override;
    void rope(BackendBuffer &x, int64_t n_embd, int64_t n_head, int64_t pos, int64_t n_tokens) override;
    void embed_lookup_q4_0(BackendBuffer &hidden, BackendBuffer &embd_table, int token_id, int64_t n_embd) override;
    void argmax(BackendBuffer &out_idx, BackendBuffer &logits, int64_t n) override;

    bool ok() const { return initialized_; }
    cudaStream_t stream() const { return stream_; }

private:
    bool valid_range(const BackendBuffer &buffer, size_t bytes, const char *operation) const;
    bool check(cudaError_t result, const char *operation) const;

    int device_ = 0;
    bool initialized_ = false;
    std::string name_ = "CUDA (uninitialized)";
    cudaStream_t stream_ = nullptr;
};
