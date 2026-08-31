#pragma once
#include <CL/cl.h>
#include <string>
#include <vector>
#include <memory>
#include "backend.h"

class IntelUhdBuffer : public BackendBuffer {
public:
    IntelUhdBuffer(cl_mem mem, size_t size, MemoryTier tier, void *host_ptr = nullptr);
    ~IntelUhdBuffer() override;

    void* raw_handle() override { return (void*)mem_; }
    size_t size() const override { return size_; }
    MemoryTier tier() const override { return tier_; }
    void* host_ptr() const { return host_ptr_; }

    cl_mem mem() const { return mem_; }

private:
    cl_mem mem_;
    size_t size_;
    MemoryTier tier_;
    void *host_ptr_;
};

class IntelUhdBackend : public Backend {
public:
    IntelUhdBackend();
    ~IntelUhdBackend() override;

    const std::string& name() const override { return name_; }
    BackendDeviceType type() const override { return BackendDeviceType::INTEL_IGPU; }
    bool initialize() override;
    DeviceStats query_stats() override;

    std::unique_ptr<BackendBuffer> allocate(size_t bytes, MemoryTier tier = MemoryTier::TIER1_SHARED_IGPU) override;
    bool upload(BackendBuffer &dst, const void *host_src, size_t bytes, bool async = false) override;
    bool download(void *host_dst, const BackendBuffer &src, size_t bytes, bool async = false) override;
    bool copy(BackendBuffer &dst, const BackendBuffer &src, size_t bytes) override;
    void synchronize() override;

    // Compute Primitives
    void rms_norm(BackendBuffer &out, BackendBuffer &x, BackendBuffer &weight, int64_t n, float eps = 1e-6f) override;
    void add_rms_norm(BackendBuffer &residual, BackendBuffer &branch, BackendBuffer &weight, BackendBuffer &norm_out, int64_t n, float eps = 1e-6f) override;
    void gemv_q4_0(BackendBuffer &dst, BackendBuffer &a, BackendBuffer &b, int64_t N, int64_t K) override;
    void gemv_q8_0(BackendBuffer &dst, BackendBuffer &a, BackendBuffer &b, int64_t N, int64_t K) override;
    void gemv_q4_0_fused_ffn(BackendBuffer &dst, BackendBuffer &a, BackendBuffer &gate, BackendBuffer &up, int64_t N, int64_t K) override;
    void rope(BackendBuffer &x, int64_t n_embd, int64_t n_head, int64_t pos, int64_t n_tokens) override;
    void embed_lookup_q4_0(BackendBuffer &hidden, BackendBuffer &embd_table, int token_id, int64_t n_embd) override;
    void argmax(BackendBuffer &out_idx, BackendBuffer &logits, int64_t n) override;

private:
    std::string name_;
    cl_platform_id platform_;
    cl_device_id device_;
    cl_context context_;
    cl_command_queue queue_;
    cl_program program_;
    bool initialized_;
};
