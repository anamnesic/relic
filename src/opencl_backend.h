#pragma once
#include <CL/cl.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <thread>
#include "model.h"

// Macro for OpenCL error checking
#define CL_CHECK(call, msg) \
    do { \
        cl_int err = (call); \
        if (err != CL_SUCCESS) { \
            fprintf(stderr, "OpenCL error %d at %s:%d: %s\n", err, __FILE__, __LINE__, msg); \
            return false; \
        } \
    } while (0)

#define CL_CHECK_VOID(call, msg) \
    do { \
        cl_int err = (call); \
        if (err != CL_SUCCESS) { \
            fprintf(stderr, "OpenCL error %d at %s:%d: %s\n", err, __FILE__, __LINE__, msg); \
            return; \
        } \
    } while (0)

struct ClDevice {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    std::string name;
    std::string platform_name;
    size_t global_mem = 0;
    size_t max_alloc = 0;
    size_t max_wg_size = 0;
    cl_uint compute_units = 0;
    bool fp16 = false;
    int alignment = 128;
    int opencl_c_major = 1;
    int opencl_c_minor = 2;
    cl_program program = nullptr;
};

struct ClMemoryTracker {
    static inline size_t current_bytes = 0;
    static inline size_t peak_bytes = 0;

    static void record_alloc(size_t bytes) {
        current_bytes += bytes;
        if (current_bytes > peak_bytes) {
            peak_bytes = current_bytes;
        }
    }

    static void record_free(size_t bytes) {
        if (current_bytes >= bytes) {
            current_bytes -= bytes;
        } else {
            current_bytes = 0;
        }
    }

    static void reset_peak() {
        peak_bytes = current_bytes;
    }
};

#include "backends/backend.h"

struct ClBuffer : public BackendBuffer {
    cl_mem mem = nullptr;
    size_t size_bytes = 0;
    bool is_owner = true;
    MemoryTier tier_ = MemoryTier::TIER0_DEDICATED_VRAM;

    ClBuffer() = default;
    ~ClBuffer() override { release(); }

    void *raw_handle() override { return (void *)mem; }
    size_t size() const override { return size_bytes; }
    MemoryTier tier() const override { return tier_; }

    ClBuffer(ClBuffer &&other) noexcept : mem(other.mem), size_bytes(other.size_bytes), is_owner(other.is_owner), tier_(other.tier_) {
        other.mem = nullptr;
        other.size_bytes = 0;
        other.is_owner = true;
    }

    ClBuffer &operator=(ClBuffer &&other) noexcept {
        if (this != &other) {
            release();
            mem = other.mem;
            size_bytes = other.size_bytes;
            is_owner = other.is_owner;
            tier_ = other.tier_;
            other.mem = nullptr;
            other.size_bytes = 0;
            other.is_owner = true;
        }
        return *this;
    }

    ClBuffer(const ClBuffer &) = delete;
    ClBuffer &operator=(const ClBuffer &) = delete;

    static ClBuffer borrow(cl_mem m, size_t sz, MemoryTier t = MemoryTier::TIER0_DEDICATED_VRAM) {
        ClBuffer buf;
        buf.mem = m;
        buf.size_bytes = sz;
        buf.is_owner = false;
        buf.tier_ = t;
        return buf;
    }

    bool alloc(cl_context ctx, size_t bytes, cl_mem_flags flags = CL_MEM_READ_WRITE, MemoryTier t = MemoryTier::TIER0_DEDICATED_VRAM) {
        release();
        if (bytes == 0) return true;
        cl_int err;
        mem = clCreateBuffer(ctx, flags, bytes, nullptr, &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to allocate %zu bytes in OpenCL: %d\n", bytes, err);
            return false;
        }
        size_bytes = bytes;
        is_owner = true;
        tier_ = t;
        ClMemoryTracker::record_alloc(bytes);
        return true;
    }

    void release() {
        if (mem) {
            if (is_owner) {
                ClMemoryTracker::record_free(size_bytes);
                clReleaseMemObject(mem);
            }
            mem = nullptr;
            size_bytes = 0;
            is_owner = true;
        }
    }
};

struct ClKernel {
    cl_kernel kernel = nullptr;
    cl_program program = nullptr;
    std::string name;

    ClKernel() = default;
    ~ClKernel() { release(); }

    ClKernel(ClKernel &&other) noexcept
        : kernel(other.kernel), program(other.program), name(std::move(other.name)) {
        other.kernel = nullptr;
        other.program = nullptr;
    }

    ClKernel &operator=(ClKernel &&other) noexcept {
        if (this != &other) {
            release();
            kernel = other.kernel;
            program = other.program;
            name = std::move(other.name);
            other.kernel = nullptr;
            other.program = nullptr;
        }
        return *this;
    }

    ClKernel(const ClKernel &) = delete;
    ClKernel &operator=(const ClKernel &) = delete;

    void release() {
        if (kernel) { clReleaseKernel(kernel); kernel = nullptr; }
        if (program) { clReleaseProgram(program); program = nullptr; }
    }
};

class OpenClBackend : public Backend {
public:
    ClDevice dev;
    bool initialized = false;

    ~OpenClBackend() override { shutdown(); }

    const std::string &name() const override { return dev.name; }
    BackendDeviceType type() const override { return BackendDeviceType::NVIDIA_GPU; }
    bool initialize() override { return init(); }
    DeviceStats query_stats() override;

    static void list_devices();
    bool init(int platform_idx = -1, int device_idx = 0);
    void shutdown();

    bool build_kernel(ClKernel &k, const char *source, const char *kname, const char *opts = "");

    // Memory operations
    std::unique_ptr<BackendBuffer> allocate(size_t bytes, MemoryTier tier = MemoryTier::TIER0_DEDICATED_VRAM) override;
    bool upload(BackendBuffer &dst, const void *host_src, size_t bytes, bool async = false) override;
    bool download(void *host_dst, const BackendBuffer &src, size_t bytes, bool async = false) override;
    bool copy(BackendBuffer &dst, const BackendBuffer &src, size_t bytes) override;
    void synchronize() override;

    // Backend abstract compute primitives
    void rms_norm(BackendBuffer &out, BackendBuffer &x, BackendBuffer &weight, int64_t n, float eps = 1e-6f) override;
    void add_rms_norm(BackendBuffer &residual, BackendBuffer &branch, BackendBuffer &weight, BackendBuffer &norm_out, int64_t n, float eps = 1e-6f) override;
    void gemv_q4_0(BackendBuffer &dst, BackendBuffer &a, BackendBuffer &b, int64_t N, int64_t K) override;
    void gemv_q8_0(BackendBuffer &dst, BackendBuffer &a, BackendBuffer &b, int64_t N, int64_t K) override;
    void gemv_q4_0_fused_ffn(BackendBuffer &dst, BackendBuffer &a, BackendBuffer &gate, BackendBuffer &up, int64_t N, int64_t K) override;
    void rope(BackendBuffer &x, int64_t n_embd, int64_t n_head, int64_t pos, int64_t n_tokens) override;
    void embed_lookup_q4_0(BackendBuffer &hidden, BackendBuffer &embd_table, int token_id, int64_t n_embd) override;
    void argmax(BackendBuffer &out_idx, BackendBuffer &logits, int64_t n) override;

    // High-performance GEMV and fused kernels
    void gemv_f32_nt(ClBuffer &dst, ClBuffer &a, ClBuffer &b, int64_t N, int64_t K);
    void gemv_q8_0(ClBuffer &dst, ClBuffer &a, ClBuffer &b, int64_t N, int64_t K);
    void gemv_q4_0(ClBuffer &dst, ClBuffer &a, ClBuffer &b, int64_t N, int64_t K);
    void gemv_q4_0_ffn_swiglu(ClBuffer &dst, ClBuffer &a, ClBuffer &b_gate, ClBuffer &b_up, int64_t N, int64_t K);
    void swiglu(ClBuffer &dst, ClBuffer &gate, ClBuffer &up, int64_t n);
    void add_rms_norm(ClBuffer &residual, ClBuffer &branch, ClBuffer &weight, ClBuffer &norm_out, int64_t n, float eps = 1e-6f);
    void qwen_conv1d(ClBuffer &conv_state, ClBuffer &conv_in, ClBuffer &weight, ClBuffer &conv_out, int64_t C);
    void qwen_deltanet(ClBuffer &ssm_state, ClBuffer &conv_out, ClBuffer &alpha, ClBuffer &beta, ClBuffer &delta_out, int64_t key_dim, int64_t qk_dim, int64_t linear_inner);
    void qwen_deltanet_fused(ClBuffer &ssm_state, ClBuffer &conv_out, ClBuffer &alpha, ClBuffer &beta,
                             ClBuffer *ssm_a, ClBuffer *ssm_dt,
                             ClBuffer *ssm_norm_w, ClBuffer *attn_gate, ClBuffer &delta_out,
                             int64_t key_dim, int64_t qk_dim, int64_t linear_inner, float norm_eps = 1e-6f);
    void qwen_deinterleave_q_gate(ClBuffer &q_full, ClBuffer &q, ClBuffer &gate, int64_t num_heads, int64_t head_dim);
    void qwen_qk_norm(ClBuffer &dst, ClBuffer &src, ClBuffer &norm_w, int64_t num_heads, int64_t head_dim, float eps = 1e-6f);
    void qwen_attn_gate_mul(ClBuffer &attn_out, ClBuffer &gate, int64_t n);
    void qwen_attention_step(ClBuffer &q_buf, ClBuffer &k_cache, ClBuffer &v_cache, ClBuffer &attn_out, int64_t n_head, int64_t n_kv_head, int64_t head_dim, int64_t pos, int64_t max_seq);
    void gemv_q4_k(ClBuffer &dst, ClBuffer &a, ClBuffer &b, int64_t N, int64_t K);
    void gemv_q6_k(ClBuffer &dst, ClBuffer &a, ClBuffer &b, int64_t N, int64_t K);
    void gemm_q4_0(ClBuffer &dst, ClBuffer &a, ClBuffer &b, int64_t M, int64_t N, int64_t K);
    void kv_cache_append_fp16(ClBuffer &k_cache, ClBuffer &v_cache, ClBuffer &k_in, ClBuffer &v_in, int64_t pos, int64_t kv_stride);
    void qwen_attention_step_fp16(ClBuffer &q_buf, ClBuffer &k_cache, ClBuffer &v_cache, ClBuffer &attn_out, int64_t n_head, int64_t n_kv_head, int64_t head_dim, int64_t pos, int64_t max_seq);
    void kv_cache_append_q4(ClBuffer &k_cache, ClBuffer &v_cache, ClBuffer &k_in, ClBuffer &v_in, int64_t pos, int64_t kv_stride);
    void qwen_attention_step_q4(ClBuffer &q_buf, ClBuffer &k_cache, ClBuffer &v_cache, ClBuffer &attn_out, int64_t n_head, int64_t n_kv_head, int64_t head_dim, int64_t pos, int64_t max_seq);
    int sample_logits(ClBuffer &logits, int64_t n_vocab, float temperature = 0.0f, int top_k = 40, float top_p = 0.9f);

    // Core operations
    void rms_norm(ClBuffer &out, ClBuffer &x, ClBuffer &weight, int64_t n, int64_t rows);
    void matmul_f32(ClBuffer &dst, ClBuffer &a, ClBuffer &b, int64_t M, int64_t N, int64_t K);
    void matmul_f32_nt(ClBuffer &dst, ClBuffer &a, ClBuffer &b, int64_t M, int64_t N, int64_t K);
    void rope(ClBuffer &x, int64_t n_embd, int64_t n_head, int64_t pos, int64_t n_tokens, float freq_base = 10000.0f, int64_t rope_dim = 0);
    void softmax(ClBuffer &x, int64_t n, int64_t rows);
    void silu(ClBuffer &out, ClBuffer &x, int64_t n);
    void add(ClBuffer &dst, ClBuffer &a, ClBuffer &b, int64_t n);
    void mul(ClBuffer &dst, ClBuffer &a, ClBuffer &b, int64_t n);
    void copy(ClBuffer &dst, ClBuffer &src, int64_t n);
    void fill(ClBuffer &buf, float val, int64_t n);
    void argmax(ClBuffer &out_idx, ClBuffer &logits, int64_t n);
    void embed_lookup(ClBuffer &hidden, ClBuffer &embd_table, int token_id, int64_t n_embd);
};

struct GpuTensorStore {
    std::unordered_map<std::string, ClBuffer> buffers;
    std::unordered_map<std::string, GgmlType> types;

    bool upload_auto_q4(cl_context ctx, cl_command_queue queue, const std::string &name,
                        const void *host_data, size_t bytes, GgmlType orig_type, int64_t n_elements,
                        size_t &out_uploaded_bytes) {
        if (!host_data || bytes == 0) return false;

        if (orig_type == GgmlType::Q8_0 && n_elements > 0 && (n_elements % 32 == 0)) {
            // Dynamic On-the-Fly Q8_0 -> Q4_0 Repacking
            int64_t n_blocks = n_elements / 32;
            size_t q4_bytes = (size_t)n_blocks * 18;
            std::vector<uint8_t> q4_data(q4_bytes);

            const uint8_t *src_q8 = (const uint8_t *)host_data;
            uint8_t *dst_q4 = q4_data.data();

            int n_threads = std::min((int)std::thread::hardware_concurrency(), 8);
            if (n_threads <= 0) n_threads = 1;
            if (n_blocks < 2048) n_threads = 1;

            auto worker_func = [=](int64_t b_start, int64_t b_end) {
                for (int64_t b = b_start; b < b_end; b++) {
                    const uint8_t *b_q8 = src_q8 + b * 34;
                    uint8_t *b_q4 = dst_q4 + b * 18;

                    uint16_t d8_bits = (uint16_t)b_q8[0] | ((uint16_t)b_q8[1] << 8);
                    float d8 = half_bits_to_float(d8_bits);
                    const int8_t *qs8 = (const int8_t *)(b_q8 + 2);

                    float w[32];
                    float amax = 0.0f;
                    for (int i = 0; i < 32; i++) {
                        float val = (float)qs8[i] * d8;
                        w[i] = val;
                        float abs_val = fabsf(val);
                        if (abs_val > amax) amax = abs_val;
                    }

                    float d4 = amax / 7.0f;
                    float id = (d4 > 0.0f) ? (1.0f / d4) : 0.0f;

                    uint16_t d4_bits = float_to_half_bits(d4);
                    b_q4[0] = (uint8_t)(d4_bits & 0xFF);
                    b_q4[1] = (uint8_t)(d4_bits >> 8);

                    for (int i = 0; i < 16; i++) {
                        int v0 = (int)roundf(w[i] * id);
                        v0 = std::min(7, std::max(-8, v0));
                        uint8_t q0 = (uint8_t)(v0 + 8);

                        int v1 = (int)roundf(w[i + 16] * id);
                        v1 = std::min(7, std::max(-8, v1));
                        uint8_t q1 = (uint8_t)(v1 + 8);

                        b_q4[2 + i] = (uint8_t)(q0 | (q1 << 4));
                    }
                }
            };

            if (n_threads == 1) {
                worker_func(0, n_blocks);
            } else {
                std::vector<std::thread> workers;
                workers.reserve(n_threads);
                for (int t = 0; t < n_threads; t++) {
                    int64_t b_start = t * n_blocks / n_threads;
                    int64_t b_end = (t == n_threads - 1) ? n_blocks : (t + 1) * n_blocks / n_threads;
                    workers.emplace_back(worker_func, b_start, b_end);
                }
                for (auto &w : workers) {
                    w.join();
                }
            }

            ClBuffer buf;
            if (!buf.alloc(ctx, q4_bytes)) return false;
            cl_int err = clEnqueueWriteBuffer(queue, buf.mem, CL_FALSE, 0, q4_bytes, q4_data.data(), 0, nullptr, nullptr);
            if (err != CL_SUCCESS) return false;

            buffers[name] = std::move(buf);
            types[name] = GgmlType::Q4_0;
            out_uploaded_bytes = q4_bytes;
            return true;
        }

        // Direct upload for F32, Q4_0, etc.
        ClBuffer buf;
        if (!buf.alloc(ctx, bytes)) return false;
        cl_int err = clEnqueueWriteBuffer(queue, buf.mem, CL_FALSE, 0, bytes, host_data, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) return false;

        buffers[name] = std::move(buf);
        types[name] = orig_type;
        out_uploaded_bytes = bytes;
        return true;
    }

    ClBuffer *get(const std::string &name) {
        auto it = buffers.find(name);
        if (it != buffers.end()) return &it->second;
        return nullptr;
    }

    GgmlType get_type(const std::string &name, GgmlType def) const {
        auto it = types.find(name);
        if (it != types.end()) return it->second;
        return def;
    }

    void clear() {
        buffers.clear();
        types.clear();
    }
};

extern const char *caicos_kernel_source;
