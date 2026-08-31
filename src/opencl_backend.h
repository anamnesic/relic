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
};

struct ClBuffer {
    cl_mem mem = nullptr;
    size_t size = 0;

    ClBuffer() = default;
    ~ClBuffer() { release(); }

    ClBuffer(ClBuffer &&other) noexcept : mem(other.mem), size(other.size) {
        other.mem = nullptr;
        other.size = 0;
    }

    ClBuffer &operator=(ClBuffer &&other) noexcept {
        if (this != &other) {
            release();
            mem = other.mem;
            size = other.size;
            other.mem = nullptr;
            other.size = 0;
        }
        return *this;
    }

    ClBuffer(const ClBuffer &) = delete;
    ClBuffer &operator=(const ClBuffer &) = delete;

    bool alloc(cl_context ctx, size_t bytes, cl_mem_flags flags = CL_MEM_READ_WRITE) {
        release();
        if (bytes == 0) return true;
        cl_int err;
        mem = clCreateBuffer(ctx, flags, bytes, nullptr, &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to allocate %zu bytes in OpenCL: %d\n", bytes, err);
            return false;
        }
        size = bytes;
        return true;
    }

    void release() {
        if (mem) {
            clReleaseMemObject(mem);
            mem = nullptr;
            size = 0;
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

class OpenClBackend {
public:
    ClDevice dev;
    bool initialized = false;

    bool init(int platform_idx = -1, int device_idx = 0);
    void shutdown();

    bool build_kernel(ClKernel &k, const char *source, const char *kname, const char *opts = "");

    // High-performance GEMV and fused kernels
    void gemv_f32_nt(ClBuffer &dst, ClBuffer &a, ClBuffer &b, int64_t N, int64_t K);
    void gemv_q8_0(ClBuffer &dst, ClBuffer &a, ClBuffer &b, int64_t N, int64_t K);
    void gemv_q4_0(ClBuffer &dst, ClBuffer &a, ClBuffer &b, int64_t N, int64_t K);
    void swiglu(ClBuffer &dst, ClBuffer &gate, ClBuffer &up, int64_t n);
    void add_rms_norm(ClBuffer &residual, ClBuffer &branch, ClBuffer &weight, ClBuffer &norm_out, int64_t n, float eps = 1e-6f);
    void qwen_conv1d(ClBuffer &conv_state, ClBuffer &conv_in, ClBuffer &weight, ClBuffer &conv_out, int64_t C);
    void qwen_deltanet(ClBuffer &ssm_state, ClBuffer &conv_out, ClBuffer &alpha, ClBuffer &beta, ClBuffer &delta_out, int64_t key_dim, int64_t qk_dim, int64_t linear_inner);
    void qwen_attention_step(ClBuffer &q_buf, ClBuffer &k_buf, ClBuffer &v_buf, ClBuffer &k_cache, ClBuffer &v_cache, ClBuffer &attn_out, int64_t n_head, int64_t n_kv_head, int64_t head_dim, int64_t n_embd, int64_t pos, int64_t max_seq);

    // Core operations
    void rms_norm(ClBuffer &out, ClBuffer &x, ClBuffer &weight, int64_t n, int64_t rows);
    void matmul_f32(ClBuffer &dst, ClBuffer &a, ClBuffer &b, int64_t M, int64_t N, int64_t K);
    void matmul_f32_nt(ClBuffer &dst, ClBuffer &a, ClBuffer &b, int64_t M, int64_t N, int64_t K);
    void rope(ClBuffer &x, int64_t n_embd, int64_t n_head, int64_t pos, int64_t n_tokens);
    void softmax(ClBuffer &x, int64_t n, int64_t rows);
    void silu(ClBuffer &out, ClBuffer &x, int64_t n);
    void add(ClBuffer &dst, ClBuffer &a, ClBuffer &b, int64_t n);
    void mul(ClBuffer &dst, ClBuffer &a, ClBuffer &b, int64_t n);
    void copy(ClBuffer &dst, ClBuffer &src, int64_t n);
    void fill(ClBuffer &buf, float val, int64_t n);
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

            for (int64_t b = 0; b < n_blocks; b++) {
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
