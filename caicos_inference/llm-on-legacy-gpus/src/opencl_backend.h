#pragma once
#include <CL/cl.h>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct ClDeviceInfo {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    std::string name;
    std::string platform_name;
    size_t global_mem;
    size_t max_alloc;
    size_t max_wg_size;
    cl_uint compute_units;
    int opencl_c_major = 0;
    int opencl_c_minor = 0;
    bool fp16 = false;
    cl_int alignment;
};

struct ClBuffer {
    cl_mem mem = nullptr;
    size_t size = 0;
    cl_context context = nullptr;

    ~ClBuffer() { release(); }
    ClBuffer() = default;
    ClBuffer(ClBuffer &&o) noexcept : mem(o.mem), size(o.size), context(o.context) {
        o.mem = nullptr; o.size = 0;
    }
    ClBuffer &operator=(ClBuffer &&o) noexcept {
        if (this != &o) { release(); mem = o.mem; size = o.size; context = o.context; o.mem = nullptr; o.size = 0; }
        return *this;
    }
    ClBuffer(const ClBuffer &) = delete;
    ClBuffer &operator=(const ClBuffer &) = delete;

    bool alloc(cl_context ctx, size_t sz) {
        release();
        context = ctx;
        size = sz;
        cl_int err;
        mem = clCreateBuffer(context, CL_MEM_READ_WRITE, size, nullptr, &err);
        return err == CL_SUCCESS;
    }

    void release() {
        if (mem) { clReleaseMemObject(mem); mem = nullptr; }
        size = 0;
    }
};

struct ClKernel {
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    std::string name;

    ~ClKernel() { release(); }
    ClKernel() = default;
    ClKernel(ClKernel &&o) noexcept : program(o.program), kernel(o.kernel), name(std::move(o.name)) {
        o.program = nullptr; o.kernel = nullptr;
    }
    ClKernel &operator=(ClKernel &&o) noexcept {
        if (this != &o) { release(); program = o.program; kernel = o.kernel; name = std::move(o.name); o.program = nullptr; o.kernel = nullptr; }
        return *this;
    }
    ClKernel(const ClKernel &) = delete;
    ClKernel &operator=(const ClKernel &) = delete;

    void release() {
        if (kernel) { clReleaseKernel(kernel); kernel = nullptr; }
        if (program) { clReleaseProgram(program); program = nullptr; }
    }
};

struct OpenClBackend {
    ClDeviceInfo dev;
    bool initialized = false;

    ~OpenClBackend() { shutdown(); }

    bool init(int platform_idx = 0, int device_idx = 0);
    void shutdown();
    bool build_kernel(ClKernel &k, const char *source, const char *kname, const char *opts = "");

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

extern const char *caicos_kernel_source;
