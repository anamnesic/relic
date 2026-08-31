#include "intel_uhd_backend.h"
#include <cstdio>
#include <cstring>
#include "../embedded_kernels.h"

IntelUhdBuffer::IntelUhdBuffer(cl_mem mem, size_t size, MemoryTier tier, void *host_ptr)
    : mem_(mem), size_(size), tier_(tier), host_ptr_(host_ptr) {}

IntelUhdBuffer::~IntelUhdBuffer() {
    if (mem_) {
        clReleaseMemObject(mem_);
        mem_ = nullptr;
    }
}

IntelUhdBackend::IntelUhdBackend()
    : name_("Intel UHD Graphics (Shared Memory OpenCL 3.0)"),
      platform_(nullptr), device_(nullptr), context_(nullptr),
      queue_(nullptr), program_(nullptr), initialized_(false) {}

IntelUhdBackend::~IntelUhdBackend() {
    if (program_) clReleaseProgram(program_);
    if (queue_) clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);
}

bool IntelUhdBackend::initialize() {
    if (initialized_) return true;

    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    if (num_platforms == 0) return false;

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    for (auto p : platforms) {
        cl_uint num_devs = 0;
        clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devs);
        if (num_devs == 0) continue;

        std::vector<cl_device_id> devs(num_devs);
        clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, num_devs, devs.data(), nullptr);

        for (auto d : devs) {
            char dev_name[256] = {0};
            clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);
            if (strstr(dev_name, "Intel") != nullptr || strstr(dev_name, "UHD") != nullptr || strstr(dev_name, "Iris") != nullptr) {
                platform_ = p;
                device_ = d;
                name_ = dev_name;
                break;
            }
        }
        if (device_) break;
    }

    if (!device_) return false;

    cl_int err;
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || !context_) return false;

    queue_ = clCreateCommandQueue(context_, device_, 0, &err);
    if (err != CL_SUCCESS || !queue_) return false;

    const char *src = (const char *)caicos_kernel_embedded_bytes;
    size_t src_len = strlen(src);
    program_ = clCreateProgramWithSource(context_, 1, &src, &src_len, &err);
    if (err == CL_SUCCESS) {
        clBuildProgram(program_, 1, &device_, "-cl-mad-enable -cl-fast-relaxed-math", nullptr, nullptr);
    }

    initialized_ = true;
    return true;
}

DeviceStats IntelUhdBackend::query_stats() {
    DeviceStats st;
    st.device_name = name_;
    st.type = BackendDeviceType::INTEL_IGPU;
    st.total_memory_bytes = 0;
    st.free_memory_bytes = 0;
    st.max_alloc_bytes = 0;
    st.compute_units = 0;
    st.memory_bandwidth_gbs = 25.6;
    st.dma_transfer_bandwidth_gbs = 25.6;
    st.tflops_fp32 = 0.55;

    if (device_) {
        cl_ulong mem = 0, max_a = 0;
        cl_uint cu = 0;
        clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_a), &max_a, nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
        st.total_memory_bytes = (size_t)mem;
        st.free_memory_bytes = (size_t)mem;
        st.max_alloc_bytes = (size_t)max_a;
        st.compute_units = (int)cu;
    }
    return st;
}

std::unique_ptr<BackendBuffer> IntelUhdBackend::allocate(size_t bytes, MemoryTier tier) {
    if (!initialized_ || bytes == 0) return nullptr;

    cl_mem_flags flags = CL_MEM_READ_WRITE;
    if (tier == MemoryTier::TIER1_SHARED_IGPU || tier == MemoryTier::TIER2_HOST_PINNED_RAM) {
        // Zero-copy shared host pointer allocation
        flags |= CL_MEM_ALLOC_HOST_PTR;
    }

    cl_int err;
    cl_mem mem = clCreateBuffer(context_, flags, bytes, nullptr, &err);
    if (err != CL_SUCCESS || !mem) return nullptr;

    return std::make_unique<IntelUhdBuffer>(mem, bytes, tier);
}

bool IntelUhdBackend::upload(BackendBuffer &dst, const void *host_src, size_t bytes, bool async) {
    if (!initialized_ || !host_src || bytes == 0) return false;
    auto *buf = dynamic_cast<IntelUhdBuffer*>(&dst);
    if (!buf) return false;

    cl_int err = clEnqueueWriteBuffer(queue_, buf->mem(), async ? CL_FALSE : CL_TRUE, 0, bytes, host_src, 0, nullptr, nullptr);
    return (err == CL_SUCCESS);
}

bool IntelUhdBackend::download(void *host_dst, const BackendBuffer &src, size_t bytes, bool async) {
    if (!initialized_ || !host_dst || bytes == 0) return false;
    const auto *buf = dynamic_cast<const IntelUhdBuffer*>(&src);
    if (!buf) return false;

    cl_int err = clEnqueueReadBuffer(queue_, buf->mem(), async ? CL_FALSE : CL_TRUE, 0, bytes, host_dst, 0, nullptr, nullptr);
    return (err == CL_SUCCESS);
}

bool IntelUhdBackend::copy(BackendBuffer &dst, const BackendBuffer &src, size_t bytes) {
    if (!initialized_ || bytes == 0) return false;
    auto *d = dynamic_cast<IntelUhdBuffer*>(&dst);
    const auto *s = dynamic_cast<const IntelUhdBuffer*>(&src);
    if (!d || !s) return false;

    cl_int err = clEnqueueCopyBuffer(queue_, s->mem(), d->mem(), 0, 0, bytes, 0, nullptr, nullptr);
    return (err == CL_SUCCESS);
}

void IntelUhdBackend::synchronize() {
    if (queue_) clFinish(queue_);
}

void IntelUhdBackend::rms_norm(BackendBuffer &out, BackendBuffer &x, BackendBuffer &weight, int64_t n, float eps) {
    (void)out; (void)x; (void)weight; (void)n; (void)eps;
}

void IntelUhdBackend::add_rms_norm(BackendBuffer &residual, BackendBuffer &branch, BackendBuffer &weight, BackendBuffer &norm_out, int64_t n, float eps) {
    (void)residual; (void)branch; (void)weight; (void)norm_out; (void)n; (void)eps;
}

void IntelUhdBackend::gemv_q4_0(BackendBuffer &dst, BackendBuffer &a, BackendBuffer &b, int64_t N, int64_t K) {
    (void)dst; (void)a; (void)b; (void)N; (void)K;
}

void IntelUhdBackend::gemv_q8_0(BackendBuffer &dst, BackendBuffer &a, BackendBuffer &b, int64_t N, int64_t K) {
    (void)dst; (void)a; (void)b; (void)N; (void)K;
}

void IntelUhdBackend::gemv_q4_0_fused_ffn(BackendBuffer &dst, BackendBuffer &a, BackendBuffer &gate, BackendBuffer &up, int64_t N, int64_t K) {
    (void)dst; (void)a; (void)gate; (void)up; (void)N; (void)K;
}

void IntelUhdBackend::rope(BackendBuffer &x, int64_t n_embd, int64_t n_head, int64_t pos, int64_t n_tokens) {
    (void)x; (void)n_embd; (void)n_head; (void)pos; (void)n_tokens;
}

void IntelUhdBackend::embed_lookup_q4_0(BackendBuffer &hidden, BackendBuffer &embd_table, int token_id, int64_t n_embd) {
    (void)hidden; (void)embd_table; (void)token_id; (void)n_embd;
}

void IntelUhdBackend::argmax(BackendBuffer &out_idx, BackendBuffer &logits, int64_t n) {
    (void)out_idx; (void)logits; (void)n;
}
