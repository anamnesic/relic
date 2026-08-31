#include "intel_uhd_backend.h"
#include <cstdio>
#include <cstring>
#include "../embedded_kernels.h"

IntelUhdBuffer::IntelUhdBuffer(cl_mem mem, size_t size, MemoryTier tier, void *host_ptr)
    : mem_(mem), size_(size), tier_(tier), host_ptr_(host_ptr) {}

IntelUhdBuffer::~IntelUhdBuffer()
{
    if (mem_)
    {
        clReleaseMemObject(mem_);
        mem_ = nullptr;
    }
}

IntelUhdBackend::IntelUhdBackend()
    : name_("Intel UHD Graphics (Shared Memory OpenCL 3.0)"),
      platform_(nullptr), device_(nullptr), context_(nullptr),
      queue_(nullptr), program_(nullptr), initialized_(false) {}

IntelUhdBackend::~IntelUhdBackend()
{
    for (auto &kv : kernel_cache_)
    {
        if (kv.second)
            clReleaseKernel(kv.second);
    }
    kernel_cache_.clear();
    if (program_)
        clReleaseProgram(program_);
    if (queue_)
        clReleaseCommandQueue(queue_);
    if (context_)
        clReleaseContext(context_);
}

bool IntelUhdBackend::initialize()
{
    if (initialized_)
        return true;

    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    if (num_platforms == 0)
        return false;

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    for (auto p : platforms)
    {
        cl_uint num_devs = 0;
        clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devs);
        if (num_devs == 0)
            continue;

        std::vector<cl_device_id> devs(num_devs);
        clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, num_devs, devs.data(), nullptr);

        for (auto d : devs)
        {
            char dev_name[256] = {0};
            clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);
            if (strstr(dev_name, "Intel") != nullptr || strstr(dev_name, "UHD") != nullptr || strstr(dev_name, "Iris") != nullptr)
            {
                platform_ = p;
                device_ = d;
                name_ = dev_name;
                break;
            }
        }
        if (device_)
            break;
    }

    if (!device_)
        return false;

    cl_int err;
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || !context_)
        return false;

    queue_ = clCreateCommandQueue(context_, device_, 0, &err);
    if (err != CL_SUCCESS || !queue_)
        return false;

    const char *src = (const char *)caicos_kernel_embedded_bytes;
    size_t src_len = strlen(src);
    program_ = clCreateProgramWithSource(context_, 1, &src, &src_len, &err);
    if (err == CL_SUCCESS)
    {
        clBuildProgram(program_, 1, &device_, "-cl-mad-enable -cl-fast-relaxed-math", nullptr, nullptr);
    }

    initialized_ = true;
    return true;
}

DeviceStats IntelUhdBackend::query_stats()
{
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

    if (device_)
    {
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

std::unique_ptr<BackendBuffer> IntelUhdBackend::allocate(size_t bytes, MemoryTier tier)
{
    if (!initialized_ || bytes == 0)
        return nullptr;

    cl_mem_flags flags = CL_MEM_READ_WRITE;
    if (tier == MemoryTier::TIER1_SHARED_IGPU || tier == MemoryTier::TIER2_HOST_PINNED_RAM)
    {
        flags |= CL_MEM_ALLOC_HOST_PTR;
    }

    cl_int err;
    cl_mem mem = clCreateBuffer(context_, flags, bytes, nullptr, &err);
    if (err != CL_SUCCESS || !mem)
        return nullptr;

    return std::make_unique<IntelUhdBuffer>(mem, bytes, tier);
}

bool IntelUhdBackend::upload(BackendBuffer &dst, const void *host_src, size_t bytes, bool async)
{
    if (!initialized_ || !host_src || bytes == 0)
        return false;
    auto *buf = dynamic_cast<IntelUhdBuffer *>(&dst);
    if (!buf)
        return false;

    cl_int err = clEnqueueWriteBuffer(queue_, buf->mem(), async ? CL_FALSE : CL_TRUE, 0, bytes, host_src, 0, nullptr, nullptr);
    return (err == CL_SUCCESS);
}

bool IntelUhdBackend::download(void *host_dst, const BackendBuffer &src, size_t bytes, bool async)
{
    if (!initialized_ || !host_dst || bytes == 0)
        return false;
    const auto *buf = dynamic_cast<const IntelUhdBuffer *>(&src);
    if (!buf)
        return false;

    cl_int err = clEnqueueReadBuffer(queue_, buf->mem(), async ? CL_FALSE : CL_TRUE, 0, bytes, host_dst, 0, nullptr, nullptr);
    return (err == CL_SUCCESS);
}

bool IntelUhdBackend::copy(BackendBuffer &dst, const BackendBuffer &src, size_t bytes)
{
    if (!initialized_ || bytes == 0)
        return false;
    auto *d = dynamic_cast<IntelUhdBuffer *>(&dst);
    const auto *s = dynamic_cast<const IntelUhdBuffer *>(&src);
    if (!d || !s)
        return false;

    cl_int err = clEnqueueCopyBuffer(queue_, s->mem(), d->mem(), 0, 0, bytes, 0, nullptr, nullptr);
    return (err == CL_SUCCESS);
}

void IntelUhdBackend::synchronize()
{
    if (queue_)
        clFinish(queue_);
}

cl_kernel IntelUhdBackend::get_kernel(const char *name)
{
    if (!program_)
        return nullptr;
    auto it = kernel_cache_.find(name);
    if (it != kernel_cache_.end())
        return it->second;

    cl_int err;
    cl_kernel k = clCreateKernel(program_, name, &err);
    if (err == CL_SUCCESS && k)
    {
        kernel_cache_[name] = k;
        return k;
    }
    return nullptr;
}

void IntelUhdBackend::rms_norm(BackendBuffer &out, BackendBuffer &x, BackendBuffer &weight, int64_t n, float eps)
{
    cl_kernel k = get_kernel("rms_norm_f32");
    if (!k || !queue_)
        return;

    auto *b_out = dynamic_cast<IntelUhdBuffer *>(&out);
    auto *b_x = dynamic_cast<IntelUhdBuffer *>(&x);
    auto *b_w = dynamic_cast<IntelUhdBuffer *>(&weight);
    if (!b_out || !b_x || !b_w)
        return;

    cl_mem m_out = b_out->mem();
    cl_mem m_x = b_x->mem();
    cl_mem m_w = b_w->mem();
    cl_int nn = (cl_int)n;
    cl_float e = (cl_float)eps;

    clSetKernelArg(k, 0, sizeof(cl_mem), &m_out);
    clSetKernelArg(k, 1, sizeof(cl_mem), &m_x);
    clSetKernelArg(k, 2, sizeof(cl_mem), &m_w);
    clSetKernelArg(k, 3, sizeof(cl_int), &nn);
    clSetKernelArg(k, 4, sizeof(cl_float), &e);

    size_t global = 1;
    clEnqueueNDRangeKernel(queue_, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
}

void IntelUhdBackend::add_rms_norm(BackendBuffer &residual, BackendBuffer &branch, BackendBuffer &weight, BackendBuffer &norm_out, int64_t n, float eps)
{
    cl_kernel k = get_kernel("add_rms_norm_f32");
    if (!k || !queue_)
        return;

    auto *b_res = dynamic_cast<IntelUhdBuffer *>(&residual);
    auto *b_bra = dynamic_cast<IntelUhdBuffer *>(&branch);
    auto *b_w = dynamic_cast<IntelUhdBuffer *>(&weight);
    auto *b_out = dynamic_cast<IntelUhdBuffer *>(&norm_out);
    if (!b_res || !b_bra || !b_w || !b_out)
        return;

    cl_mem m_res = b_res->mem();
    cl_mem m_bra = b_bra->mem();
    cl_mem m_w = b_w->mem();
    cl_mem m_out = b_out->mem();
    cl_int nn = (cl_int)n;
    cl_float e = (cl_float)eps;

    clSetKernelArg(k, 0, sizeof(cl_mem), &m_res);
    clSetKernelArg(k, 1, sizeof(cl_mem), &m_bra);
    clSetKernelArg(k, 2, sizeof(cl_mem), &m_w);
    clSetKernelArg(k, 3, sizeof(cl_mem), &m_out);
    clSetKernelArg(k, 4, sizeof(cl_int), &nn);
    clSetKernelArg(k, 5, sizeof(cl_float), &e);

    size_t global = 1;
    clEnqueueNDRangeKernel(queue_, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
}

void IntelUhdBackend::gemv_q4_0(BackendBuffer &dst, BackendBuffer &a, BackendBuffer &b, int64_t N, int64_t K)
{
    cl_kernel k = get_kernel("gemv_q4_0");
    if (!k || !queue_)
        return;

    auto *b_dst = dynamic_cast<IntelUhdBuffer *>(&dst);
    auto *b_a = dynamic_cast<IntelUhdBuffer *>(&a);
    auto *b_b = dynamic_cast<IntelUhdBuffer *>(&b);
    if (!b_dst || !b_a || !b_b)
        return;

    cl_mem m_dst = b_dst->mem();
    cl_mem m_a = b_a->mem();
    cl_mem m_b = b_b->mem();
    cl_int n = (cl_int)N, kk = (cl_int)K;

    clSetKernelArg(k, 0, sizeof(cl_mem), &m_a);
    clSetKernelArg(k, 1, sizeof(cl_mem), &m_b);
    clSetKernelArg(k, 2, sizeof(cl_mem), &m_dst);
    clSetKernelArg(k, 3, sizeof(cl_int), &n);
    clSetKernelArg(k, 4, sizeof(cl_int), &kk);

    size_t local = 32;
    size_t n_groups = ((size_t)N + 15) / 16;
    size_t global = n_groups * local;
    clEnqueueNDRangeKernel(queue_, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
}

void IntelUhdBackend::gemv_q8_0(BackendBuffer &dst, BackendBuffer &a, BackendBuffer &b, int64_t N, int64_t K)
{
    cl_kernel k = get_kernel("gemv_q8_0");
    if (!k || !queue_)
        return;

    auto *b_dst = dynamic_cast<IntelUhdBuffer *>(&dst);
    auto *b_a = dynamic_cast<IntelUhdBuffer *>(&a);
    auto *b_b = dynamic_cast<IntelUhdBuffer *>(&b);
    if (!b_dst || !b_a || !b_b)
        return;

    cl_mem m_dst = b_dst->mem();
    cl_mem m_a = b_a->mem();
    cl_mem m_b = b_b->mem();
    cl_int n = (cl_int)N, kk = (cl_int)K;

    clSetKernelArg(k, 0, sizeof(cl_mem), &m_a);
    clSetKernelArg(k, 1, sizeof(cl_mem), &m_b);
    clSetKernelArg(k, 2, sizeof(cl_mem), &m_dst);
    clSetKernelArg(k, 3, sizeof(cl_int), &n);
    clSetKernelArg(k, 4, sizeof(cl_int), &kk);

    size_t local = 32;
    size_t n_groups = ((size_t)N + 3) / 4;
    size_t global = n_groups * local;
    clEnqueueNDRangeKernel(queue_, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
}

void IntelUhdBackend::gemv_q4_0_fused_ffn(BackendBuffer &dst, BackendBuffer &a, BackendBuffer &gate, BackendBuffer &up, int64_t N, int64_t K)
{
    cl_kernel k = get_kernel("gemv_q4_0_ffn_swiglu");
    if (!k || !queue_)
        return;

    auto *b_dst = dynamic_cast<IntelUhdBuffer *>(&dst);
    auto *b_a = dynamic_cast<IntelUhdBuffer *>(&a);
    auto *b_g = dynamic_cast<IntelUhdBuffer *>(&gate);
    auto *b_u = dynamic_cast<IntelUhdBuffer *>(&up);
    if (!b_dst || !b_a || !b_g || !b_u)
        return;

    cl_mem m_dst = b_dst->mem();
    cl_mem m_a = b_a->mem();
    cl_mem m_g = b_g->mem();
    cl_mem m_u = b_u->mem();
    cl_int n = (cl_int)N, kk = (cl_int)K;

    clSetKernelArg(k, 0, sizeof(cl_mem), &m_a);
    clSetKernelArg(k, 1, sizeof(cl_mem), &m_g);
    clSetKernelArg(k, 2, sizeof(cl_mem), &m_u);
    clSetKernelArg(k, 3, sizeof(cl_mem), &m_dst);
    clSetKernelArg(k, 4, sizeof(cl_int), &n);
    clSetKernelArg(k, 5, sizeof(cl_int), &kk);

    size_t local = 32;
    size_t n_groups = ((size_t)N + 7) / 8;
    size_t global = n_groups * local;
    clEnqueueNDRangeKernel(queue_, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
}

void IntelUhdBackend::rope(BackendBuffer &x, int64_t n_embd, int64_t n_head, int64_t pos, int64_t n_tokens)
{
    cl_kernel k = get_kernel("rope_f32");
    if (!k || !queue_)
        return;

    auto *b_x = dynamic_cast<IntelUhdBuffer *>(&x);
    if (!b_x)
        return;

    cl_mem m_x = b_x->mem();
    cl_int ne = (cl_int)n_embd, nh = (cl_int)n_head, p = (cl_int)pos, nt = (cl_int)n_tokens;

    clSetKernelArg(k, 0, sizeof(cl_mem), &m_x);
    clSetKernelArg(k, 1, sizeof(cl_int), &ne);
    clSetKernelArg(k, 2, sizeof(cl_int), &nh);
    clSetKernelArg(k, 3, sizeof(cl_int), &p);
    clSetKernelArg(k, 4, sizeof(cl_int), &nt);

    int head_dim = (int)(n_embd / n_head);
    size_t global[3] = {(size_t)n_tokens, (size_t)n_head, (size_t)(head_dim / 2)};
    clEnqueueNDRangeKernel(queue_, k, 3, nullptr, global, nullptr, 0, nullptr, nullptr);
}

void IntelUhdBackend::embed_lookup_q4_0(BackendBuffer &hidden, BackendBuffer &embd_table, int token_id, int64_t n_embd)
{
    cl_kernel k = get_kernel("embed_lookup_q4_0");
    if (!k || !queue_)
        return;

    auto *b_hid = dynamic_cast<IntelUhdBuffer *>(&hidden);
    auto *b_tbl = dynamic_cast<IntelUhdBuffer *>(&embd_table);
    if (!b_hid || !b_tbl)
        return;

    cl_mem m_hid = b_hid->mem();
    cl_mem m_tbl = b_tbl->mem();
    cl_int tok = (cl_int)token_id, ne = (cl_int)n_embd;

    clSetKernelArg(k, 0, sizeof(cl_mem), &m_hid);
    clSetKernelArg(k, 1, sizeof(cl_mem), &m_tbl);
    clSetKernelArg(k, 2, sizeof(cl_int), &tok);
    clSetKernelArg(k, 3, sizeof(cl_int), &ne);

    size_t local = 64;
    size_t global = (size_t)(n_embd / 4);
    if (global % local != 0)
    {
        global = ((global + local - 1) / local) * local;
    }
    clEnqueueNDRangeKernel(queue_, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
}

void IntelUhdBackend::argmax(BackendBuffer &out_idx, BackendBuffer &logits, int64_t n)
{
    cl_kernel k = get_kernel("argmax_f32");
    if (!k || !queue_)
        return;

    auto *b_idx = dynamic_cast<IntelUhdBuffer *>(&out_idx);
    auto *b_log = dynamic_cast<IntelUhdBuffer *>(&logits);
    if (!b_idx || !b_log)
        return;

    cl_mem m_idx = b_idx->mem();
    cl_mem m_log = b_log->mem();
    cl_int nn = (cl_int)n;

    clSetKernelArg(k, 0, sizeof(cl_mem), &m_log);
    clSetKernelArg(k, 1, sizeof(cl_mem), &m_idx);
    clSetKernelArg(k, 2, sizeof(cl_int), &nn);

    size_t local = 256;
    size_t global = 256;
    clEnqueueNDRangeKernel(queue_, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
}
