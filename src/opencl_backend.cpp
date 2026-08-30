#include "opencl_backend.h"
#include "embedded_kernels.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>

#define CL_CHECK(err, msg) do { \
    cl_int err_ = (err); \
    if (err_ != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error %d at %s:%d: %s\n", err_, __FILE__, __LINE__, msg); \
        fflush(stderr); \
        return false; \
    } \
} while(0)

#define CL_CHECK_VOID(err, msg) do { \
    cl_int err_ = (err); \
    if (err_ != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error %d at %s:%d: %s\n", err_, __FILE__, __LINE__, msg); \
        fflush(stderr); \
        return; \
    } \
} while(0)

static std::string read_file(const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs) return "";
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

const char *caicos_kernel_source = nullptr;
static std::string g_kernel_source;

bool OpenClBackend::init(int platform_idx, int device_idx) {
    cl_uint n_platforms;
    if (clGetPlatformIDs(0, nullptr, &n_platforms) != CL_SUCCESS || n_platforms == 0) {
        fprintf(stderr, "No OpenCL platforms found\n");
        return false;
    }

    std::vector<cl_platform_id> platforms(n_platforms);
    clGetPlatformIDs(n_platforms, platforms.data(), nullptr);

    if (platform_idx < 0) {
        // Auto-detect dedicated GPU platform (e.g. NVIDIA CUDA)
        for (cl_uint p = 0; p < n_platforms; p++) {
            char pbuf[512];
            clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sizeof(pbuf), pbuf, nullptr);
            if (strstr(pbuf, "NVIDIA") || strstr(pbuf, "CUDA") || strstr(pbuf, "AMD")) {
                platform_idx = (int)p;
                break;
            }
        }
        if (platform_idx < 0) platform_idx = 0;
    }

    if ((int)n_platforms <= platform_idx) {
        fprintf(stderr, "Platform index %d out of range (%u platforms)\n", platform_idx, n_platforms);
        return false;
    }

    dev.platform = platforms[platform_idx];

    char buf[1024];
    clGetPlatformInfo(dev.platform, CL_PLATFORM_NAME, sizeof(buf), buf, nullptr);
    dev.platform_name = buf;
    fprintf(stdout, "Platform: %s\n", buf);

    cl_uint n_devices;
    if (clGetDeviceIDs(dev.platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &n_devices) != CL_SUCCESS || n_devices == 0) {
        fprintf(stderr, "No GPU devices found on platform\n");
        return false;
    }

    // Try GPU first, then CPU
    std::vector<cl_device_id> devices(n_devices);
    clGetDeviceIDs(dev.platform, CL_DEVICE_TYPE_GPU, n_devices, devices.data(), nullptr);

    cl_device_id target_device = nullptr;

    // Enumerate all devices
    for (cl_uint i = 0; i < n_devices; i++) {
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(buf), buf, nullptr);
        cl_device_type dtype;
        clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(dtype), &dtype, nullptr);
        fprintf(stdout, "  Device %d: %s (%s)\n", i, buf,
                dtype == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU");

        if ((int)i == device_idx) {
            target_device = devices[i];
        }
    }

    if (!target_device) {
        fprintf(stderr, "Device index %d not found\n", device_idx);
        return false;
    }

    dev.device = target_device;

    clGetDeviceInfo(dev.device, CL_DEVICE_NAME, sizeof(buf), buf, nullptr);
    dev.name = buf;

    clGetDeviceInfo(dev.device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(size_t), &dev.global_mem, nullptr);
    clGetDeviceInfo(dev.device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(size_t), &dev.max_alloc, nullptr);
    clGetDeviceInfo(dev.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &dev.max_wg_size, nullptr);
    clGetDeviceInfo(dev.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &dev.compute_units, nullptr);

    cl_int align;
    clGetDeviceInfo(dev.device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_int), &align, nullptr);
    dev.alignment = align / 8;

    // Check OpenCL C version
    clGetDeviceInfo(dev.device, CL_DEVICE_OPENCL_C_VERSION, sizeof(buf), buf, nullptr);
    fprintf(stdout, "  OpenCL C: %s\n", buf);

    int major = 0, minor = 0;
    if (sscanf(buf, "OpenCL C %d.%d", &major, &minor) >= 2) {
        dev.opencl_c_major = major;
        dev.opencl_c_minor = minor;
    }

    // Check FP16
    clGetDeviceInfo(dev.device, CL_DEVICE_EXTENSIONS, sizeof(buf), buf, nullptr);
    dev.fp16 = strstr(buf, "cl_khr_fp16") != nullptr;
    fprintf(stdout, "  FP16 support: %s\n", dev.fp16 ? "yes" : "no");

    fprintf(stdout, "  Global mem: %zu MB\n", dev.global_mem / (1024 * 1024));
    fprintf(stdout, "  Max alloc: %zu MB\n", dev.max_alloc / (1024 * 1024));
    fprintf(stdout, "  Max WG size: %zu\n", dev.max_wg_size);
    fprintf(stdout, "  Compute units: %u\n", dev.compute_units);
    fprintf(stdout, "  Alignment: %d bytes\n", dev.alignment);

    // Create context
    cl_int err;
    dev.context = clCreateContext(nullptr, 1, &dev.device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create OpenCL context\n");
        return false;
    }

    // Create command queue (OpenCL 1.2)
    dev.queue = clCreateCommandQueue(dev.context, dev.device, 0, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create command queue\n");
        return false;
    }

    // Use embedded kernel source as primary
    g_kernel_source = (const char *)caicos_kernel_embedded_bytes;
    // Try external file as fallback (for development)
    std::string ext_src = read_file("kernels/kernels.cl");
    if (ext_src.empty()) ext_src = read_file("../kernels/kernels.cl");
    if (!ext_src.empty()) g_kernel_source = std::move(ext_src);
    caicos_kernel_source = g_kernel_source.c_str();

    initialized = true;
    return true;
}

void OpenClBackend::shutdown() {
    if (dev.queue) { clReleaseCommandQueue(dev.queue); dev.queue = nullptr; }
    if (dev.context) { clReleaseContext(dev.context); dev.context = nullptr; }
    initialized = false;
}

bool OpenClBackend::build_kernel(ClKernel &k, const char *source, const char *kname, const char *opts) {
    if (!initialized) return false;

    k.release();
    k.name = kname;

    cl_int err;
    const char *src = source;
    size_t src_len = strlen(source);
    k.program = clCreateProgramWithSource(dev.context, 1, &src, &src_len, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create program for %s\n", kname);
        return false;
    }

    err = clBuildProgram(k.program, 1, &dev.device, opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(k.program, dev.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size + 1);
        clGetProgramBuildInfo(k.program, dev.device, CL_PROGRAM_BUILD_LOG, log_size + 1, log.data(), nullptr);
        fprintf(stderr, "Build log for %s:\n%s\n", kname, log.data());
        k.release();
        return false;
    }

    k.kernel = clCreateKernel(k.program, kname, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create kernel %s\n", kname);
        k.release();
        return false;
    }

    return true;
}

//------------------------------------------------------------------------------
// High-level operations
//------------------------------------------------------------------------------

void OpenClBackend::rms_norm(ClBuffer &out, ClBuffer &x, ClBuffer &weight, int64_t n, int64_t rows) {
    static ClKernel knl;
    if (!knl.kernel) {
        build_kernel(knl, caicos_kernel_source, "rms_norm_f32");
    }

    clSetKernelArg(knl.kernel, 0, sizeof(cl_mem), &out.mem);
    clSetKernelArg(knl.kernel, 1, sizeof(cl_mem), &x.mem);
    clSetKernelArg(knl.kernel, 2, sizeof(cl_mem), &weight.mem);
    clSetKernelArg(knl.kernel, 3, sizeof(cl_int), &n);
    float eps = 1e-5f;
    clSetKernelArg(knl.kernel, 4, sizeof(float), &eps);

    size_t global = (size_t)rows;
    CL_CHECK_VOID(clEnqueueNDRangeKernel(dev.queue, knl.kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr), "rms_norm");
}

void OpenClBackend::matmul_f32(ClBuffer &dst, ClBuffer &a, ClBuffer &b, int64_t M, int64_t N, int64_t K) {
    static ClKernel knl;
    if (!knl.kernel) {
        build_kernel(knl, caicos_kernel_source, "matmul_f32");
    }

    clSetKernelArg(knl.kernel, 0, sizeof(cl_mem), &a.mem);
    clSetKernelArg(knl.kernel, 1, sizeof(cl_mem), &b.mem);
    clSetKernelArg(knl.kernel, 2, sizeof(cl_mem), &dst.mem);
    cl_int m = (cl_int)M, n = (cl_int)N, k = (cl_int)K;
    clSetKernelArg(knl.kernel, 3, sizeof(cl_int), &m);
    clSetKernelArg(knl.kernel, 4, sizeof(cl_int), &n);
    clSetKernelArg(knl.kernel, 5, sizeof(cl_int), &k);

    size_t global[2] = {(size_t)M, (size_t)N};
    CL_CHECK_VOID(clEnqueueNDRangeKernel(dev.queue, knl.kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr), "matmul_f32");
}

void OpenClBackend::matmul_f32_nt(ClBuffer &dst, ClBuffer &a, ClBuffer &b, int64_t M, int64_t N, int64_t K) {
    static ClKernel knl;
    if (!knl.kernel) {
        build_kernel(knl, caicos_kernel_source, "matmul_f32_nt");
    }

    clSetKernelArg(knl.kernel, 0, sizeof(cl_mem), &a.mem);
    clSetKernelArg(knl.kernel, 1, sizeof(cl_mem), &b.mem);
    clSetKernelArg(knl.kernel, 2, sizeof(cl_mem), &dst.mem);
    cl_int m = (cl_int)M, n = (cl_int)N, k = (cl_int)K;
    clSetKernelArg(knl.kernel, 3, sizeof(cl_int), &m);
    clSetKernelArg(knl.kernel, 4, sizeof(cl_int), &n);
    clSetKernelArg(knl.kernel, 5, sizeof(cl_int), &k);

    size_t global[2] = {(size_t)M, (size_t)N};
    CL_CHECK_VOID(clEnqueueNDRangeKernel(dev.queue, knl.kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr), "matmul_f32_nt");
}

void OpenClBackend::rope(ClBuffer &x, int64_t n_embd, int64_t n_head, int64_t pos, int64_t n_tokens) {
    static ClKernel knl;
    if (!knl.kernel) {
        build_kernel(knl, caicos_kernel_source, "rope_f32");
    }

    clSetKernelArg(knl.kernel, 0, sizeof(cl_mem), &x.mem);
    cl_int ne = (cl_int)n_embd, nh = (cl_int)n_head, p = (cl_int)pos, nt = (cl_int)n_tokens;
    clSetKernelArg(knl.kernel, 1, sizeof(cl_int), &ne);
    clSetKernelArg(knl.kernel, 2, sizeof(cl_int), &nh);
    clSetKernelArg(knl.kernel, 3, sizeof(cl_int), &p);
    clSetKernelArg(knl.kernel, 4, sizeof(cl_int), &nt);

    int head_dim = (int)(n_embd / n_head);
    size_t global[3] = {(size_t)n_tokens, (size_t)n_head, (size_t)(head_dim / 2)};
    CL_CHECK_VOID(clEnqueueNDRangeKernel(dev.queue, knl.kernel, 3, nullptr, global, nullptr, 0, nullptr, nullptr), "rope");
}

void OpenClBackend::softmax(ClBuffer &x, int64_t n, int64_t rows) {
    static ClKernel knl;
    if (!knl.kernel) {
        build_kernel(knl, caicos_kernel_source, "softmax_f32");
    }

    clSetKernelArg(knl.kernel, 0, sizeof(cl_mem), &x.mem);
    cl_int nn = (cl_int)n, nr = (cl_int)rows;
    clSetKernelArg(knl.kernel, 1, sizeof(cl_int), &nn);
    clSetKernelArg(knl.kernel, 2, sizeof(cl_int), &nr);

    size_t global = (size_t)rows;
    CL_CHECK_VOID(clEnqueueNDRangeKernel(dev.queue, knl.kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr), "softmax");
}

void OpenClBackend::silu(ClBuffer &out, ClBuffer &x, int64_t n) {
    static ClKernel knl;
    if (!knl.kernel) {
        build_kernel(knl, caicos_kernel_source, "silu_f32");
    }

    clSetKernelArg(knl.kernel, 0, sizeof(cl_mem), &out.mem);
    clSetKernelArg(knl.kernel, 1, sizeof(cl_mem), &x.mem);
    cl_int nn = (cl_int)n;
    clSetKernelArg(knl.kernel, 2, sizeof(cl_int), &nn);

    size_t global = (size_t)n;
    CL_CHECK_VOID(clEnqueueNDRangeKernel(dev.queue, knl.kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr), "silu");
}

void OpenClBackend::add(ClBuffer &dst, ClBuffer &a, ClBuffer &b, int64_t n) {
    static ClKernel knl;
    if (!knl.kernel) {
        build_kernel(knl, caicos_kernel_source, "add_f32");
    }

    clSetKernelArg(knl.kernel, 0, sizeof(cl_mem), &dst.mem);
    clSetKernelArg(knl.kernel, 1, sizeof(cl_mem), &a.mem);
    clSetKernelArg(knl.kernel, 2, sizeof(cl_mem), &b.mem);
    cl_int nn = (cl_int)n;
    clSetKernelArg(knl.kernel, 3, sizeof(cl_int), &nn);

    size_t global = (size_t)n;
    CL_CHECK_VOID(clEnqueueNDRangeKernel(dev.queue, knl.kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr), "add");
}

void OpenClBackend::mul(ClBuffer &dst, ClBuffer &a, ClBuffer &b, int64_t n) {
    static ClKernel knl;
    if (!knl.kernel) {
        build_kernel(knl, caicos_kernel_source, "mul_f32");
    }

    clSetKernelArg(knl.kernel, 0, sizeof(cl_mem), &dst.mem);
    clSetKernelArg(knl.kernel, 1, sizeof(cl_mem), &a.mem);
    clSetKernelArg(knl.kernel, 2, sizeof(cl_mem), &b.mem);
    cl_int nn = (cl_int)n;
    clSetKernelArg(knl.kernel, 3, sizeof(cl_int), &nn);

    size_t global = (size_t)n;
    CL_CHECK_VOID(clEnqueueNDRangeKernel(dev.queue, knl.kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr), "mul");
}

void OpenClBackend::copy(ClBuffer &dst, ClBuffer &src, int64_t n) {
    static ClKernel knl;
    if (!knl.kernel) {
        build_kernel(knl, caicos_kernel_source, "copy_f32");
    }

    clSetKernelArg(knl.kernel, 0, sizeof(cl_mem), &dst.mem);
    clSetKernelArg(knl.kernel, 1, sizeof(cl_mem), &src.mem);
    cl_int nn = (cl_int)n;
    clSetKernelArg(knl.kernel, 2, sizeof(cl_int), &nn);

    size_t global = (size_t)n;
    CL_CHECK_VOID(clEnqueueNDRangeKernel(dev.queue, knl.kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr), "copy");
}

void OpenClBackend::fill(ClBuffer &buf, float val, int64_t n) {
    static ClKernel knl;
    if (!knl.kernel) {
        build_kernel(knl, caicos_kernel_source, "fill_f32");
    }

    clSetKernelArg(knl.kernel, 0, sizeof(cl_mem), &buf.mem);
    clSetKernelArg(knl.kernel, 1, sizeof(float), &val);
    cl_int nn = (cl_int)n;
    clSetKernelArg(knl.kernel, 2, sizeof(cl_int), &nn);

    size_t global = (size_t)n;
    CL_CHECK_VOID(clEnqueueNDRangeKernel(dev.queue, knl.kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr), "fill");
}

void OpenClBackend::gemv_f32_nt(ClBuffer &dst, ClBuffer &a, ClBuffer &b, int64_t N, int64_t K) {
    static ClKernel knl;
    if (!knl.kernel) {
        build_kernel(knl, caicos_kernel_source, "gemv_f32_nt");
    }

    clSetKernelArg(knl.kernel, 0, sizeof(cl_mem), &a.mem);
    clSetKernelArg(knl.kernel, 1, sizeof(cl_mem), &b.mem);
    clSetKernelArg(knl.kernel, 2, sizeof(cl_mem), &dst.mem);
    cl_int n = (cl_int)N, k = (cl_int)K;
    clSetKernelArg(knl.kernel, 3, sizeof(cl_int), &n);
    clSetKernelArg(knl.kernel, 4, sizeof(cl_int), &k);

    size_t local = 32;
    size_t global = (size_t)N * local;
    CL_CHECK_VOID(clEnqueueNDRangeKernel(dev.queue, knl.kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr), "gemv_f32_nt");
}

void OpenClBackend::gemv_q8_0(ClBuffer &dst, ClBuffer &a, ClBuffer &b, int64_t N, int64_t K) {
    static ClKernel knl;
    if (!knl.kernel) {
        build_kernel(knl, caicos_kernel_source, "gemv_q8_0");
    }

    clSetKernelArg(knl.kernel, 0, sizeof(cl_mem), &a.mem);
    clSetKernelArg(knl.kernel, 1, sizeof(cl_mem), &b.mem);
    clSetKernelArg(knl.kernel, 2, sizeof(cl_mem), &dst.mem);
    cl_int n = (cl_int)N, k = (cl_int)K;
    clSetKernelArg(knl.kernel, 3, sizeof(cl_int), &n);
    clSetKernelArg(knl.kernel, 4, sizeof(cl_int), &k);

    size_t local = 32;
    size_t n_groups = ((size_t)N + 1) / 2;
    size_t global = n_groups * local;
    CL_CHECK_VOID(clEnqueueNDRangeKernel(dev.queue, knl.kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr), "gemv_q8_0");
}

void OpenClBackend::gemv_q4_0(ClBuffer &dst, ClBuffer &a, ClBuffer &b, int64_t N, int64_t K) {
    static ClKernel knl;
    if (!knl.kernel) {
        build_kernel(knl, caicos_kernel_source, "gemv_q4_0");
    }

    clSetKernelArg(knl.kernel, 0, sizeof(cl_mem), &a.mem);
    clSetKernelArg(knl.kernel, 1, sizeof(cl_mem), &b.mem);
    clSetKernelArg(knl.kernel, 2, sizeof(cl_mem), &dst.mem);
    cl_int n = (cl_int)N, k = (cl_int)K;
    clSetKernelArg(knl.kernel, 3, sizeof(cl_int), &n);
    clSetKernelArg(knl.kernel, 4, sizeof(cl_int), &k);

    size_t local = 32;
    size_t n_groups = ((size_t)N + 1) / 2;
    size_t global = n_groups * local;
    CL_CHECK_VOID(clEnqueueNDRangeKernel(dev.queue, knl.kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr), "gemv_q4_0");
}

void OpenClBackend::swiglu(ClBuffer &dst, ClBuffer &gate, ClBuffer &up, int64_t n) {
    static ClKernel knl;
    if (!knl.kernel) {
        build_kernel(knl, caicos_kernel_source, "swiglu_f32");
    }

    clSetKernelArg(knl.kernel, 0, sizeof(cl_mem), &dst.mem);
    clSetKernelArg(knl.kernel, 1, sizeof(cl_mem), &gate.mem);
    clSetKernelArg(knl.kernel, 2, sizeof(cl_mem), &up.mem);
    cl_int count = (cl_int)n;
    clSetKernelArg(knl.kernel, 3, sizeof(cl_int), &count);

    size_t global = (size_t)n;
    CL_CHECK_VOID(clEnqueueNDRangeKernel(dev.queue, knl.kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr), "swiglu_f32");
}

void OpenClBackend::add_rms_norm(ClBuffer &residual, ClBuffer &branch, ClBuffer &weight, ClBuffer &norm_out, int64_t n, float eps) {
    static ClKernel knl;
    if (!knl.kernel) {
        build_kernel(knl, caicos_kernel_source, "add_rms_norm_f32");
    }

    clSetKernelArg(knl.kernel, 0, sizeof(cl_mem), &residual.mem);
    clSetKernelArg(knl.kernel, 1, sizeof(cl_mem), &branch.mem);
    clSetKernelArg(knl.kernel, 2, sizeof(cl_mem), &weight.mem);
    clSetKernelArg(knl.kernel, 3, sizeof(cl_mem), &norm_out.mem);
    cl_int count = (cl_int)n;
    clSetKernelArg(knl.kernel, 4, sizeof(cl_int), &count);
    cl_float e = (cl_float)eps;
    clSetKernelArg(knl.kernel, 5, sizeof(cl_float), &e);

    size_t global = 1;
    CL_CHECK_VOID(clEnqueueNDRangeKernel(dev.queue, knl.kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr), "add_rms_norm_f32");
}

void OpenClBackend::qwen_conv1d(ClBuffer &conv_state, ClBuffer &conv_in, ClBuffer &weight, ClBuffer &conv_out, int64_t C) {
    static ClKernel knl;
    if (!knl.kernel) {
        build_kernel(knl, caicos_kernel_source, "qwen_conv1d_silu");
    }

    clSetKernelArg(knl.kernel, 0, sizeof(cl_mem), &conv_state.mem);
    clSetKernelArg(knl.kernel, 1, sizeof(cl_mem), &conv_in.mem);
    clSetKernelArg(knl.kernel, 2, sizeof(cl_mem), &weight.mem);
    clSetKernelArg(knl.kernel, 3, sizeof(cl_mem), &conv_out.mem);
    cl_int channels = (cl_int)C;
    clSetKernelArg(knl.kernel, 4, sizeof(cl_int), &channels);

    size_t global = (size_t)C;
    CL_CHECK_VOID(clEnqueueNDRangeKernel(dev.queue, knl.kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr), "qwen_conv1d_silu");
}

void OpenClBackend::qwen_deltanet(ClBuffer &ssm_state, ClBuffer &conv_out, ClBuffer &alpha, ClBuffer &beta, ClBuffer &delta_out, int64_t key_dim, int64_t qk_dim, int64_t linear_inner) {
    static ClKernel knl;
    if (!knl.kernel) {
        build_kernel(knl, caicos_kernel_source, "qwen_gated_deltanet_step");
    }

    clSetKernelArg(knl.kernel, 0, sizeof(cl_mem), &ssm_state.mem);
    clSetKernelArg(knl.kernel, 1, sizeof(cl_mem), &conv_out.mem);
    clSetKernelArg(knl.kernel, 2, sizeof(cl_mem), &alpha.mem);
    clSetKernelArg(knl.kernel, 3, sizeof(cl_mem), &beta.mem);
    clSetKernelArg(knl.kernel, 4, sizeof(cl_mem), &delta_out.mem);
    cl_int kd = (cl_int)key_dim;
    cl_int qkd = (cl_int)qk_dim;
    cl_int li = (cl_int)linear_inner;
    clSetKernelArg(knl.kernel, 5, sizeof(cl_int), &kd);
    clSetKernelArg(knl.kernel, 6, sizeof(cl_int), &qkd);
    clSetKernelArg(knl.kernel, 7, sizeof(cl_int), &li);

    size_t local = 128;
    size_t global = 16 * local; // 16 heads * 128 work items
    CL_CHECK_VOID(clEnqueueNDRangeKernel(dev.queue, knl.kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr), "qwen_gated_deltanet_step");
}

void OpenClBackend::qwen_attention_step(ClBuffer &q_buf, ClBuffer &k_buf, ClBuffer &v_buf, ClBuffer &k_cache, ClBuffer &v_cache, ClBuffer &attn_out, int64_t n_head, int64_t n_kv_head, int64_t head_dim, int64_t n_embd, int64_t pos, int64_t max_seq) {
    static ClKernel knl;
    if (!knl.kernel) {
        build_kernel(knl, caicos_kernel_source, "qwen_full_attention_step");
    }

    clSetKernelArg(knl.kernel, 0, sizeof(cl_mem), &q_buf.mem);
    clSetKernelArg(knl.kernel, 1, sizeof(cl_mem), &k_buf.mem);
    clSetKernelArg(knl.kernel, 2, sizeof(cl_mem), &v_buf.mem);
    clSetKernelArg(knl.kernel, 3, sizeof(cl_mem), &k_cache.mem);
    clSetKernelArg(knl.kernel, 4, sizeof(cl_mem), &v_cache.mem);
    clSetKernelArg(knl.kernel, 5, sizeof(cl_mem), &attn_out.mem);
    cl_int nh = (cl_int)n_head, nkv = (cl_int)n_kv_head, hd = (cl_int)head_dim, ne = (cl_int)n_embd, p = (cl_int)pos, ms = (cl_int)max_seq;
    clSetKernelArg(knl.kernel, 6, sizeof(cl_int), &nh);
    clSetKernelArg(knl.kernel, 7, sizeof(cl_int), &nkv);
    clSetKernelArg(knl.kernel, 8, sizeof(cl_int), &hd);
    clSetKernelArg(knl.kernel, 9, sizeof(cl_int), &ne);
    clSetKernelArg(knl.kernel, 10, sizeof(cl_int), &p);
    clSetKernelArg(knl.kernel, 11, sizeof(cl_int), &ms);

    size_t local = (size_t)head_dim;
    size_t global = (size_t)n_head * local;
    CL_CHECK_VOID(clEnqueueNDRangeKernel(dev.queue, knl.kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr), "qwen_full_attention_step");
}
