#include "cuda_backend.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <utility>

namespace
{
__device__ __forceinline__ float warp_sum(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

__device__ __forceinline__ float q4_scale(const unsigned char *block)
{
    const unsigned short bits = static_cast<unsigned short>(block[0]) |
                                (static_cast<unsigned short>(block[1]) << 8);
    return __half2float(*reinterpret_cast<const __half *>(&bits));
}

// Eight half-warps produce eight rows. The 16 lanes read one packed Q4 byte
// and two activation values each, keeping both weight and activation accesses
// contiguous. The input vector is served from L2 across rows.
__global__ void gemv_q4_0_kernel(float *dst, const float *a, const unsigned char *b, int N, int K)
{
    const int half_lane = threadIdx.x & 15;
    const int half_warp = threadIdx.x >> 4;
    const int row = blockIdx.x * 8 + half_warp;
    const int blocks = K / 32;
    if (row >= N) return;
    const unsigned int active = __activemask();
    float sum = 0.f;

    for (int blk = 0; blk < blocks; ++blk)
    {
        const unsigned char *q = b + (static_cast<size_t>(row) * blocks + blk) * 18;
        const unsigned char packed = q[2 + half_lane];
        const float a0 = a[blk * 32 + half_lane];
        const float a1 = a[blk * 32 + half_lane + 16];
        const float d = q4_scale(q);
        sum = fmaf(a0, (static_cast<float>(packed & 15) - 8.f) * d, sum);
        sum = fmaf(a1, (static_cast<float>(packed >> 4) - 8.f) * d, sum);
    }

    for (int offset = 8; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(active, sum, offset, 16);
    if (half_lane == 0) dst[row] = sum;
}

__global__ void gemv_q8_0_kernel(float *dst, const float *a, const unsigned char *b, int N, int K)
{
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x;
    const int blocks = K / 32;
    float sum = 0.f;
    for (int blk = lane; blk < blocks; blk += 32)
    {
        const unsigned char *q = b + (static_cast<size_t>(row) * blocks + blk) * 34;
        const signed char *values = reinterpret_cast<const signed char *>(q + 2);
        float dot = 0.f;
#pragma unroll
        for (int i = 0; i < 32; ++i) dot = fmaf(a[blk * 32 + i], static_cast<float>(values[i]), dot);
        sum += dot * q4_scale(q);
    }
    sum = warp_sum(sum);
    if (lane == 0 && row < N) dst[row] = sum;
}

__global__ void fused_ffn_q4_kernel(float *dst, const float *a, const unsigned char *gate,
                                    const unsigned char *up, int N, int K)
{
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x;
    const int blocks = K / 32;
    float gate_sum = 0.f, up_sum = 0.f;
    for (int blk = lane; blk < blocks; blk += 32)
    {
        const float *av = a + blk * 32;
        const unsigned char *g = gate + (static_cast<size_t>(row) * blocks + blk) * 18;
        const unsigned char *u = up + (static_cast<size_t>(row) * blocks + blk) * 18;
        float sum_a = 0.f, gd = 0.f, ud = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            const float a0 = av[i], a1 = av[i + 16];
            sum_a += a0 + a1;
            gd = fmaf(a0, static_cast<float>(g[2 + i] & 15), gd);
            gd = fmaf(a1, static_cast<float>(g[2 + i] >> 4), gd);
            ud = fmaf(a0, static_cast<float>(u[2 + i] & 15), ud);
            ud = fmaf(a1, static_cast<float>(u[2 + i] >> 4), ud);
        }
        gate_sum += (gd - 8.f * sum_a) * q4_scale(g);
        up_sum += (ud - 8.f * sum_a) * q4_scale(u);
    }
    gate_sum = warp_sum(gate_sum);
    up_sum = warp_sum(up_sum);
    if (lane == 0 && row < N) dst[row] = (gate_sum / (1.f + expf(-gate_sum))) * up_sum;
}

__global__ void rms_norm_kernel(float *out, const float *x, const float *weight, int n, float eps)
{
    __shared__ float reduction[256];
    const int tid = threadIdx.x;
    float sum = 0.f;
    for (int i = tid; i < n; i += blockDim.x) sum = fmaf(x[i], x[i], sum);
    reduction[tid] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(reduction[0] / n + eps);
    for (int i = tid; i < n; i += blockDim.x) out[i] = x[i] * inv * weight[i];
}

__global__ void add_rms_norm_kernel(float *residual, const float *branch, const float *weight,
                                    float *out, int n, float eps)
{
    __shared__ float reduction[256];
    const int tid = threadIdx.x;
    float sum = 0.f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = residual[i] + branch[i];
        residual[i] = v;
        sum = fmaf(v, v, sum);
    }
    reduction[tid] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(reduction[0] / n + eps);
    for (int i = tid; i < n; i += blockDim.x) out[i] = residual[i] * inv * weight[i];
}

__global__ void rope_kernel(float *x, int n_embd, int n_head, int pos, int n_tokens)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int head_dim = n_embd / n_head;
    const int pairs = n_tokens * n_head * (head_dim / 2);
    if (i >= pairs) return;
    const int d = (i % (head_dim / 2)) * 2;
    const int base = (i / (head_dim / 2)) * head_dim;
    const float theta = (pos + i / (n_head * (head_dim / 2))) * powf(10000.f, -static_cast<float>(d) / head_dim);
    const float c = cosf(theta), s = sinf(theta), x0 = x[base + d], x1 = x[base + d + 1];
    x[base + d] = x0 * c - x1 * s;
    x[base + d + 1] = x0 * s + x1 * c;
}

__global__ void embed_q4_kernel(float *hidden, const unsigned char *table, int token, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int blocks = n / 32;
    const unsigned char *q = table + (static_cast<size_t>(token) * blocks + i / 32) * 18;
    const int j = i & 31;
    const unsigned char packed = q[2 + (j & 15)];
    const int v = (j < 16 ? (packed & 15) : (packed >> 4)) - 8;
    hidden[i] = static_cast<float>(v) * q4_scale(q);
}

__global__ void argmax_kernel(const float *values, int *result, int n)
{
    __shared__ float max_values[256];
    __shared__ int max_indices[256];
    const int tid = threadIdx.x;
    float best = -INFINITY;
    int index = 0;
    for (int i = tid; i < n; i += blockDim.x) if (values[i] > best) { best = values[i]; index = i; }
    max_values[tid] = best; max_indices[tid] = index;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (tid < stride && max_values[tid + stride] > max_values[tid]) {
            max_values[tid] = max_values[tid + stride]; max_indices[tid] = max_indices[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) *result = max_indices[0];
}
}

CudaBuffer::CudaBuffer(void *ptr, size_t bytes, MemoryTier tier) : ptr_(ptr), bytes_(bytes), tier_(tier) {}
CudaBuffer::~CudaBuffer() { if (ptr_) cudaFree(ptr_); }

CudaBackend::~CudaBackend()
{
    if (stream_) cudaStreamDestroy(stream_);
}

bool CudaBackend::check(cudaError_t result, const char *operation) const
{
    if (result == cudaSuccess) return true;
    std::fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(result));
    return false;
}

bool CudaBackend::valid_range(const BackendBuffer &buffer, size_t bytes, const char *operation) const
{
    if (const_cast<BackendBuffer &>(buffer).raw_handle() && bytes <= buffer.size()) return true;
    std::fprintf(stderr, "CUDA buffer range error in %s: requested %zu, available %zu\n", operation, bytes, buffer.size());
    return false;
}

bool CudaBackend::initialize()
{
    int count = 0;
    if (!check(cudaGetDeviceCount(&count), "cudaGetDeviceCount") || count == 0) return false;
    if (!check(cudaSetDevice(device_), "cudaSetDevice")) return false;
    cudaDeviceProp prop{};
    if (!check(cudaGetDeviceProperties(&prop, device_), "cudaGetDeviceProperties")) return false;
    name_ = prop.name;
    if (!check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate")) return false;
    initialized_ = true;
    return true;
}

DeviceStats CudaBackend::query_stats()
{
    DeviceStats stats{};
    cudaDeviceProp prop{};
    size_t free_bytes = 0, total_bytes = 0;
    cudaGetDeviceProperties(&prop, device_);
    cudaMemGetInfo(&free_bytes, &total_bytes);
    stats.device_name = name_; stats.type = type(); stats.total_memory_bytes = total_bytes;
    stats.free_memory_bytes = free_bytes; stats.max_alloc_bytes = free_bytes; stats.compute_units = prop.multiProcessorCount;
    stats.memory_bandwidth_gbs = 2.0 * prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8.0) / 1.0e9;
    stats.profile.device_name = name_; stats.profile.type = type(); stats.profile.total_memory_bytes = total_bytes;
    stats.profile.free_memory_bytes = free_bytes; stats.profile.max_alloc_bytes = free_bytes;
    stats.profile.local_mem_bytes = prop.sharedMemPerBlock; stats.profile.max_workgroup_size = prop.maxThreadsPerBlock;
    stats.profile.compute_units = prop.multiProcessorCount; stats.profile.subgroup_size = prop.warpSize;
    stats.profile.fp16_supported = prop.major >= 5; stats.profile.subgroups_supported = true;
    stats.profile.memory_bandwidth_gbs = stats.memory_bandwidth_gbs;
    return stats;
}

std::unique_ptr<BackendBuffer> CudaBackend::allocate(size_t bytes, MemoryTier tier)
{
    void *ptr = nullptr;
    if (!check(cudaMalloc(&ptr, bytes), "cudaMalloc")) return nullptr;
    return std::make_unique<CudaBuffer>(ptr, bytes, tier);
}

bool CudaBackend::upload(BackendBuffer &dst, const void *src, size_t bytes, bool async)
{
    if (!valid_range(dst, bytes, "upload")) return false;
    cudaError_t result = cudaMemcpyAsync(dst.raw_handle(), src, bytes, cudaMemcpyHostToDevice, stream_);
    return check(result, "cudaMemcpyAsync H2D") && (async || check(cudaStreamSynchronize(stream_), "upload sync"));
}

bool CudaBackend::download(void *dst, const BackendBuffer &src, size_t bytes, bool async)
{
    if (!valid_range(src, bytes, "download")) return false;
    cudaError_t result = cudaMemcpyAsync(dst, const_cast<BackendBuffer &>(src).raw_handle(), bytes, cudaMemcpyDeviceToHost, stream_);
    return check(result, "cudaMemcpyAsync D2H") && (async || check(cudaStreamSynchronize(stream_), "download sync"));
}

bool CudaBackend::copy(BackendBuffer &dst, const BackendBuffer &src, size_t bytes)
{
    if (!valid_range(dst, bytes, "copy dst") || !valid_range(src, bytes, "copy src")) return false;
    return check(cudaMemcpyAsync(dst.raw_handle(), const_cast<BackendBuffer &>(src).raw_handle(), bytes, cudaMemcpyDeviceToDevice, stream_), "cudaMemcpyAsync D2D");
}

void CudaBackend::synchronize() { check(cudaStreamSynchronize(stream_), "synchronize"); }

void CudaBackend::gemv_q4_0(BackendBuffer &dst, BackendBuffer &a, BackendBuffer &b, int64_t N, int64_t K)
{
    gemv_q4_0_kernel<<<(N + 7) / 8, 128, 0, stream_>>>(static_cast<float *>(dst.raw_handle()), static_cast<const float *>(a.raw_handle()), static_cast<const unsigned char *>(b.raw_handle()), static_cast<int>(N), static_cast<int>(K));
}

void CudaBackend::gemv_q8_0(BackendBuffer &dst, BackendBuffer &a, BackendBuffer &b, int64_t N, int64_t K)
{
    gemv_q8_0_kernel<<<N, 32, 0, stream_>>>(static_cast<float *>(dst.raw_handle()), static_cast<const float *>(a.raw_handle()), static_cast<const unsigned char *>(b.raw_handle()), static_cast<int>(N), static_cast<int>(K));
}

void CudaBackend::gemv_q4_0_fused_ffn(BackendBuffer &dst, BackendBuffer &a, BackendBuffer &gate, BackendBuffer &up, int64_t N, int64_t K)
{
    fused_ffn_q4_kernel<<<N, 32, 0, stream_>>>(static_cast<float *>(dst.raw_handle()), static_cast<const float *>(a.raw_handle()), static_cast<const unsigned char *>(gate.raw_handle()), static_cast<const unsigned char *>(up.raw_handle()), static_cast<int>(N), static_cast<int>(K));
}

void CudaBackend::rms_norm(BackendBuffer &out, BackendBuffer &x, BackendBuffer &weight, int64_t n, float eps)
{
    rms_norm_kernel<<<1, 256, 0, stream_>>>(static_cast<float *>(out.raw_handle()), static_cast<const float *>(x.raw_handle()), static_cast<const float *>(weight.raw_handle()), static_cast<int>(n), eps);
}

void CudaBackend::add_rms_norm(BackendBuffer &residual, BackendBuffer &branch, BackendBuffer &weight, BackendBuffer &out, int64_t n, float eps)
{
    add_rms_norm_kernel<<<1, 256, 0, stream_>>>(static_cast<float *>(residual.raw_handle()), static_cast<const float *>(branch.raw_handle()), static_cast<const float *>(weight.raw_handle()), static_cast<float *>(out.raw_handle()), static_cast<int>(n), eps);
}

void CudaBackend::rope(BackendBuffer &x, int64_t n_embd, int64_t n_head, int64_t pos, int64_t n_tokens)
{
    const int64_t count = n_tokens * n_head * (n_embd / n_head / 2);
    rope_kernel<<<(count + 255) / 256, 256, 0, stream_>>>(static_cast<float *>(x.raw_handle()), static_cast<int>(n_embd), static_cast<int>(n_head), static_cast<int>(pos), static_cast<int>(n_tokens));
}

void CudaBackend::embed_lookup_q4_0(BackendBuffer &hidden, BackendBuffer &table, int token, int64_t n)
{
    embed_q4_kernel<<<(n + 255) / 256, 256, 0, stream_>>>(static_cast<float *>(hidden.raw_handle()), static_cast<const unsigned char *>(table.raw_handle()), token, static_cast<int>(n));
}

void CudaBackend::argmax(BackendBuffer &out, BackendBuffer &logits, int64_t n)
{
    argmax_kernel<<<1, 256, 0, stream_>>>(static_cast<const float *>(logits.raw_handle()), static_cast<int *>(out.raw_handle()), static_cast<int>(n));
}
