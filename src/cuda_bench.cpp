#include "backends/cuda_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace
{
struct Result { double us; double gbs; float max_error; };

Result bench_q4(CudaBackend &cuda, int N, int K, int warmup, int iterations)
{
    const size_t weight_bytes = static_cast<size_t>(N) * (K / 32) * 18;
    std::vector<float> a(K);
    std::vector<unsigned char> b(weight_bytes);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> f(-1.f, 1.f);
    for (float &v : a) v = f(rng);
    for (size_t i = 0; i < weight_bytes; i += 18) {
        const unsigned short one = 0x3c00;
        b[i] = one & 0xff; b[i + 1] = one >> 8;
        for (int j = 2; j < 18; ++j) b[i + j] = static_cast<unsigned char>(rng());
    }
    auto da = cuda.allocate(a.size() * sizeof(float));
    auto db = cuda.allocate(b.size());
    auto dc = cuda.allocate(static_cast<size_t>(N) * sizeof(float));
    cuda.upload(*da, a.data(), a.size() * sizeof(float));
    cuda.upload(*db, b.data(), b.size());
    for (int i = 0; i < warmup; ++i) cuda.gemv_q4_0(*dc, *da, *db, N, K);
    cuda.synchronize();
    std::vector<float> output(N);
    cuda.download(output.data(), *dc, output.size() * sizeof(float));
    float max_error = 0.f;
    for (int row : {0, N / 2, N - 1}) {
        float reference = 0.f;
        for (int blk = 0; blk < K / 32; ++blk) {
            const unsigned char *q = b.data() + (static_cast<size_t>(row) * (K / 32) + blk) * 18;
            for (int i = 0; i < 16; ++i) {
                reference += a[blk * 32 + i] * (static_cast<int>(q[2 + i] & 15) - 8);
                reference += a[blk * 32 + i + 16] * (static_cast<int>(q[2 + i] >> 4) - 8);
            }
        }
        max_error = std::max(max_error, std::fabs(reference - output[row]));
    }
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) cuda.gemv_q4_0(*dc, *da, *db, N, K);
    cuda.synchronize();
    const auto end = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(end - start).count() / iterations;
    return {us, (weight_bytes + a.size() * sizeof(float) + static_cast<size_t>(N) * sizeof(float)) / us / 1000.0, max_error};
}
}

int main(int argc, char **argv)
{
    const int warmup = argc > 1 ? std::atoi(argv[1]) : 50;
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 200;
    CudaBackend cuda;
    if (!cuda.initialize()) return 1;
    const auto stats = cuda.query_stats();
    std::printf("Relic CUDA benchmark\nGPU: %s\nVRAM: %.0f MiB free / %.0f MiB total\nTheoretical bandwidth: %.1f GB/s\nWarmup: %d, iterations: %d\n",
                stats.device_name.c_str(), stats.free_memory_bytes / 1048576.0, stats.total_memory_bytes / 1048576.0,
                stats.memory_bandwidth_gbs, warmup, iterations);
    for (const auto shape : {std::pair{8192, 2048}, std::pair{6144, 2048}, std::pair{2048, 6144}}) {
        const Result r = bench_q4(cuda, shape.first, shape.second, warmup, iterations);
        std::printf("Q4_0 GEMV N=%d K=%d: %.3f ms, %.2f GB/s, max abs error %.6f\n",
                    shape.first, shape.second, r.us / 1000.0, r.gbs, r.max_error);
    }
    return 0;
}
