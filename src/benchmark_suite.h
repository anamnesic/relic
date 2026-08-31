#pragma once
#include <string>
#include <vector>
#include <memory>
#include "inference.h"
#include "planner/hardware_profile.h"

struct StageTimings
{
    double forward_wall_time_ms = 0.0; // Measured forward pass wall-clock duration per token
    double pcie_readback_ms = 0.0;     // Logits readback duration
    double sampling_ms = 0.0;          // Sampler (top-k / min-heap / greedy) duration
    double tokenizer_ms = 0.0;         // Piece decode / tokenizer duration
    double speculative_verification_ms = 0.0;
    double total_per_token_ms = 0.0;
};

struct BenchmarkRunResult
{
    int run_index = 0;
    bool is_warmup = false;
    double cold_start_ms = 0.0;
    double prompt_time_ms = 0.0;
    size_t prompt_tokens = 0;
    double prompt_tok_per_sec = 0.0;
    double decode_time_ms = 0.0;
    size_t decode_tokens = 0;
    double decode_tok_per_sec = 0.0;
    double total_time_ms = 0.0;
    StageTimings stage_timings;
};

struct BenchmarkSummaryStats
{
    double median_decode_tok_per_sec = 0.0;
    double p50_decode_tok_per_sec = 0.0;
    double p95_decode_tok_per_sec = 0.0;
    double stddev_decode_tok_per_sec = 0.0;

    double median_prompt_tok_per_sec = 0.0;
    double p50_prompt_tok_per_sec = 0.0;
    double p95_prompt_tok_per_sec = 0.0;
    double stddev_prompt_tok_per_sec = 0.0;

    double median_total_ms = 0.0;
    double p50_total_ms = 0.0;
    double p95_total_ms = 0.0;
    double stddev_total_ms = 0.0;

    StageTimings avg_stage_timings;
};

struct BenchmarkMetadata
{
    std::string model_name;
    std::string quantization;
    size_t context_length = 2048;
    std::string gpu_name;
    std::string driver_version;
    std::string opencl_version;
    size_t vram_total_mb = 0;
    size_t vram_free_mb = 0;
    std::string commit_sha = "relic-v0.2.0-2026";
    std::string build_flags = "MSVC C++23 OpenCL 1.2/3.0 Release -O2";
    std::string execution_device_config = "GTX 1650 + Intel UHD + CPU (Heterogeneous)";
};

struct BenchmarkSuiteResult
{
    BenchmarkMetadata metadata;
    std::vector<BenchmarkRunResult> runs;
    BenchmarkSummaryStats stats;
};

class BenchmarkSuite
{
public:
    static BenchmarkSuiteResult run_full_suite(
        InferenceEngine &engine,
        const std::string &prompt,
        int n_tokens,
        int num_runs = 5,
        int warmup_runs = 1,
        float temperature = 0.0f,
        int top_k = 40);

    static bool export_json(const BenchmarkSuiteResult &result, const std::string &filepath);
    static bool export_csv(const BenchmarkSuiteResult &result, const std::string &filepath);
    static void print_summary(const BenchmarkSuiteResult &result);
    static void print_comparative_table(const BenchmarkSuiteResult &result);
};
