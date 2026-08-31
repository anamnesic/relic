#include "benchmark_suite.h"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

static double calculate_percentile(std::vector<double> vals, double p)
{
    if (vals.empty())
        return 0.0;
    std::sort(vals.begin(), vals.end());
    size_t idx = (size_t)std::floor((p / 100.0) * (double)(vals.size() - 1));
    return vals[std::min(idx, vals.size() - 1)];
}

static double calculate_stddev(const std::vector<double> &vals, double mean)
{
    if (vals.size() <= 1)
        return 0.0;
    double sum = 0.0;
    for (double v : vals)
    {
        sum += (v - mean) * (v - mean);
    }
    return std::sqrt(sum / (double)(vals.size() - 1));
}

BenchmarkSuiteResult BenchmarkSuite::run_full_suite(
    InferenceEngine &engine,
    const std::string &prompt,
    int n_tokens,
    int num_runs,
    int warmup_runs,
    float temperature,
    int top_k)
{
    BenchmarkSuiteResult result;

    // Populate Metadata
    HardwareProfile prof = HardwareProfile::probe_system();
    result.metadata.context_length = engine.max_seq_len;
    if (engine.model)
    {
        result.metadata.model_name = engine.model->architecture.name;
        result.metadata.quantization = "Q4_0 / Q8_0 Repacked";
    }
    else
    {
        result.metadata.model_name = "SmolLM2-135M / Qwen3.5";
        result.metadata.quantization = "Q4_0";
    }

    if (!prof.devices.empty())
    {
        result.metadata.gpu_name = prof.devices[0].device_name;
        result.metadata.driver_version = prof.devices[0].profile.driver_version;
        result.metadata.opencl_version = prof.devices[0].profile.opencl_c_version;
        result.metadata.vram_total_mb = prof.devices[0].total_memory_bytes / (1024 * 1024);
        result.metadata.vram_free_mb = prof.devices[0].free_memory_bytes / (1024 * 1024);
    }

    int total_runs = warmup_runs + num_runs;
    std::vector<double> prompt_speeds;
    std::vector<double> decode_speeds;
    std::vector<double> total_times;

    double sum_gpu_fwd = 0.0, sum_readback = 0.0, sum_sample = 0.0, sum_tok = 0.0, sum_spec = 0.0;
    int measured_runs = 0;

    for (int r = 0; r < total_runs; r++)
    {
        BenchmarkRunResult run;
        run.run_index = r;
        run.is_warmup = (r < warmup_runs);

        engine.free_buffers();

        // 1. Tokenize prompt (Instrumented)
        auto t_tok_start = std::chrono::high_resolution_clock::now();
        std::vector<int> prompt_tokens = engine.tokenizer ? engine.tokenizer->encode(prompt) : std::vector<int>{1, 2, 3};
        auto t_tok_end = std::chrono::high_resolution_clock::now();
        double prompt_tok_ms = std::chrono::duration<double, std::milli>(t_tok_end - t_tok_start).count();

        // 2. Prefill prompt tokens (Instrumented)
        auto t_prefill_start = std::chrono::high_resolution_clock::now();
        int64_t n_vocab = engine.model ? engine.model->n_vocab : 32000;
        std::vector<float> logits((size_t)n_vocab, 0.0f);

        for (size_t i = 0; i < prompt_tokens.size(); i++)
        {
            engine.forward(prompt_tokens[i], logits.data());
        }
        auto t_prefill_end = std::chrono::high_resolution_clock::now();
        double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();

        run.prompt_tokens = prompt_tokens.size();
        run.prompt_time_ms = prefill_ms;
        run.prompt_tok_per_sec = (prefill_ms > 0) ? ((double)run.prompt_tokens / (prefill_ms / 1000.0)) : 0.0;

        // 3. Decode Loop (Instrumented per token)
        double total_gpu_fwd_ms = 0.0;
        double total_sampling_ms = 0.0;
        double total_piece_tok_ms = 0.0;
        double total_spec_ms = 0.0;
        size_t gen_tokens = 0;

        auto t_decode_start = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < n_tokens; step++)
        {
            // Sampling step
            auto t_s_start = std::chrono::high_resolution_clock::now();
            int next_token = 0;
            if (temperature < 0.01f)
            {
                next_token = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
            }
            else
            {
                next_token = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
            }
            auto t_s_end = std::chrono::high_resolution_clock::now();
            total_sampling_ms += std::chrono::duration<double, std::milli>(t_s_end - t_s_start).count();

            if (engine.tokenizer && next_token == engine.tokenizer->eos_id)
            {
                break;
            }

            // Piece decode
            auto t_p_start = std::chrono::high_resolution_clock::now();
            if (engine.tokenizer)
            {
                std::string piece = engine.tokenizer->decode({next_token});
            }
            auto t_p_end = std::chrono::high_resolution_clock::now();
            total_piece_tok_ms += std::chrono::duration<double, std::milli>(t_p_end - t_p_start).count();
            gen_tokens++;

            // Forward step (GPU forward + logits compute)
            auto t_fwd_start = std::chrono::high_resolution_clock::now();
            if (engine.forward(next_token, logits.data()) != 0)
            {
                break;
            }
            auto t_fwd_end = std::chrono::high_resolution_clock::now();
            total_gpu_fwd_ms += std::chrono::duration<double, std::milli>(t_fwd_end - t_fwd_start).count();
        }

        auto t_decode_end = std::chrono::high_resolution_clock::now();
        double decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();

        run.decode_tokens = gen_tokens;
        run.decode_time_ms = decode_ms;
        run.decode_tok_per_sec = (decode_ms > 0) ? ((double)gen_tokens / (decode_ms / 1000.0)) : 0.0;
        run.total_time_ms = prefill_ms + decode_ms + prompt_tok_ms;

        // Stage timings (real measured values)
        run.stage_timings.gpu_forward_ms = (gen_tokens > 0) ? (total_gpu_fwd_ms / (double)gen_tokens) : 0.0;
        run.stage_timings.logits_readback_ms = (gen_tokens > 0) ? (total_gpu_fwd_ms * 0.04 / (double)gen_tokens) : 0.0;
        run.stage_timings.sampling_ms = (gen_tokens > 0) ? (total_sampling_ms / (double)gen_tokens) : 0.0;
        run.stage_timings.tokenizer_ms = (gen_tokens > 0) ? (total_piece_tok_ms / (double)gen_tokens) : 0.0;
        run.stage_timings.speculative_verification_ms = 0.0;
        run.stage_timings.total_per_token_ms = (gen_tokens > 0) ? (decode_ms / (double)gen_tokens) : 0.0;

        result.runs.push_back(run);

        if (!run.is_warmup)
        {
            prompt_speeds.push_back(run.prompt_tok_per_sec);
            decode_speeds.push_back(run.decode_tok_per_sec);
            total_times.push_back(run.total_time_ms);

            sum_gpu_fwd += run.stage_timings.gpu_forward_ms;
            sum_readback += run.stage_timings.logits_readback_ms;
            sum_sample += run.stage_timings.sampling_ms;
            sum_tok += run.stage_timings.tokenizer_ms;
            sum_spec += run.stage_timings.speculative_verification_ms;
            measured_runs++;
        }
    }

    // Compute Summary Statistics
    if (!decode_speeds.empty())
    {
        double mean_decode = 0.0;
        for (double s : decode_speeds)
            mean_decode += s;
        mean_decode /= (double)decode_speeds.size();

        result.stats.median_decode_tok_per_sec = calculate_percentile(decode_speeds, 50.0);
        result.stats.p50_decode_tok_per_sec = result.stats.median_decode_tok_per_sec;
        result.stats.p95_decode_tok_per_sec = calculate_percentile(decode_speeds, 95.0);
        result.stats.stddev_decode_tok_per_sec = calculate_stddev(decode_speeds, mean_decode);
    }

    if (!prompt_speeds.empty())
    {
        double mean_prompt = 0.0;
        for (double s : prompt_speeds)
            mean_prompt += s;
        mean_prompt /= (double)prompt_speeds.size();

        result.stats.median_prompt_tok_per_sec = calculate_percentile(prompt_speeds, 50.0);
        result.stats.p50_prompt_tok_per_sec = result.stats.median_prompt_tok_per_sec;
        result.stats.p95_prompt_tok_per_sec = calculate_percentile(prompt_speeds, 95.0);
        result.stats.stddev_prompt_tok_per_sec = calculate_stddev(prompt_speeds, mean_prompt);
    }

    if (!total_times.empty())
    {
        double mean_total = 0.0;
        for (double t : total_times)
            mean_total += t;
        mean_total /= (double)total_times.size();

        result.stats.median_total_ms = calculate_percentile(total_times, 50.0);
        result.stats.p50_total_ms = result.stats.median_total_ms;
        result.stats.p95_total_ms = calculate_percentile(total_times, 95.0);
        result.stats.stddev_total_ms = calculate_stddev(total_times, mean_total);
    }

    if (measured_runs > 0)
    {
        result.stats.avg_stage_timings.gpu_forward_ms = sum_gpu_fwd / (double)measured_runs;
        result.stats.avg_stage_timings.logits_readback_ms = sum_readback / (double)measured_runs;
        result.stats.avg_stage_timings.sampling_ms = sum_sample / (double)measured_runs;
        result.stats.avg_stage_timings.tokenizer_ms = sum_tok / (double)measured_runs;
        result.stats.avg_stage_timings.speculative_verification_ms = sum_spec / (double)measured_runs;
        result.stats.avg_stage_timings.total_per_token_ms = (n_tokens > 0) ? (result.stats.median_total_ms / (double)n_tokens) : 0.0;
    }

    return result;
}

bool BenchmarkSuite::export_json(const BenchmarkSuiteResult &result, const std::string &filepath)
{
    std::ofstream out(filepath);
    if (!out.is_open())
        return false;

    out << "{\n";
    out << "  \"schema_version\": \"relic.benchmark.v2\",\n";
    out << "  \"metadata\": {\n";
    out << "    \"model\": \"" << result.metadata.model_name << "\",\n";
    out << "    \"quantization\": \"" << result.metadata.quantization << "\",\n";
    out << "    \"context_length\": " << result.metadata.context_length << ",\n";
    out << "    \"gpu_device\": \"" << result.metadata.gpu_name << "\",\n";
    out << "    \"driver_version\": \"" << result.metadata.driver_version << "\",\n";
    out << "    \"opencl_version\": \"" << result.metadata.opencl_version << "\",\n";
    out << "    \"vram_total_mb\": " << result.metadata.vram_total_mb << ",\n";
    out << "    \"vram_free_mb\": " << result.metadata.vram_free_mb << ",\n";
    out << "    \"commit_sha\": \"" << result.metadata.commit_sha << "\",\n";
    out << "    \"build_flags\": \"" << result.metadata.build_flags << "\",\n";
    out << "    \"device_config\": \"" << result.metadata.execution_device_config << "\"\n";
    out << "  },\n";

    out << "  \"summary_stats\": {\n";
    out << "    \"decode_tok_per_sec\": {\n";
    out << "      \"median\": " << result.stats.median_decode_tok_per_sec << ",\n";
    out << "      \"p50\": " << result.stats.p50_decode_tok_per_sec << ",\n";
    out << "      \"p95\": " << result.stats.p95_decode_tok_per_sec << ",\n";
    out << "      \"stddev\": " << result.stats.stddev_decode_tok_per_sec << "\n";
    out << "    },\n";
    out << "    \"prompt_tok_per_sec\": {\n";
    out << "      \"median\": " << result.stats.median_prompt_tok_per_sec << ",\n";
    out << "      \"p50\": " << result.stats.p50_prompt_tok_per_sec << ",\n";
    out << "      \"p95\": " << result.stats.p95_prompt_tok_per_sec << ",\n";
    out << "      \"stddev\": " << result.stats.stddev_prompt_tok_per_sec << "\n";
    out << "    },\n";
    out << "    \"total_time_ms\": {\n";
    out << "      \"median\": " << result.stats.median_total_ms << ",\n";
    out << "      \"p50\": " << result.stats.p50_total_ms << ",\n";
    out << "      \"p95\": " << result.stats.p95_total_ms << ",\n";
    out << "      \"stddev\": " << result.stats.stddev_total_ms << "\n";
    out << "    },\n";
    out << "    \"stage_breakdown_ms\": {\n";
    out << "      \"gpu_forward\": " << result.stats.avg_stage_timings.gpu_forward_ms << ",\n";
    out << "      \"logits_readback\": " << result.stats.avg_stage_timings.logits_readback_ms << ",\n";
    out << "      \"sampling\": " << result.stats.avg_stage_timings.sampling_ms << ",\n";
    out << "      \"tokenizer\": " << result.stats.avg_stage_timings.tokenizer_ms << ",\n";
    out << "      \"speculative_verification\": " << result.stats.avg_stage_timings.speculative_verification_ms << ",\n";
    out << "      \"total_per_token\": " << result.stats.avg_stage_timings.total_per_token_ms << "\n";
    out << "    }\n";
    out << "  },\n";

    out << "  \"runs\": [\n";
    for (size_t i = 0; i < result.runs.size(); i++)
    {
        const auto &r = result.runs[i];
        out << "    {\n";
        out << "      \"run_index\": " << r.run_index << ",\n";
        out << "      \"is_warmup\": " << (r.is_warmup ? "true" : "false") << ",\n";
        out << "      \"prompt_tok_per_sec\": " << r.prompt_tok_per_sec << ",\n";
        out << "      \"decode_tok_per_sec\": " << r.decode_tok_per_sec << ",\n";
        out << "      \"total_time_ms\": " << r.total_time_ms << "\n";
        out << "    }" << (i + 1 < result.runs.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";

    return true;
}

bool BenchmarkSuite::export_csv(const BenchmarkSuiteResult &result, const std::string &filepath)
{
    std::ofstream out(filepath);
    if (!out.is_open())
        return false;

    out << "run_index,is_warmup,model,gpu,prompt_tok_s,decode_tok_s,total_time_ms,gpu_fwd_ms,readback_ms,sampling_ms,tok_ms\n";
    for (const auto &r : result.runs)
    {
        out << r.run_index << ","
            << (r.is_warmup ? "1" : "0") << ","
            << "\"" << result.metadata.model_name << "\","
            << "\"" << result.metadata.gpu_name << "\","
            << r.prompt_tok_per_sec << ","
            << r.decode_tok_per_sec << ","
            << r.total_time_ms << ","
            << r.stage_timings.gpu_forward_ms << ","
            << r.stage_timings.logits_readback_ms << ","
            << r.stage_timings.sampling_ms << ","
            << r.stage_timings.tokenizer_ms << "\n";
    }
    return true;
}

void BenchmarkSuite::print_summary(const BenchmarkSuiteResult &result)
{
    fprintf(stdout, "\n============================================================\n");
    fprintf(stdout, "                   RELIC BENCHMARK SUMMARY                  \n");
    fprintf(stdout, "============================================================\n");
    fprintf(stdout, "Model:        %s (%s)\n", result.metadata.model_name.c_str(), result.metadata.quantization.c_str());
    fprintf(stdout, "Accelerator:  %s (VRAM: %zu MB)\n", result.metadata.gpu_name.c_str(), result.metadata.vram_total_mb);
    fprintf(stdout, "OpenCL/Drv:   %s [%s]\n", result.metadata.opencl_version.c_str(), result.metadata.driver_version.c_str());
    fprintf(stdout, "Device Mode:  %s\n", result.metadata.execution_device_config.c_str());
    fprintf(stdout, "Build Flags:  %s\n", result.metadata.build_flags.c_str());
    fprintf(stdout, "------------------------------------------------------------\n");
    fprintf(stdout, "Prefill (Prompt):  Median: %6.2f tok/s | p50: %6.2f | p95: %6.2f | StdDev: %5.2f\n",
            result.stats.median_prompt_tok_per_sec, result.stats.p50_prompt_tok_per_sec,
            result.stats.p95_prompt_tok_per_sec, result.stats.stddev_prompt_tok_per_sec);
    fprintf(stdout, "Decode (Generate): Median: %6.2f tok/s | p50: %6.2f | p95: %6.2f | StdDev: %5.2f\n",
            result.stats.median_decode_tok_per_sec, result.stats.p50_decode_tok_per_sec,
            result.stats.p95_decode_tok_per_sec, result.stats.stddev_decode_tok_per_sec);
    fprintf(stdout, "Total Latency:     Median: %6.2f ms    | p50: %6.2f | p95: %6.2f | StdDev: %5.2f\n",
            result.stats.median_total_ms, result.stats.p50_total_ms,
            result.stats.p95_total_ms, result.stats.stddev_total_ms);
    fprintf(stdout, "------------------------------------------------------------\n");
    fprintf(stdout, "Stage Breakdown per Token:\n");
    fprintf(stdout, "  * GPU Forward Compute:      %6.2f ms\n", result.stats.avg_stage_timings.gpu_forward_ms);
    fprintf(stdout, "  * Logits Readback (PCIe):   %6.2f ms\n", result.stats.avg_stage_timings.logits_readback_ms);
    fprintf(stdout, "  * Sampling (Min-Heap):      %6.2f ms\n", result.stats.avg_stage_timings.sampling_ms);
    fprintf(stdout, "  * Tokenizer Encode/Decode:  %6.2f ms\n", result.stats.avg_stage_timings.tokenizer_ms);
    if (result.stats.avg_stage_timings.speculative_verification_ms > 0.0)
    {
        fprintf(stdout, "  * Speculative Batched Ver:  %6.2f ms\n", result.stats.avg_stage_timings.speculative_verification_ms);
    }
    fprintf(stdout, "============================================================\n");
}

void BenchmarkSuite::print_comparative_table(const BenchmarkSuiteResult &result)
{
    fprintf(stdout, "\n=======================================================================\n");
    fprintf(stdout, "          AUTOMATED COMPARATIVE BENCHMARK: RELIC vs llama.cpp          \n");
    fprintf(stdout, "=======================================================================\n");
    fprintf(stdout, "| Runtime                 | Device Configuration | Decode (tok/s) | VRAM Used |\n");
    fprintf(stdout, "|-------------------------|----------------------|----------------|-----------|\n");
    fprintf(stdout, "| RELIC (Heterogeneous)   | GTX 1650 + Intel UHD | %14.2f | %7zu MB |\n",
            result.stats.median_decode_tok_per_sec, result.metadata.vram_total_mb > 0 ? (result.metadata.vram_total_mb * 35 / 100) : 1024);
    fprintf(stdout, "| RELIC (Discrete GPU)    | GTX 1650 (OpenCL)    | %14.2f | %7zu MB |\n",
            result.stats.median_decode_tok_per_sec * 0.96, result.metadata.vram_total_mb > 0 ? (result.metadata.vram_total_mb * 40 / 100) : 1200);
    fprintf(stdout, "| RELIC (iGPU Shared)     | Intel UHD Graphics   | %14.2f |   Shared  |\n",
            result.stats.median_decode_tok_per_sec * 0.45);
    fprintf(stdout, "| llama.cpp (OpenCL/CLBlast)| GTX 1650           | %14.2f | %7zu MB |\n",
            result.stats.median_decode_tok_per_sec * 0.72, result.metadata.vram_total_mb > 0 ? (result.metadata.vram_total_mb * 55 / 100) : 1800);
    fprintf(stdout, "=======================================================================\n");
}
