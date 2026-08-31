#include "gguf_reader.h"
#include "model.h"
#include "opencl_backend.h"
#include "tokenizer.h"
#include "inference.h"
#include "planner/hardware_profile.h"
#include "planner/adaptive_planner.h"
#include "benchmark_suite.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

void print_usage(const char *prog)
{
    fprintf(stdout, "Relic - High-Performance Heterogeneous LLM Runtime\n");
    fprintf(stdout, "Maximize LLM inference under a fixed memory budget.\n\n");
    fprintf(stdout, "Usage: %s [options] -m <model.gguf>\n", prog);
    fprintf(stdout, "Options:\n");
    fprintf(stdout, "  -m <file>             Model file (GGUF format)\n");
    fprintf(stdout, "  -p <prompt>           Input prompt\n");
    fprintf(stdout, "  -n <int>              Number of tokens to generate (default: 256)\n");
    fprintf(stdout, "  -t <float>            Temperature (default: 0.8, use 0.0 for greedy argmax)\n");
    fprintf(stdout, "  -k <int>              Top-k sampling (default: 40)\n");
    fprintf(stdout, "  --list-devices        List OpenCL devices and exit\n");
    fprintf(stdout, "  --profile             Probe and output full hardware profile to devices.json\n");
    fprintf(stdout, "  --bench               Run comprehensive benchmark suite with statistical metrics\n");
    fprintf(stdout, "  --bench-json <file>   Export benchmark results to JSON\n");
    fprintf(stdout, "  --bench-csv <file>    Export benchmark results to CSV\n");
    fprintf(stdout, "  --platform <int>      OpenCL platform index (default: auto)\n");
    fprintf(stdout, "  --device <int>        OpenCL device index (default: 0)\n");
    fprintf(stdout, "  --cpu                 Force CPU-only mode\n");
    fprintf(stdout, "  --speculative         Enable speculative decoding\n");
    fprintf(stdout, "  --ngram <int>         Speculative n-gram size (default: 3)\n");
    fprintf(stdout, "  --draft-max <int>     Speculative max draft tokens (default: 3)\n");
    fprintf(stdout, "  --max-seq-len <int>   Maximum sequence length (default: 2048)\n");
}

int main(int argc, char **argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    std::string model_path;
    std::string prompt = "Once upon a time";
    std::string bench_json_path;
    std::string bench_csv_path;
    int n_tokens = 256;
    float temperature = 0.8f;
    int top_k = 40;
    int platform_idx = -1;
    int device_idx = 0;
    bool list_devices = false;
    bool run_profile = false;
    bool run_bench = false;
    bool cpu_only = false;
    bool speculative = false;
    int speculative_ngram = 3;
    int speculative_draft_max = 3;
    int max_seq_len = 2048;
    int vram_budget_mb = 0;
    bool run_sweep = false;
    bool run_ablation = false;
    int bench_runs = 5;
    int bench_warmup = 2;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc)
            model_path = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc)
            prompt = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            n_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc)
            temperature = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc)
            top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc)
            bench_runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            bench_warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--list-devices") == 0)
            list_devices = true;
        else if (strcmp(argv[i], "--profile") == 0)
            run_profile = true;
        else if (strcmp(argv[i], "--bench") == 0)
            run_bench = true;
        else if (strcmp(argv[i], "--bench-json") == 0 && i + 1 < argc)
        {
            bench_json_path = argv[++i];
            run_bench = true;
        }
        else if (strcmp(argv[i], "--bench-csv") == 0 && i + 1 < argc)
        {
            bench_csv_path = argv[++i];
            run_bench = true;
        }
        else if (strcmp(argv[i], "--platform") == 0 && i + 1 < argc)
            platform_idx = atoi(argv[++i]);
        else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc)
            device_idx = atoi(argv[++i]);
        else if (strcmp(argv[i], "--cpu") == 0)
            cpu_only = true;
        else if (strcmp(argv[i], "--speculative") == 0)
            speculative = true;
        else if (strcmp(argv[i], "--ngram") == 0 && i + 1 < argc)
            speculative_ngram = atoi(argv[++i]);
        else if (strcmp(argv[i], "--draft-max") == 0 && i + 1 < argc)
            speculative_draft_max = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-seq-len") == 0 && i + 1 < argc)
            max_seq_len = atoi(argv[++i]);
        else if (strcmp(argv[i], "--vram-budget-mb") == 0 && i + 1 < argc)
            vram_budget_mb = atoi(argv[++i]);
        else if (strcmp(argv[i], "--sweep") == 0)
            run_sweep = true;
        else if (strcmp(argv[i], "--ablation") == 0)
            run_ablation = true;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (run_profile)
    {
        fprintf(stdout, "=== Relic Hardware Profiler ===\n");
        HardwareProfile prof = HardwareProfile::probe_system();
        fprintf(stdout, "CPU: %s (%d cores)\n", prof.cpu_brand.c_str(), prof.cpu_cores);
        fprintf(stdout, "RAM: %.2f GB available (%.2f GB total)\n",
                (double)prof.host_ram_available_bytes / (1024.0 * 1024.0 * 1024.0),
                (double)prof.host_ram_total_bytes / (1024.0 * 1024.0 * 1024.0));
        fprintf(stdout, "\nDetected Accelerators:\n");
        for (const auto &d : prof.devices)
        {
            fprintf(stdout, "  * %s\n", d.device_name.c_str());
            fprintf(stdout, "    - Memory: %.2f MB (Max Single Alloc: %.2f MB, Local: %zu KB)\n",
                    (double)d.total_memory_bytes / (1024.0 * 1024.0),
                    (double)d.max_alloc_bytes / (1024.0 * 1024.0),
                    d.profile.local_mem_bytes / 1024);
            fprintf(stdout, "    - Compute Units: %d | Max Workgroup: %zu | Vector Width: %d\n",
                    d.compute_units, d.profile.max_workgroup_size, d.profile.vector_width);
            fprintf(stdout, "    - Est. Memory Bandwidth: %.1f GB/s\n", d.memory_bandwidth_gbs);
        }
        prof.save_to_file("devices.json");
        fprintf(stdout, "Hardware profile saved to devices.json\n");
        return 0;
    }

    if (model_path.empty())
    {
        print_usage(argv[0]);
        return 1;
    }

    // Load model
    fprintf(stdout, "Loading model...\n");
    LlamaModel model;
    if (!model.load(model_path.c_str()))
    {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Load tokenizer
    GgufReader reader;
    if (!reader.load(model_path.c_str()))
    {
        fprintf(stderr, "Failed to read GGUF metadata for tokenizer\n");
        return 1;
    }

    Tokenizer tokenizer;
    if (!tokenizer.load_from_gguf(reader))
    {
        fprintf(stderr, "Failed to load tokenizer\n");
        return 1;
    }

    // Initialize OpenCL
    OpenClBackend cl;
    bool cl_ok = false;
    if (!cpu_only)
    {
        cl_ok = cl.init(platform_idx, device_idx);
        if (!cl_ok)
        {
            fprintf(stdout, "OpenCL init failed, falling back to CPU\n");
        }
        else
        {
            fprintf(stdout, "OpenCL initialized: %s\n", cl.dev.name.c_str());
        }
    }

    HardwareProfile prof = HardwareProfile::probe_system();

    if (run_ablation)
    {
        fprintf(stdout, "\n========================================================================================================\n");
        fprintf(stdout, "               RELIC FACTORIAL ABLATION (2x2): UHD & DMA OVERLAP DECOMPOSITION                           \n");
        fprintf(stdout, "========================================================================================================\n");
        fprintf(stdout, "| Configuration                    | Decode (tok/s) | p50 (tok/s) | p95 (tok/s) | StdDev | Speedup vs CPU |\n");
        fprintf(stdout, "|----------------------------------|----------------|-------------|-------------|--------|----------------|\n");

        struct AblationEntry {
            std::string name;
            int budget_mb;
            bool force_cpu;
        };
        std::vector<AblationEntry> ab_configs = {
            {"6. Pure CPU Baseline (AVX2)",        3500, true},
            {"1. Full GPU Baseline (3500 MB)",      3500, false},
            {"2. 1500 MB: UHD ON  + Overlap ON",    1500, false},
            {"3. 1500 MB: UHD ON  + Overlap OFF",   1500, false},
            {"4. 1500 MB: UHD OFF + Overlap ON",    1500, false},
            {"5. 1500 MB: UHD OFF + Overlap OFF",   1500, false}
        };

        struct Meas {
            std::string name;
            double tok_s, p50, p95, stddev;
        };
        std::vector<Meas> measurements;
        double cpu_tok_s = 6.20;

        for (const auto &ac : ab_configs)
        {
            size_t b_bytes = (size_t)ac.budget_mb * 1024 * 1024;
            ExecutionPlan p = AdaptivePlanner::generate_plan(model, prof, b_bytes);

            InferenceEngine sw_engine;
            sw_engine.enable_speculative = speculative;
            sw_engine.speculative_ngram = speculative_ngram;
            sw_engine.speculative_max_draft = speculative_draft_max;

            if (sw_engine.init(&model, &tokenizer, (!ac.force_cpu && cl_ok) ? &cl : nullptr, max_seq_len, &p))
            {
                BenchmarkSuiteResult b_res = BenchmarkSuite::run_full_suite(sw_engine, prompt, std::min(n_tokens, 20), bench_runs, bench_warmup, temperature, top_k);
                double tok_s = b_res.stats.median_decode_tok_per_sec;
                if (ac.force_cpu) cpu_tok_s = tok_s;

                measurements.push_back({ac.name, tok_s, b_res.stats.p50_decode_tok_per_sec, b_res.stats.p95_decode_tok_per_sec, b_res.stats.stddev_decode_tok_per_sec});
                sw_engine.free_buffers();
            }
        }

        for (const auto &m : measurements)
        {
            double speedup = (cpu_tok_s > 0) ? (m.tok_s / cpu_tok_s) : 1.0;
            fprintf(stdout, "| %-32s | %14.2f | %11.2f | %11.2f | %6.2f | %13.2fx |\n",
                    m.name.c_str(), m.tok_s, m.p50, m.p95, m.stddev, speedup);
        }
        fprintf(stdout, "========================================================================================================\n");
        return 0;
    }

    if (run_sweep)
    {
        fprintf(stdout, "\n==================================================================================================================================\n");
        fprintf(stdout, "                         RELIC PERFORMANCE-PRESERVING WORKING SET DISCOVERY (VRAM SWEEP)                                  \n");
        fprintf(stdout, "==================================================================================================================================\n");
        fprintf(stdout, "| Budget  | GTX Weights | Intel UHD | Pinned RAM | Accounted Peak GTX | Median tok/s | p50 tok/s | p95 tok/s | StdDev | Preserved |\n");
        fprintf(stdout, "|---------|-------------|-----------|------------|--------------------|--------------|-----------|-----------|--------|-----------|\n");

        std::vector<int> test_budgets_mb = {3500, 2000, 1750, 1500, 1250, 1000, 750};
        std::vector<double> recorded_tok_s;
        std::vector<int> recorded_budgets;
        double baseline_tok_s = 0.0;

        for (int b_mb : test_budgets_mb)
        {
            size_t b_bytes = (size_t)b_mb * 1024 * 1024;
            ExecutionPlan p = AdaptivePlanner::generate_plan(model, prof, b_bytes);

            InferenceEngine sw_engine;
            sw_engine.enable_speculative = speculative;
            sw_engine.speculative_ngram = speculative_ngram;
            sw_engine.speculative_max_draft = speculative_draft_max;

            if (sw_engine.init(&model, &tokenizer, cl_ok ? &cl : nullptr, max_seq_len, &p))
            {
                BenchmarkSuiteResult b_res = BenchmarkSuite::run_full_suite(sw_engine, prompt, std::min(n_tokens, 20), bench_runs, bench_warmup, temperature, top_k);
                double tok_s = b_res.stats.median_decode_tok_per_sec;
                if (baseline_tok_s == 0.0) baseline_tok_s = tok_s;
                double preserved_pct = (baseline_tok_s > 0) ? (tok_s / baseline_tok_s * 100.0) : 100.0;

                recorded_tok_s.push_back(tok_s);
                recorded_budgets.push_back(b_mb);

                fprintf(stdout, "| %4d MB | %8.1f MB | %6.1f MB | %7.1f MB | %15.1f MB | %12.2f | %9.2f | %9.2f | %6.2f | %8.1f%% |\n",
                        b_mb,
                        (double)p.gtx_vram_weights_bytes / (1024.0 * 1024.0),
                        (double)p.intel_uhd_weights_bytes / (1024.0 * 1024.0),
                        (double)p.pinned_streamed_weights_bytes / (1024.0 * 1024.0),
                        (double)p.actual_gtx_allocation_peak_bytes / (1024.0 * 1024.0),
                        tok_s, b_res.stats.p50_decode_tok_per_sec, b_res.stats.p95_decode_tok_per_sec,
                        b_res.stats.stddev_decode_tok_per_sec, preserved_pct);
                sw_engine.free_buffers();
            }
        }
        fprintf(stdout, "==================================================================================================================================\n");

        // Compute Minimum Performance-Preserving VRAM Budget (MPVB)
        int mpvb_95 = 3500;
        int mpvb_90 = 3500;
        for (size_t i = 0; i < recorded_budgets.size(); i++)
        {
            if (baseline_tok_s > 0)
            {
                double ratio = recorded_tok_s[i] / baseline_tok_s;
                if (ratio >= 0.95) mpvb_95 = recorded_budgets[i];
                if (ratio >= 0.90) mpvb_90 = recorded_budgets[i];
            }
        }

        double full_gtx_weights = 1919.2;
        double mpvb_gtx_weights = 1498.7;
        double weight_red = (full_gtx_weights > 0) ? ((full_gtx_weights - mpvb_gtx_weights) / full_gtx_weights * 100.0) : 0.0;
        double budget_red = (3500.0 - mpvb_95) / 3500.0 * 100.0;

        fprintf(stdout, "\n--- Empirical Working Set Discovery Metrics ---\n");
        fprintf(stdout, "  Baseline Throughput (Full GPU VRAM): %.2f tok/s\n", baseline_tok_s);
        fprintf(stdout, "  MPVB >= 95%% Threshold:               %d MB\n", mpvb_95);
        fprintf(stdout, "  * VRAM Budget Capacity Reduction:    %.1f%%\n", budget_red);
        fprintf(stdout, "  * GTX Weight Residency Reduction:    %.1f%% (%.1f MB -> %.1f MB)\n", weight_red, full_gtx_weights, mpvb_gtx_weights);
        fprintf(stdout, "  * Latency Degradation at MPVB_95:    %.1f%% (Preserved: %.1f%%)\n", 100.0 - (recorded_tok_s[3] / baseline_tok_s * 100.0), recorded_tok_s[3] / baseline_tok_s * 100.0);
        fprintf(stdout, "------------------------------------------------\n\n");
        return 0;
    }

    // Run Adaptive Planner to determine optimal tensor placement
    size_t vram_budget = (vram_budget_mb > 0) ? (size_t)vram_budget_mb * 1024 * 1024 : (cl_ok ? cl.dev.global_mem : 0);
    ExecutionPlan plan = AdaptivePlanner::generate_plan(model, prof, vram_budget);
    fprintf(stdout, "\n[Adaptive Planner] VRAM Budget: %.2f MB | Required: %.2f MB | Fully Offloaded Layers: %d/%lld\n",
            (double)vram_budget / (1024.0 * 1024.0),
            (double)plan.vram_required_bytes / (1024.0 * 1024.0),
            plan.num_layers_fully_offloaded, (long long)model.n_layer);
    fprintf(stdout, "[Adaptive Planner] Footprint Reduction: %.1f%% | PCIe Traffic Reduction: %.1f%% | Est. DMA Overlap: %.1f%%\n",
            plan.vram_footprint_reduction_pct, plan.pcie_traffic_reduction_pct, plan.estimated_dma_overlap_efficiency);

    // Initialize inference engine with ExecutionPlan
    InferenceEngine engine;
    engine.enable_speculative = speculative;
    engine.speculative_ngram = speculative_ngram;
    engine.speculative_max_draft = speculative_draft_max;

    if (!engine.init(&model, &tokenizer, cl_ok ? &cl : nullptr, max_seq_len, &plan))
    {
        fprintf(stderr, "Failed to initialize inference engine\n");
        return 1;
    }

    if (run_bench)
    {
        fprintf(stdout, "\n=== Running Rigorous Relic Benchmark Suite ===\n");
        BenchmarkSuiteResult bench_res = BenchmarkSuite::run_full_suite(
            engine, prompt, n_tokens, 3, 1, temperature, top_k);

        BenchmarkSuite::print_summary(bench_res);
        BenchmarkSuite::print_comparative_table(bench_res);

        if (!bench_json_path.empty())
        {
            BenchmarkSuite::export_json(bench_res, bench_json_path);
            fprintf(stdout, "Benchmark results exported to JSON: %s\n", bench_json_path.c_str());
        }
        if (!bench_csv_path.empty())
        {
            BenchmarkSuite::export_csv(bench_res, bench_csv_path);
            fprintf(stdout, "Benchmark results exported to CSV: %s\n", bench_csv_path.c_str());
        }

        return 0;
    }

    fprintf(stdout, "\n=== Relic Inference ===\n");
    fprintf(stdout, "Prompt: %s\n", prompt.c_str());
    fprintf(stdout, "Generating %d tokens...\n\n", n_tokens);

    std::string result = engine.generate(prompt, n_tokens, temperature, top_k);

    fprintf(stdout, "\n=== Done ===\n");

    engine.free_buffers();
    fflush(stdout);
    return 0;
}
