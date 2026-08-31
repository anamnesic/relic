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
            fprintf(stdout, "    - Est. DMA PCIe Bandwidth: %.1f GB/s\n", d.dma_transfer_bandwidth_gbs);
            fprintf(stdout, "    - Driver: %s | OpenCL C: %s\n\n", d.profile.driver_version.c_str(), d.profile.opencl_c_version.c_str());
        }
        prof.save_to_file("devices.json");
        fprintf(stdout, "Hardware profile saved to devices.json\n");
        return 0;
    }

    // List devices mode
    if (list_devices)
    {
        cl_uint n_platforms = 0;
        if (clGetPlatformIDs(0, nullptr, &n_platforms) != CL_SUCCESS || n_platforms == 0)
        {
            fprintf(stdout, "No OpenCL platforms found\n");
            return 0;
        }
        std::vector<cl_platform_id> platforms(n_platforms);
        if (clGetPlatformIDs(n_platforms, platforms.data(), nullptr) != CL_SUCCESS)
        {
            fprintf(stderr, "Failed to enumerate OpenCL platforms\n");
            return 1;
        }

        for (cl_uint p = 0; p < n_platforms; p++)
        {
            char name[1024];
            clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sizeof(name), name, nullptr);
            fprintf(stdout, "Platform %u: %s\n", p, name);

            cl_uint nd = 0;
            if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, nullptr, &nd) == CL_SUCCESS && nd > 0)
            {
                std::vector<cl_device_id> devs(nd);
                if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, nd, devs.data(), nullptr) != CL_SUCCESS)
                    continue;
                for (cl_uint d = 0; d < nd; d++)
                {
                    char dname[1024], version[128];
                    cl_device_type dtype;
                    cl_ulong mem;
                    clGetDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof(dname), dname, nullptr);
                    clGetDeviceInfo(devs[d], CL_DEVICE_VERSION, sizeof(version), version, nullptr);
                    clGetDeviceInfo(devs[d], CL_DEVICE_TYPE, sizeof(dtype), &dtype, nullptr);
                    clGetDeviceInfo(devs[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);
                    fprintf(stdout, "  Device %d: %s [%s] %zu MB %s\n",
                            d, dname, version, mem / (1024 * 1024),
                            (dtype & CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU");
                }
            }
        }
        return 0;
    }

    if (model_path.empty())
    {
        fprintf(stderr, "No model file specified. Use -m <model.gguf>\n");
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

    // Run Adaptive Planner to determine optimal tensor placement
    HardwareProfile prof = HardwareProfile::probe_system();
    size_t vram_budget = cl_ok ? cl.dev.global_mem : 0;
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
