#include "cli/cli_options.h"
#include "cli/benchmark_command.h"
#include "cli/inference_command.h"
#include "cli/server_command.h"
#include "gguf_reader.h"
#include "model.h"
#include "opencl_backend.h"
#include "tokenizer.h"
#include "inference.h"
#include "planner/hardware_profile.h"
#include "planner/adaptive_planner.h"
#include <cstdio>

int main(int argc, char **argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    CliOptions opt = parse_cli_options(argc, argv);

    if (opt.show_help)
    {
        print_usage(argv[0]);
        return 0;
    }

    if (opt.client_mode)
    {
        return ServerCommand::execute_client(opt);
    }

    if (opt.run_profile)
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

    if (opt.list_devices)
    {
        fprintf(stdout, "Available OpenCL platforms & devices:\n");
        OpenClBackend::list_devices();
        return 0;
    }

    if (opt.model_path.empty())
    {
        print_usage(argv[0]);
        return 1;
    }

    // Load model
    fprintf(stdout, "Loading model...\n");
    NeuralModel model;
    if (!model.load(opt.model_path.c_str()))
    {
        fprintf(stderr, "Failed to load model from %s\n", opt.model_path.c_str());
        return 1;
    }

    // Load tokenizer
    GgufReader reader;
    if (!reader.load(opt.model_path.c_str()))
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
    if (!opt.cpu_only)
    {
        cl_ok = cl.init(opt.platform_idx, opt.device_idx);
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

    if (opt.run_ablation)
    {
        return BenchmarkCommand::execute_ablation(opt, model, tokenizer, cl_ok ? &cl : nullptr, cl_ok, prof);
    }

    if (opt.run_cliff_analysis)
    {
        return BenchmarkCommand::execute_cliff_analysis(opt, model, tokenizer, cl_ok ? &cl : nullptr, cl_ok, prof);
    }

    if (opt.run_sweep)
    {
        return BenchmarkCommand::execute_sweep(opt, model, tokenizer, cl_ok ? &cl : nullptr, cl_ok, prof);
    }

    // Run Adaptive Planner to determine optimal tensor placement
    size_t vram_budget = (opt.vram_budget_mb > 0) ? (size_t)opt.vram_budget_mb * 1024 * 1024 : (cl_ok ? cl.dev.global_mem : 0);
    ExecutionPlan plan = AdaptivePlanner::generate_plan(model, prof, vram_budget, true, true);
    fprintf(stdout, "\n[Adaptive Planner] VRAM Budget: %.2f MB | Required: %.2f MB | Fully Offloaded Layers: %d/%lld\n",
            (double)vram_budget / (1024.0 * 1024.0),
            (double)plan.vram_required_bytes / (1024.0 * 1024.0),
            plan.num_layers_fully_offloaded, (long long)model.n_layer);
    fprintf(stdout, "[Adaptive Planner] Footprint Reduction: %.1f%% | PCIe Traffic Reduction: %.1f%% | Est. DMA Overlap: %.1f%%\n",
            plan.vram_footprint_reduction_pct, plan.pcie_traffic_reduction_pct, plan.estimated_dma_overlap_efficiency);

    // Initialize inference engine with ExecutionPlan
    InferenceEngine engine;
    engine.enable_speculative = opt.speculative;
    engine.speculative_ngram = opt.speculative_ngram;
    engine.speculative_max_draft = opt.speculative_draft_max;

    if (!engine.init(&model, &tokenizer, cl_ok ? &cl : nullptr, opt.max_seq_len, &plan))
    {
        fprintf(stderr, "Failed to initialize inference engine\n");
        return 1;
    }

    int ret = 0;
    if (opt.run_bench)
    {
        ret = BenchmarkCommand::execute_standard_bench(opt, engine);
    }
    else if (opt.server_mode)
    {
        ret = ServerCommand::execute_server(opt, engine);
    }
    else if (opt.interactive)
    {
        ret = InferenceCommand::execute_interactive(opt, engine);
    }
    else
    {
        ret = InferenceCommand::execute_generate(opt, engine);
    }

    engine.free_buffers();
    fflush(stdout);
    return ret;
}
