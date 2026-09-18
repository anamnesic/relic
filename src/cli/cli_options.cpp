#include "cli_options.h"
#include "../opencl_backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

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
    fprintf(stdout, "  -i, --interactive     Keep model resident in GPU VRAM and enter interactive REPL\n");
    fprintf(stdout, "  --server [port]       Run as persistent VRAM HTTP/TCP server daemon (default: 8080)\n");
    fprintf(stdout, "  --client [port]       Query running persistent server without reloading model\n");
    fprintf(stdout, "  --max-seq-len <int>   Maximum sequence length (default: 2048)\n\n");
    fprintf(stdout, "Available OpenCL platforms & devices:\n");
    OpenClBackend::list_devices();
}

CliOptions parse_cli_options(int argc, char **argv)
{
    CliOptions opt;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc)
            opt.model_path = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc)
            opt.prompt = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            opt.n_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc)
            opt.temperature = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc)
            opt.top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--interactive") == 0)
            opt.interactive = true;
        else if (strcmp(argv[i], "--server") == 0)
        {
            opt.server_mode = true;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                opt.server_port = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--client") == 0)
        {
            opt.client_mode = true;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                opt.client_port = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--platform") == 0 && i + 1 < argc)
            opt.platform_idx = atoi(argv[++i]);
        else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc)
            opt.device_idx = atoi(argv[++i]);
        else if (strcmp(argv[i], "--list-devices") == 0)
            opt.list_devices = true;
        else if (strcmp(argv[i], "--profile") == 0)
            opt.run_profile = true;
        else if (strcmp(argv[i], "--bench") == 0)
            opt.run_bench = true;
        else if (strcmp(argv[i], "--bench-json") == 0 && i + 1 < argc)
            opt.bench_json_path = argv[++i];
        else if (strcmp(argv[i], "--bench-csv") == 0 && i + 1 < argc)
            opt.bench_csv_path = argv[++i];
        else if (strcmp(argv[i], "--cpu") == 0)
            opt.cpu_only = true;
        else if (strcmp(argv[i], "--speculative") == 0)
            opt.speculative = true;
        else if (strcmp(argv[i], "--ngram") == 0 && i + 1 < argc)
            opt.speculative_ngram = atoi(argv[++i]);
        else if (strcmp(argv[i], "--draft-max") == 0 && i + 1 < argc)
            opt.speculative_draft_max = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-seq-len") == 0 && i + 1 < argc)
            opt.max_seq_len = atoi(argv[++i]);
        else if (strcmp(argv[i], "--budget-mb") == 0 && i + 1 < argc)
            opt.vram_budget_mb = atoi(argv[++i]);
        else if (strcmp(argv[i], "--sweep") == 0)
            opt.run_sweep = true;
        else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc)
            opt.bench_runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            opt.bench_warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ablation") == 0)
            opt.run_ablation = true;
        else if (strcmp(argv[i], "--cliff-analysis") == 0)
            opt.run_cliff_analysis = true;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
            opt.show_help = true;
    }
    return opt;
}
